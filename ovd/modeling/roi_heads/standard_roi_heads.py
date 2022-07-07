from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .custom_fast_rcnn import CustomFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        out_channels = cfg.MODEL.ROI_BOX_HEAD.FC_DIM

        self.with_image_labels = cfg.WITH_IMAGE_LABELS
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.box_predictor = CustomFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def forward(self, images, features, proposals, targets=None, ann_type='box'):
        del images

        features, distill_clip_features = features
        if self.training:
            if ann_type == 'box':
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

        features_box = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features_box, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        if self.training and distill_clip_features is not None:
            # distilling image embedding
            distil_regions, distill_clip_embeds = distill_clip_features
            region_level_features = self.box_pooler(features_box, distil_regions)
            image_embeds = self.box_head(region_level_features)
            # image distillation
            proj_image_embeds = self.box_predictor.cls_score.linear(image_embeds)
            norm_image_embeds = F.normalize(proj_image_embeds, p=2, dim=1)
            normalized_clip_embeds = F.normalize(distill_clip_embeds, p=2, dim=1)
            distill_features = (norm_image_embeds, normalized_clip_embeds)
        else:
            distill_features = None

        if self.training:
            del features_box
            if ann_type != 'box':
                image_labels = [x._pos_category_ids for x in targets]
                losses = self.box_predictor.image_label_losses(
                    predictions, proposals, distill_features, image_labels)
            else:
                losses = self.box_predictor.losses(
                    (predictions[0], predictions[1]), proposals, distill_features)
                # Calculate the loss for mask predictions
                losses.update(self._forward_mask(features, proposals))
                if self.with_image_labels:
                    assert 'pms_loss' not in losses
                    losses['pms_loss'] = predictions[0].new_zeros([1])[0]
            return proposals, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        return proposals
