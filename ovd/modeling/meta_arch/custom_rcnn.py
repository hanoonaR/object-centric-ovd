from typing import Dict, List, Optional, Tuple
import torch
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import Instances, Boxes
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from torch.cuda.amp import autocast


@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    """
    Add image labels
    """

    @configurable
    def __init__(
            self,
            fp16=False,
            roi_head_name='',
            distillation=False,
            **kwargs):
        """
        """
        self.roi_head_name = roi_head_name
        self.return_proposal = False
        self.fp16 = fp16
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        self.distillation = distillation

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'fp16': cfg.FP16,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'distillation': cfg.MODEL.DISTILLATION,
        })
        return ret

    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, (features, None), proposals)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        if self.fp16:
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)

        if self.distillation:
            distill_clip_features = self.get_clip_image_features(batched_inputs, images)
        else:
            distill_clip_features = None

        proposals, detector_losses = self.roi_heads(
            images, (features, distill_clip_features), proposals, gt_instances, ann_type=ann_type)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.return_proposal:
            return proposals, losses
        else:
            return losses

    def get_clip_image_features(self, batched_inputs, images):
        image_features = []
        region_boxes = []
        for n, image_size in enumerate(images.image_sizes):
            image_features.append(batched_inputs[n]['distill_feats'][1].to(images[n].device))
            region_boxes.append(Boxes(batched_inputs[n]['distill_feats'][0].to(images[n].device)))
        image_features = torch.cat(image_features, 0)
        return region_boxes, image_features
