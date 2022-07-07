import copy
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from .custom_build_augmentation import build_custom_augmentation
import pickle
from utils.util import check_image_size

__all__ = ["CustomDatasetMapper", "CustomDatasetMapperMix"]


class CustomDatasetMapper(DatasetMapper):
    @configurable
    def __init__(self, is_train: bool,
                 distillation=False,
                 rkd_feat_path='',
                 num_distil_prop=5,
                 **kwargs):
        """
        add proposals for distillation
        Args:
            is_train: whether it's used in training or inference
            distillation: whether to use region-based-knowledge distillation
            proposal_path: path of dir containing pesudo-proposals
            num_distil_prop: number of proposals to consider from pseudo proposals
        """
        self.distillation = distillation
        self.rkd_feat_path = rkd_feat_path
        self.num_distil_prop = num_distil_prop
        super().__init__(is_train, **kwargs)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            "rkd_feat_path": cfg.MODEL.RKD_FEAT_PATH,
            "distillation": cfg.MODEL.DISTILLATION,
            "num_distil_prop": cfg.MODEL.NUM_DISTIL_PROP
        })
        return ret

    def __call__(self, dataset_dict):
        """
        include pseudo distillation embeddings
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        dataset_dict["width"], dataset_dict["height"] = image.shape[1], image.shape[0]

        sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w

        # Modification made for MViT
        # Loading CLIP features for RKD (generated using MAVL proposals) with image in dataloader
        if self.distillation and self.is_train and len(self.rkd_feat_path) > 0:
            image_name = dataset_dict["file_name"].split('.')[0].split('/')[-1]
            # load predictions with dataloader only in training
            clip_features_file = f'{self.rkd_feat_path}/{image_name}.pkl'
            with open(clip_features_file, "rb") as c:
                distill_feats = pickle.load(c)
            # process proposal boxes for distillation
            top_n = self.num_distil_prop
            region_boxes = []
            clip_embeds = []
            for p in range(len(distill_feats)):
                box = transforms.apply_box(np.array([distill_feats[p][0]]))[0].clip(min=0).tolist()
                box = np.minimum(box, list(image_shape + image_shape)[::-1])
                region_boxes.append(box)
                clip_embeds.append(distill_feats[p][1])
            # select n based on num features to distill
            region_boxes = region_boxes[0: top_n]
            clip_embeds = clip_embeds[0: top_n]
            region_boxes = torch.tensor(np.array(region_boxes))
            clip_embeds = torch.cat(clip_embeds, 0)
            processed_distill_feats = (region_boxes, clip_embeds)
            dataset_dict["distill_feats"] = processed_distill_feats

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class CustomDatasetMapperMix(DatasetMapper):
    @configurable
    def __init__(self, is_train: bool,
                 with_ann_type=False,
                 dataset_ann=[],
                 use_diff_bs_size=False,
                 dataset_augs=[],
                 pis_proposal_path='',
                 rkd_feat_path='',
                 rkd_ils_feath_path='',
                 distillation=False,
                 num_distil_prop=5,
                 **kwargs):
        """
        add image labels
        """
        self.with_ann_type = with_ann_type
        self.dataset_ann = dataset_ann
        self.use_diff_bs_size = use_diff_bs_size
        if self.use_diff_bs_size and is_train:
            self.dataset_augs = [T.AugmentationList(x) for x in dataset_augs]
        self.ann_type = 'box'
        self.pis_proposal_path = pis_proposal_path
        self.rkd_feat_path = rkd_feat_path
        self.rkd_ils_feath_path = rkd_ils_feath_path
        self.distillation = distillation
        self.num_distil_prop = num_distil_prop
        super().__init__(is_train, **kwargs)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            'with_ann_type': cfg.WITH_IMAGE_LABELS,
            'dataset_ann': cfg.DATALOADER.DATASET_ANN,
            'use_diff_bs_size': cfg.DATALOADER.USE_DIFF_BS_SIZE,
            'pis_proposal_path': cfg.MODEL.PIS_PROP_PATH,
            'rkd_feat_path': cfg.MODEL.RKD_FEAT_PATH,
            'rkd_ils_feath_path': cfg.MODEL.RKD_ILS_FEAT_PATH,
            'distillation': cfg.MODEL.DISTILLATION,
            'num_distil_prop': cfg.MODEL.NUM_DISTIL_PROP,

        })
        if ret['use_diff_bs_size'] and is_train:
            assert cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge'
            min_sizes = cfg.DATALOADER.DATASET_MIN_SIZES
            max_sizes = cfg.DATALOADER.DATASET_MAX_SIZES
            ret['dataset_augs'] = [
                build_custom_augmentation(
                    cfg, True, min_size=mi, max_size=ma) \
                for mi, ma in zip(min_sizes, max_sizes)]
        else:
            ret['dataset_augs'] = []

        return ret

    def __call__(self, dataset_dict):
        """
        include image labels
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        if 'file_name' in dataset_dict:
            ori_image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format)
        check_image_size(dataset_dict, ori_image)

        sem_seg_gt = None
        aug_input = T.AugInput(copy.deepcopy(ori_image), sem_seg=sem_seg_gt)
        if self.use_diff_bs_size and self.is_train:
            self.ann_type = 'box' if dataset_dict['dataset_source'] == 0 else 'image'
            transforms = \
                self.dataset_augs[dataset_dict['dataset_source']](aug_input)
        else:
            transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w

        # Modification made for MViT psuedo labelling
        # Loading MViT proposals from annotation
        if self.ann_type == 'image' and self.is_train and len(self.pis_proposal_path) == 0:
            assert 'annotations' in dataset_dict.keys()
            dataset_dict['transforms'] = transforms
            dataset_dict['shape'] = image_shape
            dataset_dict['key_bbox'] = (dataset_dict['annotations'][0]['category_id'],
                                        dataset_dict['annotations'][0]['bbox'])

        # Loading MViT proposals with image in dataloader
        if self.ann_type == 'image' and self.is_train and len(self.pis_proposal_path) > 0:
            catid2contid = dataset_dict["catid2contid"]
            image_name = dataset_dict["file_name"].split('.')[0].split('/')[-1]
            proposal_file = f'{self.pis_proposal_path}/{image_name}.pkl'
            with open(proposal_file, "rb") as f:
                detections = pickle.load(f)
            target_keys = detections.keys()
            boxes = []
            probas = []
            oredered_detections = {}
            for k in target_keys:
                # for each target apply transforms
                box, prob = detections[k]
                box = transforms.apply_box(np.array([box[0]]))[0].clip(min=0).tolist()
                box = np.minimum(box, list(image_shape + image_shape)[::-1])
                boxes.append(box)
                probas.append(prob[0])
                # create new dict of mutated label ids
                new_key = "salient" if k == "salient" else catid2contid[k]
                oredered_detections[new_key] = box, prob
            dataset_dict["cls_specific_props"] = torch.tensor(np.array(boxes))
            dataset_dict["cls_specific_scores"] = torch.tensor(np.array(probas))
            dataset_dict["cls_specific_target_props"] = oredered_detections

        # Loading RKD features
        if self.distillation and self.is_train and len(self.rkd_feat_path) > 0:
            image_name = dataset_dict["file_name"].split('.')[0].split('/')[-1]
            if len(self.rkd_ils_feath_path) > 0 and self.ann_type == 'image':  # trick to handle the case IMAGENET-LVIS
                folder_name = image_name.split("_")[0]
                clip_features_file = f'{self.rkd_ils_feath_path}/{folder_name}/{image_name}.pkl'
            else:
                clip_features_file = f'{self.rkd_feat_path}/{image_name}.pkl'
            with open(clip_features_file, "rb") as c:
                distill_feats = pickle.load(c)
            # process proposal boxes for distillation
            region_boxes = []
            clip_embeds = []
            top_n = self.num_distil_prop
            for p in range(len(distill_feats)):
                box = transforms.apply_box(np.array([distill_feats[p][0]]))[0].clip(min=0).tolist()
                box = np.minimum(box, list(image_shape + image_shape)[::-1])
                region_boxes.append(box)
                clip_embeds.append(distill_feats[p][1])
            # select n based on num features to distill
            region_boxes = region_boxes[0: top_n]
            clip_embeds = clip_embeds[0: top_n]
            region_boxes = torch.tensor(np.array(region_boxes))
            clip_embeds = torch.cat(clip_embeds, 0)
            processed_distill_feats = (region_boxes, clip_embeds)
            dataset_dict["distill_feats"] = processed_distill_feats

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            all_annos = [
                (utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                ), obj.get("iscrowd", 0))
                for obj in dataset_dict.pop("annotations")
            ]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            del all_annos
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        if self.with_ann_type:
            dataset_dict["pos_category_ids"] = dataset_dict.get(
                'pos_category_ids', [])
            dataset_dict["ann_type"] = \
                self.dataset_ann[dataset_dict['dataset_source']]
        return dataset_dict
