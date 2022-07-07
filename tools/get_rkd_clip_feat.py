"""
This script uses MAVL from paper https://arxiv.org/abs/2111.11430 to generate class-agnostic boxes followed by
corresponding CLIP image features for region-based  knowledge distillation (RKD) in our approach. Specifically,
we select top-5 proposals from combined detections of multiple text queries including 'all objects', 'all entities',
'all visible objects and entities' and 'all obscure objects and entities', and extract CLIP feature for each proposal.

Examples:
    - Clone the external modules:
        git submodule init
        git submodule update
    - Set the PYTHONPATH:
        export PYTHONPATH="./external/mavl:$PYTHONPATH"
    - Run
    For COCO:
        python tools/get_rkd_clip_feat.py -ckpt <mavl pretrained weights path> -dataset coco -dataset_dir datasets/coco
        -output datasets/MAVL_proposals/coco_props/classagnostic_distilfeats
    For LVIS:
        # Use the same features generated from COCO, as the images are same and predictions are class-agnostic
        ln -s datasets/MAVL_proposals/coco_props/classagnostic_distilfeats
        datasets/MAVL_proposals/lvis_props/classagnostic_distilfeats/coco_distil_feats
    For ImageNet-LVIS
        python tools/get_rkd_clip_feat.py -ckpt <mavl pretrained weights path> -dataset imagenet_lvis
        -dataset_dir datasets/imagenet -output
        datasets/MAVL_proposals/lvis_props/classagnostic_distilfeats/imagenet_distil_feats
"""

import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import json

from external.mavl.models.model import Model
from external.mavl.utils.nms import nms
from utils.save_proposal_boxes import SaveProposalBoxes as SaveRKDFeats

import clip

TEXT_QUERIES = ["all objects", "all entities", "all visible objects and entities", "all obscure objects and entities"]


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ckpt", "--mavl_checkpoints_path", required=True,
                    help="The path to MAVL pretrained weights.")
    ap.add_argument("-dataset", "--dataset_name", required=False, default='coco',
                    help="The dataset name to generate the ILS labels for. Supported datasets are "
                         "['coco', 'imagenet_lvis']")
    ap.add_argument("-dataset_dir", "--dataset_base_dir_path", required=False, default='datasets/coco',
                    help="The dataset base directory path.")
    ap.add_argument("-output", "--output_dir_path", required=False,
                    default='datasets/MAVL_proposals/coco_props/classagnostic_distilfeats',
                    help="Path to save the ILS labels.")
    ap.add_argument("top_N", "top_N_rkd_proposals", type=int, required=False, default=5,
                    help="Number of proposals per image to be generated for RKD.")

    args = vars(ap.parse_args())

    return args


def crop_region(image, box):
    left, top, right, bottom = box
    im_crop = image.crop((left, top, right, bottom))
    return im_crop


def class_agnostic_nms(boxes, scores, iou=0.7):
    if len(boxes) > 1:
        boxes, scores = nms(np.array(boxes), np.array(scores), iou)
        return list(boxes), list(scores)
    else:
        return boxes, scores


def parse_coco_annotations(filename):
    dataset = json.load(open(filename, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    annos = {}
    all_annos = dataset["images"]
    for ann in all_annos:
        imag_name = ann["file_name"].split('.')[0]
        categories = ann["pos_category_ids"]
        annos[imag_name] = categories

    return annos


def get_mavl_proposals(image_path, text_query):
    boxes, scores = model.infer_image(image_path, caption=text_query)
    # Select top-50 individual query proposals for efficiency reasons
    scores = np.array(scores)
    boxes = np.array(boxes)
    sorted_ind = np.argsort(-scores)
    sorted_ind = sorted_ind[:50]
    boxes = boxes[sorted_ind, :]
    scores = scores[sorted_ind]
    boxes = boxes.tolist()
    scores = scores.tolist()

    return boxes, scores


def get_top_5_class_agnostic_mavl_proposals(image_path):
    # Generate MAVL class-agnostic proposals for each query
    all_boxes = []
    all_scores = []
    for text_query in TEXT_QUERIES:
        boxes, scores = get_mavl_proposals(image_path, text_query)
        all_boxes += boxes
        all_scores += scores
    # Combine proposals and select top-5
    boxes, scores = class_agnostic_nms(all_boxes, all_scores, iou=0.5)
    # Select top-5 individual query proposals for efficiency reasons
    scores = np.array(scores)
    boxes = np.array(boxes)
    sorted_ind = np.argsort(-scores)
    sorted_ind = sorted_ind[:top_N]
    boxes = boxes[sorted_ind, :]
    scores = scores[sorted_ind]
    boxes = boxes.tolist()
    scores = scores.tolist()

    return boxes, scores


def get_clip_features(image_path):
    # Generate MAVL proposals
    boxes, scores = get_top_5_class_agnostic_mavl_proposals(image_path)
    # General CLIP features
    image = Image.open(image_path)
    curr_rkd_region_feats = []
    try:
        for j in range(len(boxes)):
            im_crop = crop_region(image, boxes[j])
            cropped_region = clip_preprocessor(im_crop).unsqueeze(0).to("cpu")
            with torch.no_grad():
                image_features = clip_model.encode_image(cropped_region)
            clip_embeds = image_features.cpu()
            curr_rkd_region_feats.append((boxes[j], clip_embeds))
    except Exception as e:
        pass

    return curr_rkd_region_feats


def get_coco_rkd_clip_features(dataset_dir, save_dir):
    train_images_path = f"{dataset_dir}/train2017"
    # The coco dataset must be setup correctly before running this script, see datasets/README.md for details
    assert os.path.exists(train_images_path)
    # Iterate over all the images, generate class-agnostic proposals and extract CLIP features
    dumper = SaveRKDFeats()
    rkd_region_feats = {}
    for i, image_name in enumerate(tqdm(os.listdir(train_images_path))):
        if i > 0 and i % 100 == 0:  # Save every 100 iterations
            dumper.update(rkd_region_feats)
            dumper.save(save_dir)
            rkd_region_feats = {}
        image_path = f"{train_images_path}/{image_name}"
        image_name_key = image_name.split('.')[0]
        # Generate CLIP features
        rkd_region_feats[image_name_key] = get_clip_features(image_path)
    dumper.update(rkd_region_feats)
    dumper.save(save_dir)


def get_lvis_rkd_clip_features(dataset_dir, save_dir):
    images_path = f"{dataset_dir}/ImageNet-LVIS"
    ils_annotation_path = f"{dataset_dir}/annotations/imagenet_lvis_image_info.json"
    # The coco dataset must be setup correctly before running this script, see datasets/README.md for details
    assert os.path.exists(images_path)
    assert os.path.exists(ils_annotation_path)
    # Iterate over all the images, generate class-agnostic proposals and extract CLIP features
    dumper = SaveRKDFeats()
    rkd_region_feats = {}
    annotations = parse_coco_annotations(ils_annotation_path)
    images = annotations.keys()
    for i, image_name_key in enumerate(tqdm(images)):
        if i > 0 and i % 100 == 0:  # Save every 100 iterations
            dumper.update(rkd_region_feats)
            dumper.save_imagenet(save_dir)
            rkd_region_feats = {}
        image_path = f"{images_path}/{image_name_key}.JPEG"
        image_name = os.path.basename(image_name_key)
        # General CLIP features
        rkd_region_feats[image_name] = get_clip_features(image_path)
    dumper.update(rkd_region_feats)
    dumper.save_imagenet(save_dir)


if __name__ == "__main__":
    # Parse the arguments
    args = parse_arguments()
    mavl_checkpoints_path = args["mavl_checkpoints_path"]
    dataset_name = args["dataset_name"]
    dataset_base_dir = args["dataset_base_dir_path"]
    output_dir = args["output_dir_path"]
    top_N = args["top_N_rkd_proposals"]
    os.makedirs(output_dir, exist_ok=True)

    # Load MAVL model
    model = Model("mdef_detr", mavl_checkpoints_path).get_model()
    # Load CLIP model
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device="cpu")
    # Generate RKD features
    if dataset_name == "coco":
        get_coco_rkd_clip_features(dataset_base_dir, output_dir)
    elif dataset_name == "imagenet_lvis":
        get_lvis_rkd_clip_features(dataset_base_dir, output_dir)
    else:
        print(f"Only 'coco' and 'imagenet_lvis' datasets are supported.")
        raise NotImplementedError
