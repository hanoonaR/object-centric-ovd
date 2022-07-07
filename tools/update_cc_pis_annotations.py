"""
This script filter coco caption annotations based on MAVL target based prediction thresholds. This method will use
coco_caption annotations, and filters image level supervision annotations based on MAVL prediction confidence.
"""

import argparse
import pickle
import json


def parse_det_pkl(path):
    with open(path, "rb") as f:
        file_to_boxes_dict = pickle.load(f)
    return file_to_boxes_dict


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ils_path", "--ils_labels_path", required=False,
                    default="datasets/MAVL_proposals/coco_props/class_specific",
                    help="The path to the directory containing top-1 max score MAVL proposals. "
                         "One pkl file for each image.")
    ap.add_argument("-ann", "--cc_ann_path", required=False,
                    default="datasets/coco/annotations/captions_train2017_tags_allcaps.json",
                    help="The path to 'captions_train2017_tags_allcaps.json' annotation file.")
    ap.add_argument("-o_ann", "--output_ann_path", required=False,
                    default="datasets/coco/annotations/captions_train2017_tags_allcaps_pis.json",
                    help="The path to save the filtered/updated annotation file.")

    args = vars(ap.parse_args())

    return args


def main():
    # Parse arguments
    args = parse_arguments()
    ils_labels_path = args["ils_labels_path"]
    cc_ann_path = args["cc_ann_path"]
    output_ann_path = args["output_ann_path"]
    # Load 'captions_train2017_tags_allcaps.json' annotations
    dataset = json.load(open(cc_ann_path, 'r'))
    # Generate the new/filtered annotations
    images = []
    thresh = 0.95  # Threshold used in our experiments
    for i, x in enumerate(dataset['images']):
        image_name = x['file_name']
        image_key = image_name.split('.')[0]
        target_detections = parse_det_pkl(f"{ils_labels_path}/{image_key}.pkl")
        cats = x["pos_category_ids"]  # Current ids in cc annotation
        selected_preds = {}
        shorlist_ids = []
        for id in cats:
            # Filter out ids not present based on pred threshold
            if target_detections[id][1][0] > thresh:
                selected_preds[id] = target_detections[id]
                shorlist_ids.append(id)
        # Update caption annotation
        x['pos_category_ids'] = shorlist_ids
        if len(x['pos_category_ids']) > 0:
            images.append(x)
    # Create new annotation file
    try:
        out_data = {'images': images, 'categories': dataset['categories'], 'annotations': []}
        for k, v in out_data.items():
            print(k, len(v))
        print('Writing to', output_ann_path)
        json.dump(out_data, open(output_ann_path, 'w'))
    except Exception as e:
        pass


if __name__ == "__main__":
    main()
