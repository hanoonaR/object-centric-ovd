# Prepare datasets for Open Vocabulary Detection

We conduct our experiments on [COCO](https://cocodataset.org/) and [LVIS v1.0](https://www.lvisdataset.org/) datasets
under the OVD setting. We use a subset of [ImageNet-21K](https://www.image-net.org/download.php) having 997 overlapping
LVIS categories and [COCO captions](https://cocodataset.org/) dataset for ILS in LVIS and COCO experiments respectively.
We use the generalized ZSD setting where the classifier contains both base and novel categories. 
Before starting processing, please download the required datasets from the official websites and place or sim-link them
under `$object-centric-ovd/datasets/`.

```
object-centric-ovd/datasets/
    lvis/
    coco/
    imagenet/
    zeroshot_weights/
```
`zeroshot_weights/` is included in the repo, containing the prepared weights for the open-vocabulary zero-shot
classifier head. See the section zeroshot weights below for details on how to prepare them.

Download the COCO images, COCO and LVIS annotations and place them as follows. 

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
```

## COCO Open Vocabulary
 
The annotations for OVD training `instances_train2017_seen_2_oriorder.json`, `instances_train2017_seen_2_oriorder_cat_info.json`
and evaluation `instances_val2017_all_2_oriorder.json`, and annotations for image-level supervision `captions_train2017_tags_allcaps_pis.json` 
can be downloaded from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EhJUv1cVKJtCnZJzsvGgVwYBgxP6M9TWsD-PBb_KgOjhmQ?e=iYkfDZ).

The CLIP image features on class-agnostic MAVL proposals for region-based knowledge distillation (RKD) and
class-specific proposals for pseudo image-level supervision (PIS) can be downloaded from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EeJuo844j8FIsnuiX3wBxCgBcBR2MSjbhiLCuA4OC2cSWg?e=5BeESO).
Untar the file `coco_props.tar.gz` and place in the corresponding location as shown below:

```
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
        captions_train2017_tags_allcaps_pis.json
    zero-shot/
        instances_train2017_seen_2_oriorder.json
        instances_val2017_all_2_oriorder.json
        instances_train2017_seen_2_oriorder_cat_info.json
MAVL_proposals
    coco_props/
        classagnostic_distilfeats/
            000000581921.pkl
            000000581929.pkl
            ....
        class_specific/
            000000581921.pkl
            000000581929.pkl
            ....
```

Otherwise, follow the following instructions to generate them from the COCO standard annotations. We follow the 
code-base of [Detic](https://github.com/facebookresearch/Detic) for the dataset preperation.

1) COCO annotations

Following the work of [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb), we first
create the open-vocabulary COCO split. The converted files should be placed as shown below,
```
coco/
    zero-shot/
        instances_train2017_seen_2.json
        instances_val2017_all_2.json
```
These annotations are then pre-processed for easier evaluation, using the following commands:

```
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
```

2) Annotations for Image-level labels from COCO-Captions

The final annotation to prepare is `captions_train2017_tags_allcaps_pis.json`.
For the Image-level supervision, we use the COCO-captions annotations and filter them with the MAVL predictions.
First generate the Image-level labels from COCO-captions with the command, 
```
python tools/get_cc_tags.py --cc_ann datasets/coco/annotations/captions_train2017.json 
    --out_path datasets/coco/annotations/captions_train2017_tags_allcaps.json  
    --allcaps --convert_caption --cat_path datasets/coco/annotations/instances_val2017.json
```
This creates `datasets/coco/captions_train2017_tags_allcaps.json`.

To ignore the remaining classes from the COCO that are not included in seen(65)+ unseen(17),
`instances_train2017_seen_2_oriorder_cat_info.json` is used by the flag `IGNORE_ZERO_CATS`.  This is created by 
```
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_oriorder.json
```

3) Proposals for PIS

With the Image-level labels from COCO-captions `captions_train2017_tags_allcaps_pis.json`, 
generate pseudo-proposals for PIS using MAVL with the below command. Download the checkpoints from the external submodule.
```
python tools/get_ils_labels.py -ckpt <mavl pretrained weights path> -dataset coco -dataset_dir datasets/coco
        -output datasets/MAVL_proposals/coco_props/class_specific
```
The class-specific pseudo-proposals will be stored as individual pickle files for each image.

Filter the Image-level annotation `captions_train2017_tags_allcaps.json` from COCO-captions with the pseudo-proposals
from MAVL with the command,
```
python tools/update_cc_pis_annotations.py --ils_path datasets/MAVL_proposals/coco_props/class_specific 
    --cc_ann_path datasets/coco/annotations/captions_train2017_tags_allcaps.json 
    --output_ann_path datasets/coco/annotations/captions_train2017_tags_allcaps_pis.json
```
This creates `datasets/coco/captions_train2017_tags_allcaps_pis.json`.

4) Regions and CLIP embeddings for RKD

Generate the class-agnostic proposals for the images in the training set. The class-agnostic pseudo-proposals and 
corresponding CLIP images features will be stored as individual pickle files for each image.
Note: It does not depend on  steps 2 and 3.
```
python tools/get_rkd_clip_feat.py -ckpt <mavl pretrained weights path> -dataset coco -dataset_dir datasets/coco
        -output datasets/MAVL_proposals/coco_props/classagnostic_distilfeats
```
## LVIS Open Vocabulary
 
The annotations for OVD training `lvis_v1_train_norare` can be downloaded from
[here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EjaR4-EQFmVNhaYmsJhbwKMBciHdSFd8Z2J7byTplHAURA?e=tbzhUM) 
and annotations for image-level supervision `imagenet_lvis_v1_pis` can be downloaded from
[here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/Eg22Nzk-QelGgjhSKwtQ25QB_FgDR92VOQU3lr79uMXqqQ?e=FSsFnt)
Note: The provided ImageNet annoatations `imagenet_lvis_image_info_pis` contains the MAVL class-specific 
predictions for the corresponding LVIS categories, to speed-up training. 

The CLIP image features on class-agnostic MAVL proposals for region-based knowledge distillation (RKD) on ImageNet
can be downloaded from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/El9e1sKtdgBHlSxs0rEul5IBi2gcQBGthxXo0u4u-PlNcQ?e=VbqHDY).
The MAVL proposals on LVIS images are same as proposals generated for COCO, which can be downloaded from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EeJuo844j8FIsnuiX3wBxCgBcBR2MSjbhiLCuA4OC2cSWg?e=5BeESO).
Untar the file `imagenet_distil_feats.tar` and `coco_props.tar.gz`, and place in the corresponding location as shown below:

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
    lvis_v1_train_norare.json
    lvis_v1_train_norare_cat_info.json
coco/
    train2017/
    val2017/
imagenet/
    ImageNet-LVIS/
        n13000891/
        n15075141/
    annotations
        imagenet_lvis_image_info_pis.json
MAVL_proposals
    lvis_props/
        classagnostic_distilfeats/
            coco_distil_feats/
                    000000581921.pkl
                    000000581929.pkl
                    ....
            imagenet_distil_feats/
                    n13000891/
                        n13000891_995.pkl
                        n13000891_999.pkl
                        ....
                    n15075141/
                        n15075141_9997.pkl
                        n15075141_999.pkl
                        ....
        class_specific/
            imagenet_lvis_props/
                    n13000891/
                        n13000891_995.pkl
                        n13000891_999.pkl
                        ....
                    n15075141/
                        n15075141_9997.pkl
                        n15075141_999.pkl
                        ....
```

Prepare the `ImageNet-LVIS` directory by unzipping the Image-Net21k. Note we use the winter-21 version.   
```
python tools/unzip_imagenet_lvis.py --dst_path datasets/imagenet/ImageNet-LVIS
```

Otherwise, follow the following instructions to generate them from the standard annotations.
You can prepare the open-vocabulary LVIS training set using 

1) LVIS annotations
```
python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
```

This will generate `datasets/lvis/lvis_v1_train_norare.json`.

`lvis_v1_train_norare_cat_info.json` is used by the Federated loss.  This is created by 
```
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train_norare.json
```
2) Annotations for PIS

For the Image-Net annotations for image-level supervision, we first unzip the overlapping classes of LVIS 
and convert them into LVIS annotation format.

```
mkdir imagenet/annotations
python tools/create_imagenetlvis_json.py --imagenet_path datasets/imagenet/ImageNet-LVIS
 --out_path datasets/imagenet/annotations/imagenet_lvis_image_info.json
```
This creates `datasets/imagenet/annotations/imagenet_lvis_image_info.json`.

3) Proposals for PIS

With the Image-level labels from ImageNet `imagenet_lvis_image_info.json`, 
generate pseudo-proposals for PIS using MAVL with the below command. Download the checkpoints from the external submodule.
```
python tools/get_ils_labels.py -ckpt <mavl pretrained weights path> -dataset imagenet_lvis
        -dataset_dir datasets/imagenet -output datasets/MAVL_proposals/lvis_props/class_specific/imagenet_lvis_props
```
The class-specific pseudo-proposals will be stored as individual pickle files for each image.

To generate the annotation with MAVL class-specific predictions, run the below command. This command creates a single
json file to be used for ILS from the image-level pkl files. This is not mandatory,
however, doing so improves training efficiency (instead of loading pseudo-proposals from individual pickle files in 
the dataloader). Don't set `PIS_PROP_PATH` path in config if using single json file.
```
python tools/create_lvis_ils_json.py -dataset_dir datasets/imagenet
    -prop_path datasets/MAVL_proposals/lvis_props/class_specific/imagenet_lvis_props
    -target_path datasets/imagenet/annotations/imagenet_lvis_image_info_pis.json
```
4) Regions and CLIP embeddings for RKD

Generate the class-agnostic proposals for the images in the training set. The class-agnostic pseudo-proposals and 
corresponding CLIP images features will be stored as individual pickle files for each image.
Note: It does not depend on  steps 2 and 3.

For LVIS, Use the same features generated from COCO, as the images are same and predictions are class-agnostic
```
ln -s datasets/MAVL_proposals/coco_props/classagnostic_distilfeats
        datasets/MAVL_proposals/lvis_props/classagnostic_distilfeats/coco_distil_feats
```
For ImageNet-LVIS,
```
python tools/get_rkd_clip_feat.py -ckpt <mavl pretrained weights path> -dataset imagenet_lvis
        -dataset_dir datasets/imagenet 
        -output datasets/MAVL_proposals/lvis_props/classagnostic_distilfeats/imagenet_distil_feats
```

## Zeroshot Weights

We use the query `a photo of a {category}` to compute the test embeddings for the classifier. 
```
zeroshot_weights/
    coco_clip_a+photo+cname.npy
    lvis_v1_clip_a+photo+cname.npy
```
The weights for COCO can be generated with,
```
python tools/dump_clip_features.py --ann datasets/coco/annotations/instances_val2017.json --out_path zeroshot_weights/coco_clip_a+photo+cname.npy --prompt photo
```
For LVIS, the command is:
```
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path zeroshot_weights/lvis_v1_clip_a+photo+cname.npy --prompt photo
```
