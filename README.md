# Object Centric Open Vocabulary Detection
Official repository of paper titled "[Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection]()".

[Hanoona Rasheed](https://scholar.google.com/citations?user=yhDdEuEAAAAJ&hl=en&authuser=1&oi=sra), [Muhammad Maaz](https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra), [Muhammad Uzair Khattak](https://scholar.google.com/citations?user=M6fFL4gAAAAJ&hl=en&authuser=1), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://www.youtube.com/embed/JHkuK1mjP28)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1t0tthvh_-dd1BvcmokEb-3FUIaEE31DD/view?usp=sharing)

## :rocket: News
* **(July 7, 2022)**
  * Training and evaluation code with pretrained models are released.

<hr />

![main figure](docs/OVD_block_diag.png)
> **<p align="justify"> Abstract:** *Existing open-vocabulary object detectors typically enlarge their vocabulary sizes by leveraging 
> different forms of weak supervision. This helps generalize to novel objects at inference. Two popular forms of 
> weak-supervision used in open-vocabulary detection (OVD) include pretrained CLIP model and image-level supervision.
> We note that both these modes of supervision are not optimally aligned for the detection task: CLIP is trained
> with image-text pairs and lacks precise localization of objects while the image-level supervision has been used with
> heuristics that do not accurately specify local object regions. In this work, we propose to address this problem by
> performing object-centric alignment  of the language embeddings from the CLIP model. Furthermore, we visually ground
> the objects with only image-level supervision using a pseudo-labeling process that provides high-quality object 
> proposals and helps expand the vocabulary during training. We establish a bridge between the above two
> object-alignment strategies via a novel weight transfer function that aggregates their complimentary strengths.
> In essence, the proposed model seeks to minimize the gap between object and image-centric representations in the
> OVD setting. On the COCO benchmark, our proposed approach achieves 40.3 AP50 on novel classes, an absolute 11.9
> gain over the previous best performance. For LVIS, we surpass the state-of-the-art ViLD model by 5.0 mask AP for rare
> categories and 3.4 overall.* </p>

## Main Contributions

1) **Region-based Knowledge Distillation (RKD)** adapts image-centric language representations to be object-centric.
2) **Pesudo Image-level Supervision (PIS)** uses weak image-level supervision from pretrained multi-modal ViTs to improve generalization of the detector to novel classes.
3) **Weight Transfer function** efficiently combines above two proposed components.

<hr />

## Installation
The code is tested with PyTorch 1.10.0 and CUDA 11.3. After cloning the repository, follow the below steps in [INSTALL.md](docs/INSTALL.md).
All of our models are trained using 8 A100 GPUs. 
<hr />

## Results
We present performance of Object-centric Open Vocabulary object detector that demonstrates state-of-the-art results on Open Vocabulary COCO and LVIS benchmark datasets.
For COCO, base and novel categories are shown in purple and green colors respectively.
![tSNE_plots](docs/coco_lvis.jpg)


### Open-vocabulary COCO
Effect of individual components in our method. Our weight transfer method provides complimentary gains from RKD and ILS, achieving superior results as compared to naively adding both components.

| Name                                                                                        |  APnovel  | APbase |   AP   | Train-time | Download                                                                                                                            |
|:--------------------------------------------------------------------------------------------|:---------:|:------:|:------:|:----------:|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [Base-OVD-RCNN-C4](configs/coco/Base-OVD-RCNN-C4.yaml)                                      |    1.7    |  53.2  |  39.6  |     8h     |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EVLABS0bOahDqpRFOuzYR0YBzvVU-GiC4JMTsdSxMoUG4w?e=FqvWCT) |
| [COCO_OVD_Base_RKD](configs/coco/COCO_OVD_Base_RKD.yaml)                                    |   21.6    |  54.4  |  45.8  |     8h     |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EamR9AJ8tBdJqYMf2Cn9tm8B0MoL4hpK2cavnsr0NKDcUA?e=WxNGRB) |
| [COCO_OVD_Base_PIS](configs/coco/COCO_OVD_Base_PIS.yaml)                                    |   34.2    |  52.0  |  47.4  |    8.5h    |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EQSUB_pxTalIiArcEPprzaABvC5CFg2Ti8u-gA6gZlljIA?e=LEUr6i) |
| [COCO_OVD_RKD_PIS](configs/coco/COCO_OVD_RKD_PIS.yaml)                                      |   35.3    |  52.9  |  48.3  |    8.5h    |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/Ed91jL8YuwBKhgg4zrOwpJ8BlHEpl777Nl9LonxmaZHp6A?e=tCp1w9) |
| [COCO_OVD_RKD_PIS_WeightTransfer](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer.yaml)        |   40.3    |  54.1  |  50.5  |    8.5h    |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/Edj5sCOJjAFPqEH3gBrCj6UBRNl6qkanZoHiUDYkTsOHlg?e=SjR5q2) |
| [COCO_OVD_RKD_PIS_WeightTransfer_8x](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer_8x.yaml)  |   40.5    |  56.7  |  52.5  |  2.5 days  |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EUtwrJyDAO9GsI13MpyqyJ4BssrY1JZbaUhPnmFt4FJktA?e=9RxhZF) |

### New LVIS Baseline
Our Mask R-CNN based LVIS Baseline ([mask_rcnn_R50FPN_CLIP_sigmoid](configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)) 
achieves 12.2 rare class and 20.9 overall AP and trains in only 4.5 hours on 8 A100 GPUs. 
We believe this could be a good baseline to be considered for the future research work in LVIS OVD setting.

| Name                                                                 | APr  | APc  | APf  |  AP  | Epochs |
|----------------------------------------------------------------------|:----:|:----:|:----:|:----:|:------:|
| [PromptDet Baseline](https://arxiv.org/abs/2203.16513)               | 7.4  | 17.2 | 26.1 | 19.0 |   12   |
| [ViLD-text](https://arxiv.org/abs/2104.13921)                        | 10.1 | 23.9 | 32.5 | 24.9 |  384   |
| [Ours Baseline](configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)     | 12.2 | 19.4 | 26.4 | 20.9 |   12   |

<br/> 

### Open-vocabulary LVIS

| Name                                                                                       |   APr   |  APc   |  APf  |  AP   | Train-time  | Download                                                                                                                                          |
|--------------------------------------------------------------------------------------------|:-------:|:------:|:-----:|:-----:|:-----------:|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [mask_rcnn_R50FPN_CLIP_sigmoid](configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)           |  12.2   |  19.4  | 26.4  | 20.9  |    4.5h     | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EYtGSw6Cmt5JrrjIWV9rfdwBm_ncdhHuIjxJgE9BHv8d2g?e=kVcxb3) |
| [LVIS_OVD_Base_RKD](configs/lvis/LVIS_OVD_Base_RKD.yaml)                                   |  15.2   |  20.2  | 27.3  | 22.1  |    4.5h     | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EXKueSpvtGNLmjvb3iLeK8UBZ_Zawjna4Uy5EmmnafwOtw?e=45Hsu6) |
| [LVIS_OVD_Base_PIS](configs/lvis/LVIS_OVD_Base_PIS.yaml)                                   |  17.0   |  21.2  | 26.1  | 22.4  |     5h      | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ERr8V8v5Mp9NioxQ2GG_QnIB8SUzNN5NqfGWIXPIifgBmw?e=nls03R) |
| [LVIS_OVD_RKD_PIS](configs/lvis/LVIS_OVD_RKD_PIS.yaml)                                     |  17.3   |  20.9  | 25.5  | 22.1  |     5h      | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EeLjE7LRTmdHhreI-baMncYBTGUadRF9kxHVYjC700L7Xg?e=TrI3oi) |
| [LVIS_OVD_RKD_PIS_WeightTransfer](configs/lvis/LVIS_OVD_RKD_PIS_WeightTransfer.yaml)       |  17.2   |  21.5  | 26.6  | 22.8  |     5h      | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ETZ6xlqmIxlEiee7Nj1G2I8BE6iaY7ArFEAEVHohQJCamg?e=mfP1Mh) |
| [LVIS_OVD_RKD_PIS_WeightTransfer_8x](configs/lvis/LVIS_OVD_RKD_PIS_WeightTransfer_8x.yaml) |  21.1   |  25.0  | 29.1  | 25.9  |  1.5 days   | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EV8g8qped_FCugaB83jeW6EBHBAgWf9ajXv_TeLEGiPMtg?e=wsac5n) |


### t-SNE plots

![tSNE_plots](docs/tSNE_plots.png)

<hr />

## Training and Evaluation

To train or evaluate, first prepare the required [datasets](docs/DATASETS.md).

To train a model, run the below command with the corresponding config file.

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml
```

Note: Some trainings are initialized from Supervised-base or RKD models. Download the corresponding pretrained models
and place them under `$object-centric-ovd/saved_models/`.

To evaluate a pretrained model, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
```
<hr />

## Citation
If you use our work, please consider citing:
```bibtex
 @article{Rasheed2022Bridging,
    title={Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection},
    author={Hanoona Rasheed, Muhammad Maaz, Muhammad Uzair Khattak, Salman Khan and Fahad Shahbaz Khan},
    journal={ArXiv},
    year={2022}
    }
```

## Contact
If you have any questions, please create an issue on this repository or contact at hanoona.bangalath@mbzuai.ac.ae or muhammad.maaz@mbzuai.ac.ae.


## References
Our RKD and PIS methods utilize the MViT model Multiscale Attention ViT with Late fusion (MAVL) proposed in the work [Class-agnostic Object Detection with Multi-modal Transformer (ECCV 2022)](https://github.com/mmaaz60/mvits_for_class_agnostic_od).
Our code is based on [Detic](https://github.com/facebookresearch/Detic) repository. We thank them for releasing their code.