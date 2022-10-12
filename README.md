# Object Centric Open Vocabulary Detection (NeurIPS 2022)
Official repository of paper titled "[Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection](https://arxiv.org/abs/2207.03482)".

[Hanoona Rasheed](https://scholar.google.com/citations?user=yhDdEuEAAAAJ&hl=en&authuser=1&oi=sra), [Muhammad Maaz](https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra), [Muhammad Uzair Khattak](https://scholar.google.com/citations?user=M6fFL4gAAAAJ&hl=en&authuser=1), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://hanoonar.github.io/object-centric-ovd)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2207.03482)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19LBqQg0cS36rTLL_TaXZ7Ka9KJGkxiSe?usp=sharing) 
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://youtu.be/QLlxulFV0KE)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1t0tthvh_-dd1BvcmokEb-3FUIaEE31DD/view?usp=sharing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on-mscoco)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=bridging-the-gap-between-object-and-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on-1)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-1?p=bridging-the-gap-between-object-and-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on?p=bridging-the-gap-between-object-and-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on-lvis-v1-0)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-lvis-v1-0?p=bridging-the-gap-between-object-and-image)

## :rocket: News
* **(Oct 12, 2022)**
  * Interactive colab demo released.
* **(Sep 15, 2022)**
  * Paper accepted at NeurIPS 2022.
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
2) **Pesudo Image-level Supervision (PIS)** uses weak image-level supervision from pretrained multi-modal ViTs(MAVL) to improve generalization of the detector to novel classes.
3) **Weight Transfer function** efficiently combines above two proposed components.

<hr />

## Installation
The code is tested with PyTorch 1.10.0 and CUDA 11.3. After cloning the repository, follow the below steps in [INSTALL.md](docs/INSTALL.md).
All of our models are trained using 8 A100 GPUs. 
<hr />

## Demo: Create your own custom detector
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Object_Centric_OVD_Demo.ipynb) Checkout our demo using our interactive colab notebook. Create your own custom detector with your own class names. 


## Results
We present performance of Object-centric Open Vocabulary object detector that demonstrates state-of-the-art results on Open Vocabulary COCO and LVIS benchmark datasets.
For COCO, base and novel categories are shown in purple and green colors respectively.
![tSNE_plots](docs/coco_lvis.jpg)


### Open-vocabulary COCO
Effect of individual components in our method. Our weight transfer method provides complimentary gains from RKD and ILS, achieving superior results as compared to naively adding both components.

| Name                                                                                        | APnovel | APbase |  AP  | Train-time | Download                                                                                                                            |
|:--------------------------------------------------------------------------------------------|:-------:|:------:|:----:|:----------:|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [Base-OVD-RCNN-C4](configs/coco/Base-OVD-RCNN-C4.yaml)                                      |   1.7   |  53.2  | 39.6 |     8h     |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_base.pth) |
| [COCO_OVD_Base_RKD](configs/coco/COCO_OVD_Base_RKD.yaml)                                    |  21.2   |  54.7  | 45.9 |     8h     |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd.pth) |
| [COCO_OVD_Base_PIS](configs/coco/COCO_OVD_Base_PIS.yaml)                                    |  30.4   |  52.6  | 46.8 |    8.5h    |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_pis.pth) |
| [COCO_OVD_RKD_PIS](configs/coco/COCO_OVD_RKD_PIS.yaml)                                      |  31.5   |  52.8  | 47.2 |    8.5h    |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd_pis.pth) |
| [COCO_OVD_RKD_PIS_WeightTransfer](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer.yaml)        |  36.6   |  54.0  | 49.4 |    8.5h    |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd_pis_weighttransfer.pth) |
| [COCO_OVD_RKD_PIS_WeightTransfer_8x](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer_8x.yaml)  |  36.9   |  56.6  | 51.5 |  2.5 days  |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd_pis_weighttransfer_8x.pth) |

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

| Name                                                                                       | APr  | APc  | APf  |  AP   | Train-time  | Download                                                                                                                  |
|--------------------------------------------------------------------------------------------|:----:|:----:|:----:|:-----:|:-----------:|---------------------------------------------------------------------------------------------------------------------------|
| [mask_rcnn_R50FPN_CLIP_sigmoid](configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)           | 12.2 | 19.4 | 26.4 | 20.9  |    4.5h     | [model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/lvis_ovd_base.pth) |
| [LVIS_OVD_Base_RKD](configs/lvis/LVIS_OVD_Base_RKD.yaml)                                   | 15.2 | 20.2 | 27.3 | 22.1  |    4.5h     | [model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/lvis_ovd_rkd.pth) |
| [LVIS_OVD_Base_PIS](configs/lvis/LVIS_OVD_Base_PIS.yaml)                                   | 17.0 | 21.2 | 26.1 | 22.4  |     5h      | [model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/lvis_ovd_pis.pth) |
| [LVIS_OVD_RKD_PIS](configs/lvis/LVIS_OVD_RKD_PIS.yaml)                                     | 17.3 | 20.9 | 25.5 | 22.1  |     5h      | [model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/lvis_ovd_rkd_pis.pth) |
| [LVIS_OVD_RKD_PIS_WeightTransfer](configs/lvis/LVIS_OVD_RKD_PIS_WeightTransfer.yaml)       | 17.1 | 21.4 | 26.7 | 22.8  |     5h      | [model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/lvis_ovd_rkd_pis_weighttransfer.pth) |
| [LVIS_OVD_RKD_PIS_WeightTransfer_8x](configs/lvis/LVIS_OVD_RKD_PIS_WeightTransfer_8x.yaml) | 21.1 | 25.0 | 29.1 | 25.9  |  1.5 days   | [model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/lvis_ovd_rkd_pis_weighttransfer_8x.pth) |


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
@inproceedings{Hanoona2022Bridging,
    title={Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection},
    author={Rasheed, Hanoona and Maaz, Muhammad and Khattak, Muhammad Uzair  and Khan, Salman and Khan, Fahad Shahbaz},
    booktitle={36th Conference on Neural Information Processing Systems (NIPS)},
    year={2022}
}
    
@inproceedings{Maaz2022Multimodal,
      title={Class-agnostic Object Detection with Multi-modal Transformer},
      author={Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Khan, Fahad Shahbaz and Anwer, Rao Muhammad and Yang, Ming-Hsuan},
      booktitle={17th European Conference on Computer Vision (ECCV)},
      year={2022},
      organization={Springer}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at hanoona.bangalath@mbzuai.ac.ae or muhammad.maaz@mbzuai.ac.ae.


## References
Our RKD and PIS methods utilize the MViT model Multiscale Attention ViT with Late fusion (MAVL) proposed in the work [Class-agnostic Object Detection with Multi-modal Transformer (ECCV 2022)](https://github.com/mmaaz60/mvits_for_class_agnostic_od).
Our code is based on [Detic](https://github.com/facebookresearch/Detic) repository. We thank them for releasing their code.
