
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/JHkuK1mjP28" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

<br>

## Abstract

![main figure](docs/OVD_block_diag.png)
<p align="justify">
Existing open-vocabulary object detectors typically enlarge their vocabulary sizes by leveraging 
different forms of weak supervision. This helps generalize to novel objects at inference. Two popular forms of 
 weak-supervision used in open-vocabulary detection (OVD) include pretrained CLIP model and image-level supervision.
 We note that both these modes of supervision are not optimally aligned for the detection task: CLIP is trained
 with image-text pairs and lacks precise localization of objects while the image-level supervision has been used with
 heuristics that do not accurately specify local object regions. In this work, we propose to address this problem by
 performing object-centric alignment  of the language embeddings from the CLIP model. Furthermore, we visually ground
 the objects with only image-level supervision using a pseudo-labeling process that provides high-quality object 
 proposals and helps expand the vocabulary during training. We establish a bridge between the above two
 object-alignment strategies via a novel weight transfer function that aggregates their complimentary strengths.
 In essence, the proposed model seeks to minimize the gap between object and image-centric representations in the
 OVD setting. On the COCO benchmark, our proposed approach achieves 40.3 AP50 on novel classes, an absolute 11.9
 gain over the previous best performance. For LVIS, we surpass the state-of-the-art ViLD model by 5.0 mask AP for rare
 categories and 3.4 overall. </p>
 

 
## TSNE Visualizations

t-SNE plots of CLIP and our detector region embeddings on COCO novel categories.

![tSNE_plots](docs/tSNE_plots.png)


## Region Embeddings Similarity Matrices

Plots of the Region Embeddings similarity matrices of COCO Novel categories by CLIP and our detector. 

![SPKD](docs/similarity_matrix.png)
 
## Model Zoo

### New LVIS Baseline
Our Mask R-CNN based LVIS Baseline ([mask_rcnn_R50FPN_CLIP_sigmoid](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)) 
achieves 12.2 rare class and 20.9 overall AP and trains in only 4.5 hours on 8 A100 GPUs. 
We believe this could be a good baseline to be considered for the future research work in LVIS OVD setting.

| Name                                                                                       | APr | APc | APf | AP | Epochs                                                                                                                                          |
|--------------------------------------------------------------------------------------------|------|----|---|------|------|
| [PromptDet Baseline](https://arxiv.org/abs/2203.16513)          | 7.4 | 17.2 | 26.1 | 19.0 | 12 |
| [ViLD-text](https://arxiv.org/abs/2104.13921)           | 10.1 | 23.9 | 32.5 | 24.9 | 384 |
| [Ours Baseline](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)         | <b style="color:black;"> 12.2 </b> |  19.4 | 26.4 | 20.9 | 12 |



### Open-vocabulary COCO
Effect of individual components in our method. Our weight transfer method provides complimentary gains from RKD and ILS, achieving superior results as compared to naively adding both components.

| Method                                                                                       | APnovel | APbase | AP   | Download |
|--------------------------------------------------------------------------------------------|---------|--------|------|----------|
| [Base-OVD-RCNN-C4](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/coco/Base-OVD-RCNN-C4.yaml)                                     | 1.7     | 53.2   | 39.6 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EVLABS0bOahDqpRFOuzYR0YBzvVU-GiC4JMTsdSxMoUG4w?e=FqvWCT)                  |
| [COCO_OVD_Base_RKD](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/coco/COCO_OVD_Base_RKD.yaml)                                   | 21.6    | 54.4   | 45.8 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EamR9AJ8tBdJqYMf2Cn9tm8B0MoL4hpK2cavnsr0NKDcUA?e=WxNGRB)        |
| [COCO_OVD_Base_PIS](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/coco/COCO_OVD_Base_PIS.yaml)                                   | 34.2    | 52.0   | 47.4 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EQSUB_pxTalIiArcEPprzaABvC5CFg2Ti8u-gA6gZlljIA?e=LEUr6i)    |
| [COCO_OVD_RKD_PIS](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/coco/COCO_OVD_RKD_PIS.yaml)                                     | 35.3    | 52.9   | 48.3 | [model]() |
| [COCO_OVD_RKD_PIS_WeightTransfer](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/coco/COCO_OVD_RKD_PIS_WeightTransfer.yaml)       | 40.3    | 54.1   | 50.5 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/Edj5sCOJjAFPqEH3gBrCj6UBRNl6qkanZoHiUDYkTsOHlg?e=SjR5q2)            |
| [COCO_OVD_RKD_PIS_WeightTransfer_8x](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/coco/COCO_OVD_RKD_PIS_WeightTransfer_8x.yaml) | <b style="color:black;"> 40.5 </b>   |  <b style="color:black;"> 56.7 </b>   | <b style="color:black;"> 52.5 </b>  | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EUtwrJyDAO9GsI13MpyqyJ4BssrY1JZbaUhPnmFt4FJktA?e=9RxhZF) |

### Open-vocabulary LVIS
Effect of proposed components in our method on LVIS.


| Method                                                                                       | APr | APc | APf | AP | Download                                                                                                                                          |
|--------------------------------------------------------------------------------------------|------|----|---|------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [mask_rcnn_R50FPN_CLIP_sigmoid](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)           | 12.2 | 19.4 | 26.4 | 20.9 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EYtGSw6Cmt5JrrjIWV9rfdwBm_ncdhHuIjxJgE9BHv8d2g?e=kVcxb3) |
| [LVIS_OVD_Base_RKD](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/LVIS_OVD_Base_RKD.yaml)                                   | 15.2 | 20.2 | 27.3 | 22.1 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EXKueSpvtGNLmjvb3iLeK8UBZ_Zawjna4Uy5EmmnafwOtw?e=45Hsu6) |
| [LVIS_OVD_Base_PIS](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/LVIS_OVD_Base_PIS.yaml)                                   | 17.0 | 21.2 | 26.1 | 22.4 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ERr8V8v5Mp9NioxQ2GG_QnIB8SUzNN5NqfGWIXPIifgBmw?e=nls03R) |
| [LVIS_OVD_RKD_PIS](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/LVIS_OVD_RKD_PIS.yaml)                                     | 17.3 | 20.9 | 25.5 | 22.1 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EeLjE7LRTmdHhreI-baMncYBTGUadRF9kxHVYjC700L7Xg?e=TrI3oi) |
| [LVIS_OVD_RKD_PIS_WeightTransfer](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/LVIS_OVD_RKD_PIS_WeightTransfer.yaml)       | 17.2 | 21.5 | 26.6 | 22.8 | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ETZ6xlqmIxlEiee7Nj1G2I8BE6iaY7ArFEAEVHohQJCamg?e=mfP1Mh) |
| [LVIS_OVD_RKD_PIS_WeightTransfer_8x](https://github.com/hanoonaR/object-centric-ovd/blob/main/configs/lvis/LVIS_OVD_RKD_PIS_WeightTransfer_8x.yaml) | <b style="color:black;"> 21.1 </b> | <b style="color:black;"> 25.0 </b> | <b style="color:black;"> 29.1 </b>  | <b style="color:black;"> 25.9 </b> | [model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EV8g8qped_FCugaB83jeW6EBHBAgWf9ajXv_TeLEGiPMtg?e=wsac5n) |

### Comparison with Existing OVOD Works

#### Open-vocabulary COCO
We compare our OVD results with previously established methods.  †ViLD and our methods are trained for longer 8x schedule. ‡We train detic for another 1x for a fair comparison with our method.  For ViLD, we use their unified model that trains ViLD-text and ViLD-Image together. For Detic, we report their best model.
<br>
<center>
<table border="0">
<tbody>
<tr>
<td><center> <b>Method</b>  </center></td>
<td><center> <b>APnovel</b>  </center></td>
<td><center> <b>APbase</b>  </center></td>
<td><center> <b>AP</b>  </center></td>
</tr>
<tr>
  <td><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.pdf">OVR-CNN</a></td>
  <td>22.8</td>
  <td>46.0</td>
  <td>39.9</td>
</tr>
<tr>
<td><a href="https://arxiv.org/pdf/2104.13921.pdf">ViLD†</a></td>
<td>27.6</td>
<td>59.5</td>
<td>51.3</td>
</tr>
<tr>
<td><a href="https://arxiv.org/pdf/2201.02605.pdf">Detic</a></td>
<td>27.8</td>
<td>47.1</td>
<td>45.0</td>
</tr>
<tr>
<td><a href="https://arxiv.org/pdf/2201.02605.pdf">Detic‡</a></td>
<td>28.4</td>
<td>53.8</td>
<td>47.2</td>
</tr>
<tr>
<td>Ours</td>
<td><b style="color:black;"> 40.3 </b></td>
<td><b style="color:black;"> 54.1 </b></td>
<td><b style="color:black;"> 50.5 </b></td>
</tr>
<tr>
<td>Ours†</td>
<td>40.5</td>
<td>56.7</td>
<td>52.5</td>
</tr>
</tbody>
</table>
</center>
 
#### Open-vocabulary LVIS

Comparison with prior work ViLD, using their unified model (ViLD-text + ViLD-Image).

<br>

<center>
<table border="0">
<tbody>
<tr>
<td><center> <b>Method</b>  </center></td>
<td><center> <b>APr</b>  </center>   </td>
<td><center> <b>APc</b>  </center>   </td>
<td><center> <b>APf</b>  </center>   </td>
<td><center> <b>AP</b>  </center>   </td>
<td><center> <b>Epochs</b>  </center>   </td>
</tr>
<tr>
  <td><a href="https://arxiv.org/pdf/2203.14940.pdf">ViLD </a> </td>
<td>16.1</td>
<td>20.0</td>
<td> 28.3</td>
<td>22.5</td>
<td>384</td>
</tr>
<tr>
<td>Ours</td>
<td>17.2</td>
<td>21.5</td>
<td>26.6</td>
<td>22.8</td>
<td>36</td>
</tr>
<tr>
<td>Ours</td>
<td><b style="color:black;"> 21.1 </b></td>
<td><b style="color:black;"> 25.0 </b></td>
<td> <b style="color:black;"> 29.1 </b></td>
<td><b style="color:black;"> 25.9 </b></td>
<td>96</td>
</tr>
</tbody>
</table>
</center>

   
We show compare our method with Detic, by building on their strong LVIS baseline using CenterNetV2 detector.

<br> 
<center>
<table  border="0">
<tbody>
<tr>
<td><center> <b>Method</b>  </center>   </td>
<td><center> <b>APr</b>  </center>   </td>
<td><center> <b>APc</b>  </center>   </td>
<td><center> <b>APf</b>  </center>   </td>
<td><center> <b>AP</b>  </center>   </td>
</tr>
<tr>
  <td><a href="https://arxiv.org/pdf/2201.02605.pdf">Box-Supervised</a></td>
<td>16.3</td>
<td>31.0</td>
<td>35.4</td>
<td>30.0</td>
</tr>
<tr>
<td><a href="[https://arxiv.org/pdf/2201.02605.pdf](https://arxiv.org/pdf/2201.02605.pdf)">Detic (Image + Captions)</a></td>
<td> 24.6 </td>
<td> 32.5 </td>
<td> 35.6 </td>
<td> 32.4 </td>
</tr>
<tr>
<td>Ours</td>
<td> <b style="color:black;"> 25.2 </b></td>
<td> <b style="color:black;"> 33.4 </b></td>
<td><b style="color:black;"> 35.8 </b></td>
<td><b style="color:black;"> 32.9 </b></td>
</tr>
</tbody>
</table>
</center>

<br/> 


## Qualitative Results (Open Vocabulary Setting)

For COCO, base and novel categories are shown in <font color="purple">purple</font> and <font color="green">green</font> colors respectively.

![results](docs/coco_lvis.jpg)

## Qualitative Results (Cross Datasets transfer)

![results](docs/cross_data.jpg)


## BibTeX
```
@article{Rasheed2022Bridging,
        title={Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection},
        author={Hanoona Rasheed, Muhammad Maaz, Muhammad Uzair Khattak, Salman Khan and Fahad Shahbaz},
        journal={ArXiv},
        year={2022}
    }
```
