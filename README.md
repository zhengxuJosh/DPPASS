# DPPASS 

**Both Style and Distortion Matter: Dual-Path Unsupervised Domain Adaptation for Panoramic Semantic Segmentation** CVPR 2023, [[PDF]]()

<div align=center><img width="406" alt="image" src="https://user-images.githubusercontent.com/49426295/224911386-0acc6e68-fb90-4db3-bef2-85dd14c9c7bc.png"></div>

**In this paper, we studied a new problem by refining the domain gaps between the panoramic and pinhole images into two types: the inherent gap and the format gap. We accordingly proposed DPPASS, the first dual-projection UDA framework, taking ERP and tangent images as input to each path to reduce the domain gaps.**

# Abstract
The ability of scene understanding has sparked active research for panoramic image semantic segmentation. However, the performance is hampered by distortion of the equirectangular projection (ERP) and a lack of pixel-wise annotations. For this reason, some works treat the ERP and pinhole images equally and transfer knowledge from the pinhole to ERP images via unsupervised domain adaptation (UDA). However, they fail to handle the domain gaps caused by: 1) the inherent differences between camera sensors and captured scenes; 2) the distinct image formats (e.g., ERP and pinhole images). In this paper, we propose a novel yet flexible dual-path UDA framework, DPPASS, taking ERP and tangent projection (TP) images as inputs. To reduce the domain gaps, we propose cross-projection and intra-projection training. The cross-projection training includes tangent-wise feature contrastive training and prediction consistency training. That is, the former formulates the features with the same projection locations as positive examples and vice versa, for the models' awareness of distortion, while the latter ensures the consistency of cross-model predictions between the ERP and TP. Moreover, adversarial intra-projection training is proposed to reduce the inherent gap, between the features of the pinhole images and those of the ERP and TP images, respectively. Importantly, the TP path can be freely removed after training, leading to no additional inference cost. Extensive experiments on DensePASS and WildPASS datasets show that our DPPASS achieves +1.06\% mIoU increment than the state-of-the-art approaches. 


# Updates
[3/14/23] Create this repository!

# Prepare
<p>Environments:</p>
<pre><code>conda create -f DPPASS.yml
</code></pre>

## Data Preparation

### Cityscapes dataset
<img width="1169" alt="image" src="https://user-images.githubusercontent.com/49426295/224917153-551a21e9-518c-4011-b35f-75cebe44c382.png">
Below are examples of the high quality dense pixel annotations from Cityscapes dataset. Overlayed colors encode semantic classes. Note that single instances of traffic participants are annotated individually.

The Cityscapes dataset is availabel at [Cityscapes](https://www.cityscapes-dataset.com/)

### SynPASS dataset
![image](https://user-images.githubusercontent.com/49426295/224914197-efb88edd-10bf-4686-8568-be24784c39a9.png)
SynPASS dataset contains 9080 panoramic images (1024x2048) and 22 categories.

The scenes include cloudy, foggy, rainy, sunny, and day-/night-time conditions.

The SynPASS dataset is availabel at [Trans4PASS](https://github.com/jamycheung/Trans4PASS)

### DensePASS dataset
![image](https://user-images.githubusercontent.com/49426295/224915598-0779f1d8-9d54-4bc4-9cbf-64e8c69fe244.png)

The DensePASS dataset is availabel at [Trans4PASS](https://github.com/jamycheung/Trans4PASS)

<p>Data Path:</p>
<pre><code>datasets/
|--- cityscapes
|   |___ gtfine
|   |___ leftImg8bit
|--- SynPASS
|   |--- img
|   |   |___ cloud
|   |   |___ fog
|   |   |___ rain
|   |   |___ sun
|   |--- semantic
|   |   |___ cloud
|   |   |___ fog
|   |   |___ rain
|   |   |___ sun
|--- DensePASS
|   |___ gtfine
|   |___ leftImg8bit
</code></pre>

# Reference
We appreciate the previous open-source works.
* [Trans4PASS / Trans4PASS+](https://github.com/jamycheung/Trans4PASS)
* [Segformer](https://github.com/NVlabs/SegFormer)
