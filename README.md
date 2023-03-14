# Both Style and Distortion Matter: 
## Dual-Path Unsupervised Domain Adaptation for Panoramic Semantic Segmentation 

<div align=center><img width="406" alt="image" src="https://user-images.githubusercontent.com/49426295/224911386-0acc6e68-fb90-4db3-bef2-85dd14c9c7bc.png"></div>

**In this paper, we studied a new problem by refining the domain gaps between the panoramic and pinhole images into two types: the inherent gap and the format gap. We accordingly proposed DPPASS, the first dual-projection UDA framework, taking ERP and tangent images as input to each path to reduce the domain gaps.**

# Updates
[3/14/23] Create this repository!

# Prepare
<p>Environments:</p>
<pre><code>conda create -f DPPASS.yml
</code></pre>

## Data Preparation

### Cityscapes dataset
<div align=center><img width="512" alt="image" src="https://user-images.githubusercontent.com/49426295/224914735-e0a665cb-d22e-4ecc-9921-5268d64b97c1.png"></div>
Below are examples of the high quality dense pixel annotations from Cityscapes dataset. Overlayed colors encode semantic classes. Note that single instances of traffic participants are annotated individually.

The Cityscapes dataset is availabel at [Cityscapes](https://www.cityscapes-dataset.com/)

###SynPASS dataset
![image](https://user-images.githubusercontent.com/49426295/224914197-efb88edd-10bf-4686-8568-be24784c39a9.png)
SynPASS dataset contains 9080 panoramic images (1024x2048) and 22 categories.

The scenes include cloudy, foggy, rainy, sunny, and day-/night-time conditions.

The SynPASS dataset is availabel at [Trans4PASS](https://github.com/jamycheung/Trans4PASS)
