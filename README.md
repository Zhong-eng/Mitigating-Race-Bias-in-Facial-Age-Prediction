# Mitigating Race Bias in Facial Age Prediction
This is the repo for EECS 448 WN23 final project at the University of Michigan.

## Introduction
We first define the age prediction problem in 2 ways
  
  - Regression
  - Fuzzy Classification

Then, to quantify the bias, we proposed a new fairness metric VUAL(Variance of Unweighted Average Loss). 

Finally, we implemented 3 effective and generalizable methods to mitigate the race bias in facial age prediction. 
  - Ensemble Learning
  - Adaptive Sampling
  - Adversarial Learning

We compare performance these 3 methods in terms of fairness violation and overall performace. You can find more details in our [presentation video](https://drive.google.com/file/d/11wT7VsUgeRKh16JRDrEzdmxWz_G3vkE6/view?usp=sharing) and [report](https://drive.google.com/file/d/18FtlfTiIW_OKZKUKEZlg-XF5oHO4U3Sa/view?usp=sharing).

## Run our code
### Dependencies
  - torch
  - torchvision
  - numpy
  - opencv-python
  - scikit-learn
  - matplotlib
### Run our code
  - Download [UTKFace dataset](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE?resourcekey=0-01Pth1hq20K4kuGVkp3oBw) and put images in the directory ```UTKFace```
  - Run ```python [name].py``` to try our methods
### Code structure
Files ending with ```fuzzy``` is for fuzzy classification. Others are for ```regression```.
  - ensemble----benchmark performance and ensemble learning
  - balance----balance training set, simplest debiasing method
  - adaptive_sampling----adaptive sampling
  - adversarial----adversarial learning



