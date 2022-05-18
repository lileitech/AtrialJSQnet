# AtrialJSQnet

## Overview
The repository contains the core codes of "[AtrialJSQnet: A New Framework for Joint Segmentation and Quantification of Left Atrium and Scars Incorporating Spatial and Shape Information](https://www.sciencedirect.com/science/article/pii/S1361841521003480)".
The resposutory includes four folds:
### C++ script fold
This fold includes some tools written using C++ for the pre-processing of LGE MRI.
### LearnGC fold
This fold includes the python code to train and test the [LearnGC](https://www.sciencedirect.com/science/article/pii/S1361841519301355), which was published in MedIA2019.
In this manuscript, we employed LearnGC for comparison.
Note that the scripts to generate the multi-scale patches and the pre-processing code for LearnGC are not included here. For the complete version of this code, please kindly refer to https://github.com/Marie0909/LearnGC.
### Matlab fold
This fold includes some pre-processing scripts employed in AtrialJSQnet, and some of these scripts aimed to use the generated C++ tools mentioned in the C++ script fold.
### Python script AtrialJSQnet
This fold includes the python code for training and test the AtrialJSQnet.

## Dataset
The dataset employed in this work is from [MICCAI 2018: Atrial Segmentation Challenge](http://www.cardiacatlas.org/challenges/left-atrium-fibrosis-and-scar-segmentation-challenge/).

## Releated work
You may also be interested in following papers:
1. [Atrial scar quantification via multi-scale CNN in the graph-cuts framework](https://www.sciencedirect.com/science/article/pii/S1361841519301355)
2. [Medical Image Analysis on Left Atrial LGE MRI for Atrial Fibrillation Studies: A Review](https://arxiv.org/pdf/2106.09862.pdf)
3. [AtrialGeneral: Domain Generalization for Left Atrial Segmentation of Multi-center LGE MRIs](https://link.springer.com/chapter/10.1007/978-3-030-87231-1_54)

## Evaluation
For evaluation, you could run LAScarQS2022_evaluate.py. Before runing LAScarQS2022_evaluate.py, you need to install SimpleITK, medpy and hausdorff by running "pip install SimpleITK/medpy/hausdorff". Also, note that this evaluation tool can only work in windows system as we only compiled the c++ tools in windows now, which are saved in the fold namely "tools".


## Cite
If this code is useful for you, please kindly cite this work via:

@article{journal/MedIA/li2022,  
  title={Atrial{JSQ}net: a new framework for joint segmentation and quantification of left atrium and scars incorporating spatial and shape information},   
  author={Li, Lei and Zimmer, Veronika A and Schnabel, Julia A and Zhuang, Xiahai},   
  journal={Medical Image Analysis},    
  volume={76},    
  pages={102303},    
  year={2022},    
  publisher={Elsevier}    
}


If you have any questions, you are always welcome to contact with lilei.sky@outlook.com.

