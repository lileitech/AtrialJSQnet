# AtrialJSQnet

## Overview
The repository contains the core codes of "[AtrialJSQnet: A New Framework for Joint Segmentation and Quantification of Left Atrium and Scars Incorporating Spatial and Shape Information](https://www.sciencedirect.com/science/article/pii/S1361841521003480)".
The resposutory includes four folds:
### C++ script fold
This fold includes some tools written using C++ for the pre-processing of LGE MRI.
### LearnGC fold
This fold includes the python code to train and test the [LearnGC](https://www.sciencedirect.com/science/article/pii/S1361841519301355), which was published in MedIA2019.
In this manuscript, we employed LearnGC for comparison.
Note that the scripts to generate the multi-scale patches and the pre-processing code for LearnGC are not included here.
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


## Cite
If this code is useful for you, please kindly cite this work via:

@article{journal/MedIA/li2020,  
  title={Atrial scar quantification via multi-scale CNN in the graph-cuts framework},  
  author={Li, Lei and Wu, Fuping and Yang, Guang and Xu, Lingchao and Wong, Tom and Mohiaddin, Raad and Firmin, David and Keegan, Jennifer and Zhuang, Xiahai},  
  journal={Medical image analysis},  
  volume={60},  
  pages={101595},   
  year={2020},    
  publisher={Elsevier}  
}

and

@article{journal/MedIA/li2021,  
  title={AtrialJSQnet: a new framework for joint segmentation and quantification of left atrium and scars incorporating spatial and shape information},   
  author={Li, Lei and Zimmer, Veronika A and Schnabel, Julia A and Zhuang, Xiahai},   
  journal={Medical Image Analysis},   
  pages={102303},   
  year={2021},    
  publisher={Elsevier}  
}


If you have any questions, you are always welcome to contact with lilei.sky@sjtu.edu.cn.

