# AtrialJSQnet

## Overview
The repository contains the core codes of "[AtrialJSQnet: A New Framework for Joint Segmentation and Quantification of Left Atrium and Scars Incorporating Spatial and Shape Information](https://arxiv.org/pdf/2008.04729.pdf)".
The code includes four kinds of scripts:
### C++ script fold
This fold includes some tools writed using C++ for the pre-processing of LGE MRI.
### LearnGC fold
This fold includes the python code to train the [LearnGC](https://www.sciencedirect.com/science/article/pii/S1361841519301355), which is published in MedIA2019.
In this manuscript, we employed LearnGC for comparison.
Note that the scripts to generate the multi-scale patches and the pre-processing code for LearnGC is not included.
### Matlab fold
This fold includes some pre-processing scripts employed in AtrialJSQnet, and some of these scripts aimed to use the generated tools.

## Dataset
The dataset employed in this work is from [MICCAI 2018: Atrial Segmentation Challenge](http://www.cardiacatlas.org/challenges/left-atrium-fibrosis-and-scar-segmentation-challenge/).

## Releated work
You may also interested in the papers:
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

or

@article{li2020,
  title={AtrialJSQnet: a new framework for joint segmentation and quantification of left atrium and scars incorporating spatial and shape information},
  author={Li, Lei and Zimmer, Veronika A and Schnabel, Julia A and Zhuang, Xiahai},
  journal={arXiv preprint arXiv:2008.04729},
  year={2020}
}


If you have any questions, please contact lilei.sky@sjtu.edu.cn.

