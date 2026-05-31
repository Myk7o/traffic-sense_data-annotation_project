# traffic-sense_data-annotation_project

Cityscapes Annotation Quality Research
This repository contains dataset preprocessing and filtering scripts for an annotation quality research project focused on autonomous driving scene understanding.
Project Overview
The goal of this project is to study how annotation quality affects semantic segmentation model performance. We use the Cityscapes dataset and evaluate models trained on perfect, over-annotated, and under-annotated versions of the ground truth masks.
Classes

Car — passenger vehicles
Person — pedestrians
Cyclist — bicycle riders, motorcyclists, and riders combined

All other Cityscapes classes are merged into a single background class.
Dataset
This project uses the Cityscapes dataset. If you use this code in your research please cite:
@inproceedings{Cordts2016Cityscapes,
  title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
  author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}