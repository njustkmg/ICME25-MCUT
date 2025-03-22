# MCUT（ICME2025)
Coordinated Uni-modal Assistance for Enhancing Multi-modal Learning
![MCUT Framework](https://github.com/njustkmg/ICME25-MCUT/blob/main/MCUT_Framework.png)
# Data Preparation

You can download the corresponding raw data from the link below and prepare the data according the instructions of the cited paper.

Original Dataset : [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset),[Sarcasm](https://github.com/feiLinX/Multi-modal-Sarcasm-Detection),[Twitter15](https://github.com/jefferyYu/TomBERT),[NVGresutre](https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d)
# Pre-processing
The data processing details are in [OGE-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/tree/main) 
# Training

## Uni-modal Learning

python3 -W ignore Unimodal_train.py --modal=audio --lr=1e-2 --gpu_id=0 --dataset=CREMA --epoch=200 --lr_decay_epoch=150 --our_model=audio 

## Baseline 

python3 -W ignore Baseline_train.py  --lr=1e-2 --gpu_id=0 --dataset=CREMA --epoch=150 --lr_decay_epoch=120 --our_model=base --alpha=1 --batch-size=64 

## Naive Weight 

python3 -W ignore Weight_train.py  --lr=1e-2 --gpu_id=0 --dataset=CREMA --epoch=150 --lr_decay_epoch=120 --our_model=weight --fusion_method=concat

## MCUT

python3 -W ignore MCUT_train.py  --lr=1e-2 --gpu_id=0 --dataset=CREMA --epoch=150 --lr_decay_epoch=120 --our_model=MCUT --T=0.5 --batch-size=64 --fusion_method=concat --modulation=1 --caculate_cosine=1
