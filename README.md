# CDL
Official PyTorch implementation of CDL: Consistent Discrepancy Learning for Intra-camera Supervised Perosn Re-Identification.

Change the dataset path in line 450-455 in utils.py to your path to dataset. 
Set the ``load_weight'' in exp_1.yaml to your path to the pretrained model.
The pretrained model is trained to classify the persons under each view and learn pedestrian prototypes.
We provide our pretrained model at: https://pan.baidu.com/s/1kKrySNtjcoEvcuJp0ElVgQ?pwd=tkh7 【key: tkh7】 

Example: python main.py --gpu 0,1 --config config/exp_1.yaml



This project is implemented based on:
1. https://github.com/KovenYu/MAR
2. https://github.com/KaiyangZhou/deep-person-reid
