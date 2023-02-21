# CDL
Official PyTorch implementation of CDL: Consistent Discrepancy Learning for Intra-camera Supervised Perosn Re-Identification.

Change the dataset path in line 450-455 in utils.py and the ``load_weight'' in exp_1.yaml.
The pretrained model is trained ato classify the persons under each view, and we provide our pretrained model at: xxxxxx

Example: python main.py --gpu 0,1 --config config/exp_1.yaml



This project is implemented based on:
1. https://github.com/KovenYu/MAR
2. https://github.com/KaiyangZhou/deep-person-reid
