# U-Net

PyTorch implementation of 
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf).

# Usage
## Training
1 Create a new folder and put training and validation data.  
2 Write training config in "args.json".  
3 `python train.py`  
4 Start training.

## Prediction
1 Create a new folder and put test data.  
2 Write predict config in "args.json".  
3 `python predict.py`  
4 Start prediction and show result images.