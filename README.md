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

args.json
```json
{
    "train": {
        "model": "UNet",
        "train_img_dir": "data/train/img",
        "train_gt_dir": "data/train/gt",
        "val_img_dir": "data/val/img",
        "val_gt_dir": "data/val/gt",
        "epochs": 1,
        "batch_size": 24,
        "log_dir": "log"
    },
    "predict": {
        "model": "UNet",
        "img_dir": "data/val/img",
        "log_dir": "log",
        "weight_path": ""
    }
}
```
