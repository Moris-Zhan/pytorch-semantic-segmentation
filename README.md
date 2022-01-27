# ImageSegmentation
My Frame work for ImageSegmentation
## Overview
I organizize the object detection algorithms proposed in recent years, and focused on **`Cityscapes`, `COCO` and `Pascal VOC` Dataset.
This frame work also include **`EarlyStopping mechanism`**.


## Datasets:

I used 3 different datases: **`Cityscapes`, `COCO`, `Pascal VOC`** . Statistics of datasets I used for experiments is shown below

- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:

-- VOC2007
![](https://i.imgur.com/wncA2wC.png)

-- VOC2012
![](https://i.imgur.com/v3AQelB.png)

  
  
| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| VOC2007                |    20   |      209/633       |          213/582        |
| VOC2012                |    20   |      1464/3507     |         1449/3422       |

  ```
  VOCDevkit
  ├── VOC2007
  │   ├── JPEGImages  
  │   ├── SegmentationClass
  │   ├── ...
  │   └── ...
  └── VOC2012
      ├── JPEGImages  
      ├── SegmentationClass
      ├── ...
      └── ...
  ```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:

| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| COCO2014               |    21   |         64k/-         |            31k/-           |
| COCO2017               |    21   |         92k/-        |             3k/-           |
```
  COCO
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   └── instances_val2017.json
  │── images
  │   ├── train2014
  │   ├── train2017
  │   ├── val2014
  │   └── val2017
  └── mask
      ├── train2014
      ├── train2017
      ├── val2014
      └── val2017
```

- **Cityscapes**:
The Cityscapes Dataset focuses on semantic understanding of urban street scenes. In the following, we give an overview on the design choices that were made to target the dataset’s focus.

![](https://i.imgur.com/Dgi4K9S.png)



  Download the container images and annotations from [cityscapes](https://www.cityscapes-dataset.com/downloads/). Make sure to put the files as the following structure:
 ![](https://i.imgur.com/rRJSIYQ.png)
![](https://i.imgur.com/L3bVJFM.png)  

```
  Cityscapes
  ├── leftImg8bit
  │   ├── train
  │   ├── val
  │   ├── test  
  │     
  │── gtFine_trainvaltest
      ├── gtFine
          ├── train
          ├── val
          ├── test 
```



## Methods
- **FPN**
- **FCN**
- **DeConvNet**
- **UNet**
- **SegNet**
- **PSPNet**
- **DeepLabV3**
- **DeepLabv3_plus**

## Prerequisites
* **Windows 10**
* **CUDA 10.1 (lower versions may work but were not tested)**
* **NVIDIA GPU 1660 + CuDNN v7.3**
* **python 3.6.9**
* **pytorch 1.10**
* **opencv (cv2)**
* **numpy**
* **torchvision 0.4**

## Usage
### 0. Prepare the dataset
* **Download custom dataset in the  `data_paths`.** 
* **And create custom dataset `custom_dataset.py` in the `dataset`.**

### 1. Train + Evaluate
```python
python train.py --model DeepLabv3_plus --dataset Pascal --batch_size 4 --n_gpu 1
```

### 2. Predict
```python
python predict.py --model PSPNet --experiment_dir "run\Cityscapes\PSPNet\experiment_10"
```

## Reference
- U-Net :  https://github.com/bubbliiiing/unet-pytorch
- SegNet : https://github.com/yassouali/pytorch-segmentation/blob/4c51ae43c791a37b0fb80c536b6614ff971c74e8/models/segnet.py
- DeconvNet: https://github.com/Jasonlee1995/DeconvNet/blob/main/Implementation/model.py
- FCN: https://github.com/yassouali/pytorch-segmentation/blob/4c51ae43c791a37b0fb80c536b6614ff971c74e8/models/fcn.py
- PSPNet: https://github.com/bubbliiiing/pspnet-pytorch
- [x] FPN : https://github.com/Andy-zhujunwen/FPN-Semantic-segmentation/blob/master/FPN-Seg/model/FPN.py
- [x] DeepLabV3: https://github.com/giovanniguidi/deeplabV3-PyTorch/blob/master/models/deeplab.py
- DeepLabv3_plus : https://github.com/bubbliiiing/deeplabv3-plus-pytorch
- Dataset Preprare: https://github.com/jfzhang95/pytorch-deeplab-xception/tree/master/dataloaders/datasets
https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
- cityscapes: https://www.cityscapes-dataset.com/dataset-overview/#features
- VOC2007: https://pjreddie.com/media/files/VOC2007_doc.pdf
- VOC2012: https://pjreddie.com/media/files/VOC2012_doc.pdf
