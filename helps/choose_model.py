class ModelType:
    DEEPLABV3_PLUS   = 0
    DEEPLABV3        = 1
    PSPNET           = 2
    UNET             = 3 
    SEGNET           = 4 
    FCN              = 5
    DeconvNet        = 6
    FPN              = 7
    

def check_model(o):
    str__ = str(o).split(".")[1].lower()
    if "deeplabv3_plus" in str__: 
        return ModelType.DEEPLABV3_PLUS
    elif "deeplabv3" in str__: 
        return ModelType.DEEPLABV3
    elif "pspnet" in str__: 
        return ModelType.PSPNET
    elif "unet" in str__: 
        return ModelType.UNET
    elif "segnet" in str__: 
        return ModelType.SEGNET  
    elif "fcn" in str__: 
        return ModelType.FCN 
    elif "deconvnet" in str__: 
        return ModelType.DeconvNet 
    elif "fpn" in str__: 
        return ModelType.FPN