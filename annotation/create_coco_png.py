import os
import numpy as np
from pycocotools import mask
# from pycocotools.cocostuffhelper import getCMap
from pycocotools.coco import COCO
from PIL import Image, ImagePalette
# import skimage.io
import matplotlib.pyplot as plt
import matplotlib # For Matlab's color maps
from tqdm import tqdm
import cv2


def getCMap(stuffStartId=92, stuffEndId=182, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addThings: (optional) whether to add a color for the 91 thing classes
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :param addOther: (optional) whether to add a color for the 'other' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black (or any other) color for each thing class
    if addThings:
        thingsPadding = np.zeros((stuffStartId - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    # Add yellow/orange color for 'other' class
    if addOther:
        cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap

def cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=True, includeCrowd=False):
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])
    labelMap = np.zeros(imageSize)

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    if includeCrowd:
        annIds = coco.getAnnIds(imgIds=imgId)
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    imgAnnots = coco.loadAnns(annIds)
    for a in range(0, len(imgAnnots)):
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        unique, counts = np.unique(labelMask, return_counts=True)
        #labelMask = labelMasks[:, :, a] == 1
        newLabel = imgAnnots[a]['category_id']

        # if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
        #     continue
            # raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))
        # if (labelMap[labelMask] != 0).any():
        #     raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

        labelMap[labelMask] = newLabel
    
    unique, counts = np.unique(labelMap, return_counts=True)
    d = dict(zip(unique, counts))
    return labelMap

def cocoSegmentationToPng(coco,imgId,pngPath, includeCrowd=False):
    labelMap = cocoSegmentationToSegmentationMap(coco, imgId, includeCrowd=includeCrowd)
    # if labelMap is None: return
    labelMap = labelMap.astype(np.int8)
    # Get color map and convert to PIL's format
    cmap = getCMap()
    cmap = (cmap * 255).astype(int)
    padding = np.zeros((256-cmap.shape[0], 3), np.int8)
    cmap = np.vstack((cmap, padding))
    cmap = cmap.reshape((-1))
    cmap = np.uint8(cmap).tolist()
    assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'

    # Write to png file
    unique, counts = np.unique(labelMap, return_counts=True)
    d = dict(zip(unique, counts)) 

    cv2.imwrite(pngPath, labelMap)
    png = cv2.imread(pngPath, 0)
    unique, counts = np.unique(png, return_counts=True)
    d = dict(zip(unique, counts)) 
    # print(d)
    # png = Image.fromarray(labelMap).convert('P')
    # png.putpalette(cmap)
    # png.save(pngPath, format='PNG')





def cocoSegmentationToPngDemo(data_dir='D:/WorkSpace/JupyterWorkSpace/DataSet/COCO',data_type='train2017',
                              pngFolderName='D:/WorkSpace/JupyterWorkSpace/DataSet/COCO/mask',isAnnotation=True):
    # annPath = os.path.join(data_dir,'annotations/stuff_anno/stuff_{}.json'.format(data_type))
    annPath = os.path.join(data_dir,'annotations/instances_{}.json'.format(data_type))

    coco = COCO(annPath)
    imgIds = coco.getImgIds()
    imgCount = len(imgIds)
    for i in range(imgCount):
        imgId = imgIds[i]
        imgName = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.jpg', '')
        print('Exporting image %d of %d: %s' % (i + 1, imgCount, imgName))
        segmentationPath = '%s/%s/%s.png' % (pngFolderName, data_type, imgName)
        dir_ = '%s/%s' % (pngFolderName, data_type)
        if not os.path.exists(dir_): os.makedirs(dir_)
        cocoSegmentationToPng(coco, imgId, segmentationPath)

def create():
    for year in ["2014", "2017"]:
        for image_set in [('train'), ('val')]:  
            print("%s%s"%(image_set, year))
            cocoSegmentationToPngDemo(data_type='%s%s'%(image_set, year))

if __name__ == '__main__':
    create()