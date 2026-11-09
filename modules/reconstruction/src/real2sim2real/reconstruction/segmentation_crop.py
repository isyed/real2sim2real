"""Fixed native-RGB segmentation crops with exact prompt/mask backprojection.

Cropping can expose more source pixels to a fixed-resolution segmenter. It
does not create new image detail or change the full-scene camera calibration.
Validate native restored masks before tracking; no mask pixels are synthesized
outside the declared crop.
"""
from copy import deepcopy


def validate_crop(xyxy,native_size):
    if (len(xyxy)!=4 or len(native_size)!=2 or any(type(x) is not int for x in [*xyxy,*native_size])
        or not 0<=xyxy[0]<xyxy[2]<=native_size[0] or not 0<=xyxy[1]<xyxy[3]<=native_size[1]):
        raise ValueError('Bounded integer native-pixel crop and width/height required')
    return tuple(xyxy)


def crop_annotation(prompt,xyxy,native_size):
    x0,y0,x1,y1=validate_crop(xyxy,native_size);result=deepcopy(prompt)
    if prompt.get('points'):
        for point in result['points']:
            if not x0<=point['x']<x1 or not y0<=point['y']<y1:raise ValueError('Prompt point escaped segmentation crop')
            point['x']-=x0;point['y']-=y0
    if prompt.get('box'):
        box=result['box']
        if not x0<=box['x0']<box['x1']<=x1 or not y0<=box['y0']<box['y1']<=y1:raise ValueError('Prompt box escaped segmentation crop')
        for key in ('x0','x1'):box[key]-=x0
        for key in ('y0','y1'):box[key]-=y0
    return result


def restore_mask(mask,xyxy,native_size):
    import numpy as np
    x0,y0,x1,y1=validate_crop(xyxy,native_size);mask=np.asarray(mask)
    if mask.ndim!=2 or mask.shape!=(y1-y0,x1-x0):raise ValueError('Mask and declared native crop differ')
    result=np.zeros((native_size[1],native_size[0]),np.uint8)
    result[y0:y1,x0:x1]=np.where(mask!=0,255,0).astype(np.uint8)
    return result
