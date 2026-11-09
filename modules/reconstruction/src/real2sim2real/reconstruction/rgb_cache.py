"""Write complete native RGB JPEG caches with the tracker's full contract."""
from pathlib import Path
import io
import numpy as np


def write_frames(frames,output,*,expected_count):
    import h5py
    from PIL import Image
    output=Path(output)
    if output.exists() or type(expected_count)!=int or expected_count<1:raise ValueError('Fresh cache and explicit positive native frame count required')
    output.parent.mkdir(parents=True,exist_ok=True);shape=None;count=0
    with h5py.File(output,'x') as h:
        h.attrs['complete']=False
        ds=h.create_dataset('frames',(expected_count,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i,frame in enumerate(frames):
            value=np.asarray(frame)
            if (i>=expected_count or value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3
                or min(value.shape[:2])<1 or (shape is not None and value.shape!=shape)):
                raise ValueError('Native RGB count, uint8 layout or frame dimensions changed')
            if shape is None:
                shape=value.shape
                ds.attrs.update(encoding='jpeg',format='JPEG',quality=100,subsampling='4:4:4',optimize=True,
                    colorspace='RGB',height=shape[0],width=shape[1],channels=3,source_dtype='uint8')
            buffer=io.BytesIO();Image.fromarray(value).save(buffer,format='JPEG',quality=100,subsampling=0,optimize=True)
            ds[i]=np.frombuffer(buffer.getvalue(),dtype=np.uint8);count=i+1
        if count!=expected_count:raise ValueError('Native RGB stream ended before declared frame count')
        h.attrs.update(complete=True,sequence='source',camera_id=0)
    return dict(frames=count,shape=list(shape),encoding='JPEG100/444/optimized',lossless_pixels=False)
