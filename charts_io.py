import h5py
import hdf5plugin
import numpy as np
from itk import imread, image_from_array

class Reader:
    def __init__(self,filename):
        self.filename = filename
        self.reader = self.get_reader()

    def get_reader(self):
        if self.filename.endswith('.ims'):
            return IMSReader(self.filename)
        else:
            return ITKReader(self.filename)

    def __call__(self,*args,**kwargs):
        return self.reader(*args,**kwargs)


class IMSReader:
    def __init__(self,filename):
        self.filename = filename
    
    def __call__(self,t=0,channel=0,res=0):
        datapath = f"DataSet/ResolutionLevel {res}/TimePoint {t}/Channel {channel}/Data"
        with h5py.File(self.filename,'r') as f:
            volume = np.array(f[datapath])
        volume = image_from_array(volume)
        return volume

class ITKReader:
    def __init__(self,filename):
        self.filename = filename

    def __call__(self):
        volume = imread(self.filename)
        return volume


