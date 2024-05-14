import h5py
import numpy as np

class Tracks:
    def __init__(imsfile):
        self.imsfile = imsfile
        
        self._spot_xyz = None
        self._spot_t = None
        self._spot_uid = None
        self._track_ids = None
        self.load_tracks()

    def load_tracks(self):
        with h5py.File(self.imsfile, 'r') as f:
            self._spot_xyz = np.array(f["Scene8/Content/Points0/Spot"])
            self._spot_t = np.array(f["Scene8/Content/Points0/SpotTimeOffset"])
            self._spot_uid = np.array([s[0] for s in self._spot_xyz])
            self._track_ids = np.array(f["Scene8/Content/Points0/Track0"])

    def build_tracks(self,t=0,nt=10):
        pass
    
    def spot_indices(self,t):
        return np.where(self._spot_t[...,0] == t)

