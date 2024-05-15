import h5py
import numpy as np
import pickle

class Tracks:
    def __init__(self,imsfile,point_set_id=0):
        self.imsfile = imsfile
        self.point_set_id = point_set_id
        self.load()

    def load(self):
        with h5py.File(self.imsfile, 'r') as f:
            point_handle = f[f"Scene8/Content/Points{self.point_set_id}"]

            spot_xyz = np.array(point_handle["Spot"])

            _spot_t = np.array([[s[0],s[1],s[2]] for s in point_handle["SpotTimeOffset"]])
            _spot_uid = np.array([s[0] for s in spot_xyz])
            _spot_xyz = np.array([[s[1],s[2],s[3]] for s in spot_xyz])
            _track_idx = np.array(point_handle["Track0"])
            _track_edges = np.array(point_handle["TrackEdge0"])

            # Build dictionaries using unique identifiers as keys: note that we need to explicitly access each element of
            # arrays in the h5 file because they are stored in an array format that cannot be spliced
            self.spot_dict = { s : [xyz[0] ,xyz[1], xyz[2]] for s,xyz in zip(_spot_uid,_spot_xyz) }
            self.t_dict = { t[0] : [t[1],t[2]] for t in _spot_t }
            self.trackobj_dict = { ed[0] : ed[1] for ed in _track_edges }
           
            self.spot_uid = _spot_uid

            # Get the extents and resolution of the image
            self.extents_um = np.zeros((2,3))
            for i in range(3):
                self.extents_um[0][i] = self.decode_imaris_extents(f["DataSetInfo/Image"].attrs[f"ExtMin{i}"])
                self.extents_um[1][i] = self.decode_imaris_extents(f["DataSetInfo/Image"].attrs[f"ExtMax{i}"])
            
            self.volume_shape_um = self.extents_um[1] - self.extents_um[0]
            self.volume_shape = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"].shape[::-1]
            self.resolution = np.median( self.volume_shape_um / self.volume_shape ) #assume isotropic resolution

    def build_tracks_t(self,t=0,nt=10,in_voxels=True):
        # Get all spot uids at time t
        tracks = {}
        uids = []
        for tt in range(t,t+nt):
            idx = self.t_dict[tt]
            uids_t = self.spot_uid[idx[0]:idx[1]]
            for u in uids_t:
                if u not in uids:
                    tracks[u] = []
                    tracks[u].append(self.spot_dict[u])
                    uu = u
                    for i in range(t+nt-tt):
                        if uu in self.trackobj_dict:
                            uu = self.trackobj_dict[uu]
                            xyz = self.spot_dict[uu]
                            tracks[u].append(xyz)
                            uids.append(u)
                        else:
                            break
       
        if in_voxels:
            for t in tracks:
                tracks[t] = np.array(tracks[t]) / self.resolution
                
        return tracks

    @staticmethod
    def decode_imaris_extents(ext):
        val = ""
        for ee in ext:
            val += ee.decode("utf-8")
        val = float(val)
        return val

if __name__=="__main__":
    T = Tracks("/research/sharedresources/cbi/data_exchange/tayl1grp/2024_lightsheet/2023-08-23/20230823_Full_Embryo.ims")
    tracks = T.build_tracks_t(t=60,nt=18)
    with open("tracks.pkl","wb") as f:
        pickle.dump(tracks,f)
