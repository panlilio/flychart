import h5py
import numpy as np
import pickle
import copy

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
            self.toffset_dict = { t[0] : [t[1],t[2]] for t in _spot_t }
            self.trackobj_dict = { ed[0] : ed[1] for ed in _track_edges }
            self.spot_uid = _spot_uid
            
            self.t_dict = {}
            for t,idx_range in self.toffset_dict.items():
                for i in range(idx_range[0],idx_range[1]):
                    uid = self.spot_uid[i]
                    self.t_dict[uid] = t 

            # Get the extents and resolution of the image
            self.extents_um = np.zeros((2,3))
            for i in range(3):
                self.extents_um[0][i] = self.decode_imaris_extents(f["DataSetInfo/Image"].attrs[f"ExtMin{i}"])
                self.extents_um[1][i] = self.decode_imaris_extents(f["DataSetInfo/Image"].attrs[f"ExtMax{i}"])
            
            self.volume_shape_um = self.extents_um[1] - self.extents_um[0]
            self.volume_shape = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"].shape[::-1]
            self.resolution = np.median( self.volume_shape_um / self.volume_shape ) #assume isotropic resolution
            self.extents = self.extents_um / self.resolution

    def build_tracks_t_all(self,t=0,nt=10,in_voxels=True):
        # Get all spots for time t through t+nt
        tracks = {}
        timestamps = {}
        uids = []
        for tt in range(t,t+nt):
            idx = self.toffset_dict[tt]
            uids_t = self.spot_uid[idx[0]:idx[1]]
            for u in uids_t:
                if u in uids:
                    continue
                else:
                    tr,ti,ui = self.build_track(u,in_voxels=in_voxels)
                    tracks[u] = tr
                    timestamps[u] = ti
                    uids.extend(ui)
        return tracks, timestamps
    
    def build_tracks_t(self,t=0,nt=10,in_voxels=True):
        tracks = {}
        timestamps = {}
        idx = self.toffset_dict[t]
        uids = self.spot_uid[idx[0]:idx[1]]
        for u in uids:
            tr,ti,_ = self.build_track(u,in_voxels=in_voxels)
            tracks[u] = tr
            timestamps[u] = ti
        return tracks, timestamps

    def build_track(self,uid,in_voxels=True):
        track = []
        timestamps = []
        uids = []
        uu = uid
        if uu not in self.trackobj_dict:
            # Single point track
            track = [self.spot_dict[uu]]
            timestamps = [self.t_dict[uu]]
            uids = [uu]
        else:
            while uu in self.trackobj_dict:
                uids.append(uu)
                track.append(self.spot_dict[uu])
                timestamps.append(self.t_dict[uu])
                uu = self.trackobj_dict[uu]
        if in_voxels:
            track = np.array(track) / self.resolution
        return track, timestamps, uids

    @staticmethod
    def decode_imaris_extents(ext):
        val = ""
        for ee in ext:
            val += ee.decode("utf-8")
        val = float(val)
        return val

if __name__=="__main__":
    T = Tracks("/research/sharedresources/cbi/data_exchange/tayl1grp/2024_lightsheet/2023-08-23/20230823_Full_Embryo.ims")
    tracks, timestamps = T.build_tracks_t(t=61,nt=36)
    ims_extents = T.extents
    with open("tracks.pkl","wb") as f:
        pickle.dump(tracks,f)
    with open("timestamps.pkl","wb") as f:
        pickle.dump(timestamps,f)
    with open("ims_extents.pkl","wb") as f:
        pickle.dump(ims_extents,f)
