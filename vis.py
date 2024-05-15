import pickle
import matplotlib.pyplot as plt
import numpy as np
from charts import Volume, BoundaryStandardizer

class Figure:
    def __init__(self, chart_vals_file=None, tracks_file=None, fwd_transform_file=None, fwd_origin_file=None, volume_centroid_file=None):
        self.chart_vals_file = chart_vals_file
        self.tracks_file = tracks_file
        self.fwd_transform_file = fwd_transform_file
        self.fwd_origin_file = fwd_origin_file
        self.volume_centroid_file = volume_centroid_file

        self.chart_vals = None
        self.tracks = None
        self.tracks_transformed = None
        self.fwd_transform = None
        self.fwd_origin = None
        self.volume_centroid = None
        self.tform = None

        self.load_data()
        self.transform_tracks()

    def load_data(self):
        self.chart_vals = self._load(self.chart_vals_file)
        self.tracks = self._load(self.tracks_file)
        self.fwd_transform = self._load(self.fwd_transform_file)
        self.fwd_origin = self._load(self.fwd_origin_file,is_coord=True)
        self.volume_centroid = self._load(self.volume_centroid_file,is_coord=True)

    def transform_tracks(self):
        self.tracks_transformed = []
        Z = [z[0] for z in self.chart_vals["Z"]]
        if self.fwd_transform is not None and self.tracks is not None:
            self.tform = TrackTransformer(self.fwd_transform, self.fwd_origin, self.volume_centroid, Z=Z, downsample_exponent=1)
            for _,track in self.tracks.items():
                append_me = self.tform.transform(track)
                self.tracks_transformed.append(append_me)
    
    def plot_with_tracks(self,dim0="S",dim1="Z"):
        fig,ax = plt.subplots()
        Volume.plot_chart(ax,X=self.chart_vals[dim0],Y=self.chart_vals[dim1],F=self.chart_vals["F"],xlabel=dim0,ylabel=dim1)
        for track in self.tracks_transformed:
            coords = self.track_chart_coords(track,dim0,dim1)
            ax.plot(coords)
        plt.show()

    def track_chart_coords(self,track,dim0="S",dim1="Z"):
        coords = []
        for xyz in track:
            z_idx = self.tform.nearest_z_idx(z=xyz[2])
            xy = [xyz[0],xyz[1]]
            val0 = self.chart_vals[dim0][z_idx]
            val1 = self.chart_vals[dim1][z_idx]
            chart_idx = self.tform.nearest_xy_idx(xy=xy,X=self.chart_vals["x"][z_idx],Y=self.chart_vals["y"][z_idx])
            coords.append([val0[chart_idx],val1[chart_idx]])
        return coords
            
    @staticmethod
    def _load(file=None,is_coord=False):
        if file:
            with open(file, 'rb') as f:
                data = pickle.load(f)
        else:
            data = [0,0,0] if is_coord else None
        return data


class TrackTransformer:
    def __init__(self, fwd_transform, fwd_origin=[0,0,0], centroid=[0,0,0],standardizer_key="01",Z=[0],downsample_exponent=1):
        self.fwd_transform = fwd_transform
        self.fwd_origin = fwd_origin
        self.centroid = centroid
        self.standardizer = BoundaryStandardizer(standardizer_key,centroid)
        self.Z = Z
        self.downsample_exponent = downsample_exponent

    def transform(self, xyz_track):
        track_transformed = []
        for xyz in xyz_track:
            xyz_ = [float(x) for x in xyz]
            xyz_ = np.array(self.fwd_transform.TransformPoint(xyz_)) - np.array(self.fwd_origin)
            xyz_ = xyz / 2**self.downsample_exponent
            xyz_ = xyz_.tolist()
            track_transformed.append(xyz_)
        return track_transformed

    def nearest_xy_idx(self,xy=[0,0],X=[0],Y=[0]):
        x_diff = np.array(X) - xy[0]
        y_diff = np.array(Y) - xy[1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        return np.argmin(dist)

    def nearest_z_idx(self,z=0):
        z_ = z - self.centroid[2]
        z_diff = np.abs(np.array(self.Z) - z_)
        return np.argmin(z_diff)

if __name__=="__main__":
    chart_file = 'chart_vals.pkl'
    tracks_file = 'tracks.pkl'
    fwd_transform_file = 'fwd_transform.pkl'
    fwd_origin_file = 'fwd_origin.pkl'
    volume_centroid_file = 'centroid.pkl'

    fig = Figure(chart_file, tracks_file, fwd_transform_file, fwd_origin_file, volume_centroid_file)
    fig.plot_with_tracks()

