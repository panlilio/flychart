import unittest
from charts import Volume, Slice
import itk
import matplotlib.pyplot as plt

class TestWedge(unittest.TestCase):
    def setUp(self):
        self.vol = Volume(data_intensity=itk.imread('tests/dummy_intensity.tif'),
                            data_segmented=itk.imread('tests/dummy_segmentation.tif'))
        self.slice = Slice(self.vol.intensity_slice(30), 
                           self.vol.segmented_slice(30),
                           xy_centroid=self.vol.centroid[:2],
                           nS=100)

    def test_boundary_plot(self):
        S,R,wp = self.slice.boundary.process(test_pt=0)
        idx = 0
        plt.imshow(self.slice.im_intensity,cmap='inferno')
        plt.plot(self.slice.boundary.x, self.slice.boundary.y, 'r')
        plt.plot(self.slice.boundary.x[idx], self.slice.boundary.y[idx], 'ro')
        for i in range(len(wp[idx])):
            plt.plot(wp[idx][i][0], wp[idx][i][1], 'g+')
        plt.show()


        
