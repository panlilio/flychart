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
        self.slice_results = self.slice.process()

    def test_boundary_plot(self):
        plt.imshow(self.slice.im_intensity, cmap='inferno')
        plt.plot(self.slice.boundary.x, self.slice.boundary.y, 'r')
        plt.show()


        
