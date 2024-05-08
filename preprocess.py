import itk
import numpy as np
import scipy as sp

class Preprocess:
    def __init__(self, intensity_image=None, mask_image=None, aligned_intensity_image=None, aligned_mask_image=None,
                 transform=None):
        self.intensity_image = intensity_image
        self.mask_image = mask_image
        self.aligned_intensity_image = aligned_intensity_image
        self.aligned_mask_image = aligned_mask_image
        self.transform = transform
        
        self.imtype = itk.Image[itk.template(self.intensity_image)[1]]
        
    def to_principal_axes(self):
        imc = itk.ImageMomentsCalculator[self.imtype].New(self.intensity_image)
        imc.Compute()
        self.transform = imc.GetPhysicalAxesToPrincipalAxesTransform()
      
        #self.transform.SetCenter(imc.GetCenterOfGravity())
        #self.transform.SetTranslation([0,0,0])
        rsf = itk.ResampleImageFilter[self.imtype, self.imtype].New()
        rsf.SetInput(self.intensity_image)
        rsf.SetReferenceImage(self.intensity_image)
        rsf.UseReferenceImageOn()

        interp = itk.LinearInterpolateImageFunction[self.imtype, itk.D].New()
        rsf.SetInterpolator(interp)
        rsf.SetTransform(self.transform)

        rsf.Update()
        self.aligned_intensity_image = rsf.GetOutput()

        if self.mask_image:
            rsf.SetInput(self.mask_image)
            interp = itk.NearestNeighborInterpolateImageFunction[self.imtype,itk.D].New()
            rsf.SetInterpolator(interp)
            rsf.Update()
            self.aligned_mask_image = rsf.GetOutput()
        else:
            self.segment_aligned()

    def segment_aligned(self):
        otsu = itk.OtsuThresholdImageFilter[self.imtype,self.imtype].New()
        otsu.SetInput(self.aligned_intensity_image)
        otsu.Update()
        self.aligned_mask_image = otsu.GetOutput()
