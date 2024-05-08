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
        
    def to_principal_axes(self,use_mask=True):
        if use_mask:
            if not self.mask_image:
                self.mask_image = self.segment(self.intensity_image)
            imc = itk.ImageMomentsCalculator[self.imtype].New(self.mask_image)
        else:
            imc = itk.ImageMomentsCalculator[self.imtype].New(self.intensity_image)
        imc.Compute()
        T0 = imc.GetPhysicalAxesToPrincipalAxesTransform()
        
        T1 = T0.Clone()
        T1.SetTranslation([0,0,0])

        l,w,h = self.intensity_image.GetLargestPossibleRegion().GetSize()
        exts0 = np.meshgrid(range(l),range(w),range(h))
        exts0 = np.array(exts).reshape(3,-1).T
        
        exts1 = []
        for ext in exts0:
            exts1.append(T1.TransformPoint(ext))
        exts1 = np.array(exts1)
        m = np.min(exts,axis=0)
        M = np.max(exts,axis=0)
        dims = M-m
        dims = [int(d) for d in dims]

        T2 = T0.Clone()
        T2.SetTranslation([-int(i) for i in m])
        self.transform = T2
      
        refim = self.mkreferenceim(dims)

        rsf = itk.ResampleImageFilter[self.imtype,self.imtype].New()
        rsf.SetInput(self.intensity_image)
        rsf.SetReferenceImage(refim)
        rsf.UseReferenceImageOn()

        interp = itk.LinearInterpolateImageFunction[self.imtype,itk.D].New()
        rsf.SetInterpolator(interp)
        rsf.SetTransform(self.transform)
        rsf.Update()

        self.aligned_intensity_image = rsf.GetOutput()
        self.aligned_mask_image = self.segment(self.aligned_intensity_image)
    
    def mkreferenceim(self,lwh):
        im = self.imtype.New()
        region = im.GetLargestPossibleRegion()
        region.SetSize(lwh)
        im.SetRegions(region)
        im.Allocate()
        return im

    def segment(self,im):
        otsu = itk.OtsuThresholdImageFilter[self.imtype,self.imtype].New()
        otsu.SetInput(im)
        otsu.Update()
        return otsu.GetOutput()
