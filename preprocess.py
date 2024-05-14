import itk
import numpy as np
import scipy as sp
import logging
import argparse
import pickle

logfmt = '%(asctime)s [%(levelname)s] %(funcName)s %(message)s'
logging.basicConfig(format=logfmt)
logger = logging.getLogger('preprocess')
logger.setLevel(logging.DEBUG)

class Preprocess:
    def __init__(self, intensity_image=None, mask_image=None, aligned_intensity_image=None, aligned_mask_image=None,
                 fwd_transform=None,bwd_transform=None):
        self.intensity_image = intensity_image
        self.mask_image = mask_image
        self.aligned_intensity_image = aligned_intensity_image
        self.aligned_mask_image = aligned_mask_image
        self.fwd_transform = transform
        self.bwd_transform = None 
        self.fwd_origin = [0,0,0] 
        
        imtemplate = itk.template(self.intensity_image)
        self.imtype = itk.Image[imtemplate[1]]
        self.pixtype = imtemplate[1][0]
        self.labelmaptype = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,imtemplate[1][1]]]

    def to_principal_axes(self,use_mask=False,bounding_box_padding=1):
        if not self.mask_image:
            logger.debug('Creating mask image')
            self.mask_image = self.segment(self.intensity_image)
            logger.debug('...done')

        logger.debug('Calculating bounding box extents')
        slf = itk.BinaryImageToShapeLabelMapFilter[self.imtype,self.labelmaptype].New()
        slf.SetInput(self.mask_image)
        slf.Update()
        labelmap = slf.GetOutput()
        bb = []
        area = []
        for i in range(1,labelmap.GetNumberOfLabelObjects()+1):
            b = labelmap.GetLabelObject(i).GetBoundingBox()
            a = labelmap.GetLabelObject(i).GetPhysicalSize()
            area.append(a)
            bb.append(b)
        area = np.array(area)
        idx = np.argmax(area)
        bb = bb[idx]
        b0 = np.array(bb.GetIndex()) - bounding_box_padding
        b1 = np.array(bb.GetUpperIndex()) + 1 + bounding_box_padding
        b0 = np.maximum(b0,0)
        b1 = np.minimum(b1,np.array(self.intensity_image.GetLargestPossibleRegion().GetSize()))
        logger.debug('...done')

        logger.debug('Calculating transformation to principal axes')
        if use_mask:
            imc = itk.ImageMomentsCalculator[self.imtype].New(self.mask_image)
        else:
            imc = itk.ImageMomentsCalculator[self.imtype].New(self.intensity_image)
        imc.Compute()
        logger.debug('...done')
        fwd = imc.GetPhysicalAxesToPrincipalAxesTransform()
        bwd = imc.GetPrincipalAxesToPhysicalAxesTransform()
        CoG = imc.GetCenterOfGravity()
        
        logger.debug('Transforming image extents')
        exts0 = np.meshgrid((b0[0],b1[0]),(b0[1],b1[1]),(b0[2],b1[2]))
        exts0 = np.array(exts0).reshape(3,-1).T
        exts1 = []
        for ext in exts0:
            exts1.append(fwd.TransformPoint([float(i) for i in ext]))
        exts1 = np.array(exts1)
        m = np.min(exts1,axis=0)
        M = np.max(exts1,axis=0)
        dims = M-m
        m = [float(i) for i in m]
        dims = [int(d) for d in dims]
        logger.debug('...done')
       
        self.fwd_origin = m
        self.fwd_transform = fwd
        self.bwd_transform = bwd
       
        logger.debug(f'Creating reference image of size = {dims}')
        refim = self.mkreferenceim(dims,origin=self.fwd_origin)
        logger.debug('...done')

        intensity_out, mask_out = self.resample(self.bwd_transform,refim)
        self.aligned_intensity_image = intensity_out
        self.aligned_mask_image = mask_out

    def resample(self,transform,refim=None):
        if not refim:
            refim = self.mkreferenceim(self.aligned_intensity_image.GetLargestPossibleRegion().GetSize())
       
        logger.debug('Resampling intensity image using given transform')
        rsf = itk.ResampleImageFilter[self.imtype,self.imtype].New()
        rsf.SetInput(self.intensity_image)
        rsf.SetReferenceImage(refim)
        rsf.UseReferenceImageOn()

        interp = itk.LinearInterpolateImageFunction[self.imtype,itk.D].New()
        rsf.SetInterpolator(interp)
        rsf.SetTransform(transform)
        rsf.Update()

        intensity_out =  rsf.GetOutput()
        logger.debug('...done')
        logger.debug('Segmenting resampled intensity image')
        mask_out = self.segment(intensity_out)
        logger.debug('...done')
        return intensity_out, mask_out

    def mkreferenceim(self,lwh,origin=[0,0,0]):
        im = self.imtype.New()
        region = im.GetLargestPossibleRegion()
        region.SetSize(lwh)
        im.SetRegions(region)
        im.Allocate()
        im.SetOrigin(origin)
        return im

    def segment(self,im):
        thresh = itk.TriangleThresholdImageFilter[self.imtype,self.imtype].New()
        thresh.SetInput(im)
        thresh.SetOutsideValue(itk.NumericTraits[self.pixtype].max())
        thresh.SetInsideValue(0)
        thresh.Update()

        setype = itk.FlatStructuringElement[self.imtype.GetImageDimension()]
        se_close = setype.Ball(5)
        se_dilate = setype.Ball(1)

        bcf = itk.BinaryMorphologicalClosingImageFilter[self.imtype,self.imtype,setype].New()
        bcf.SetInput(thresh.GetOutput())
        bcf.SetKernel(se_close)
        bcf.Update()
        
        #bdf = itk.BinaryDilateImageFilter[self.imtype,self.imtype,setype].New()
        #bdf.SetInput(bcf.GetOutput())
        #bdf.SetKernel(se_dilate)
        #bdf.Update()

        bhf = itk.BinaryFillholeImageFilter[self.imtype].New()
        bhf.SetInput(bcf.GetOutput())
        bhf.Update()

        out = bhf.GetOutput()
        return out


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Preprocess image')
    parser.add_argument('intensity_image',type=str, help='Path to intensity image')
    parser.add_argument('--mask',type=str,default=None, help='Path to mask image')
    args = parser.parse_args()
    
    intensity_image = itk.imread(args.intensity_image)
    mask_image = itk.imread(args.mask) if args.mask else None
    
    p = Preprocess(intensity_image,mask_image)
    p.to_principal_axes()

    logger.info('Saving principal axes transformation')
    with open('fwd_transform.pkl','wb') as f:
        pickle.dump(p.fwd_transform,f,protocol=pickle.HIGHEST_PROTOCOL)
    with open('fwd_origin.pkl','wb') as f:
        pickle.dump(p.fwd_origin,f,protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Writing aligned images')
    itk.imwrite(p.aligned_intensity_image,'aligned_intensity_image.tif')
    itk.imwrite(p.aligned_mask_image,'aligned_mask_image.tif')
    logger.info('Preprocessing complete')
