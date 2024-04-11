import numpy as np
import scipy as sp
import itk

class Volume:
    def __init__(self,data_intensity=None,data_segmented=None,projection_type="hammer",intensity_method="mean"):
        self.data_intensity = data_intensity
        self.data_segmented = data_segmented
        self.projection_type = projection_type
        self.intensity_method = intensity_method
        
        self._extractor = None
        
    def chart(self):
        for z,(im,seg) in enumerate(zip(self.data_intensity,self.data_segmented)):
            sl = Slice(im,seg,intensity_method=self.intensity_method)
            f,s = sl.process()

    def extract_slices(self,z):
        for vol in [self.data_intensity,self.data_segmented]:
            imtype = itk.template(vol)
            eif = itk.ExtractImageFilter.New(vol)
            eif.SetDirectionCollapseToSubmatrix()
            ir = vol.GetBufferedRegion()
            sz = ir.GetSize()
            sz[2] = 1
            start = ir.GetIndex()
   
    @staticmethod
    def _slice_extractor(data):
        eif = itk.ExtractImageFilter.New(data)
        eif.SetDirectionCollapseToSubmatrix()
        ir = data.GetBufferedRegion()
        sz = ir.GetSize()
        sz[2] = 1
        start = ir.GetIndex()
        
        def _slice_extractor_helper(z):
            start[2] = z
            tr = ir
            tr.SetSize(sz)
            tr.SetIndex(start)
            eif.SetExtractionRegion(tr)
            eif.Update()
            return eif.GetOutput()

        return _slice_extractor_helper

class Slice:
    def __init__(self,im_intensity,im_segmented,intensity_method="mean"):
        self.im_intensity = im_intensity
        self.im_segmented = im_segmented
        self.intensity_method = intensity_method

    def process(self):
        pass

    def prepocess(self):
        pass

    def get_boundary(self):
        imtype = itk.template(self.im_segmented)
        cte = itk.ContourExtractor2DImageFilter[imtype].New(im_segmented)
        cte.Update()
        contours = cte.GetOutput()
        contour = contours.GetContour(0)
        x,y = contour.GetVertices()
        return x,y

class Boundary:
    def __init__(self,x=(),y=(),dr=2,dtheta=2,eps=1e-6,di=2):
        self.x = x
        self.y = y
        self.dr = dr
        self.dtheta = dtheta
        self.eps = eps # Tolerance for checking if a number is zero 
        self.di = di # \pm di points to use around the point of interest for tangent calculation
        self.n = len(self.x)

    def process(self):
        S = self.arclength()
        wedge_pts = []
        for i in range(self.n):;
            idx = self.get_circ_idx(i)
            xs = self.x[idx]
            ys = self.y[idx]
            wedge_pts.append(self.generate_wedge_points(xs,ys,self.dr,self.dtheta,self.eps))
        return S, wedge_pts
        
    def get_circ_idx(self,i):
        idx = []
        for j in range(i-self.di,i+self.di+1):
            idx.append(j % self.n)
        return idx
    
    def arclength(self):
        return self.xy_to_arclength(self.x,self.y)

    @staticmethod
    def xy_to_arclength(x_,y_):
        arclength = np.zeros(len(x_))
        for i in range(1,len(x_)):
            arclength[i] = arclength[i-1] + np.sqrt((x_[i]-x_[i-1])**2 + (y_[i]-y_[i-1])**2)
        return arclength

    @staticmethod
    def generate_wedge_points(xs=None,ys=None,dr=2,dtheta=2,eps=1e-6):
        midpt = int(len(xs)/2)
        xmid = xs[midpt]
        ymid = ys[midpt]
        
        if len(xs)==2:
            dxs = xs[1]-xs[0]
            dys = ys[1]-ys[0]
        else:
            # Use spline interpolation to calculate tangent
            xs = np.array(xs)
            ys = np.array(ys)
            s = xy_to_arclength(xs,ys)
            spx = sp.interpolate.UnivariateSpline(arclength,xs)
            spy = sp.interpolate.UnivariateSpline(arclength,ys)
            dxs = spx.derivative()(arclength[midpt])
            dys = spy.derivative()(arclength[midpt])

        norm = np.sqrt(dxs**2 + dys**2)
        dxs = dxs/norm
        dys = dys/norm
        
        # Coefficients of the normal line: c11*x + c12*y + c00 = 0
        c11 = -dys
        c12 = dxs
        c00 = -x0*c11 - y0*c12

        # Origin of second coordinate system translated so that the normal passes through (0,0)
        if np.abs(c12) < eps: # Line is vertical
            x0 = -c00/c11
            y0 = 0
        else:
            x0 = 0
            y0 = -c00/c12
        
        x_ = xmid - x0
        y_ = ymid - y0

        # Polar coordinates of the midpoint in second coordinate system
        r_ = np.sqrt(x_**2 + y_**2)
        theta_ = np.arctan2(y_,x_)

        # Get extents of wedge and query x_,y_ points to see if contained in wedge
        theta_exts = theta_ + np.array([-dtheta/2,dtheta/2])
        r_exts = r_ + np.array([-dr/2,dr/2])

        x_exts = r_exts[:,None]*np.cos(theta_exts)[None,:]
        y_exts = r_exts[:,None]*np.sin(theta_exts)[None,:]
        
        xm, xM = int(x_exts.min()), int(x_exts.max())
        ym, yM = int(y_exts.min()), int(y_exts.max())

        X_, Y_ = np.meshgrid(np.arange(xm,xM+1),np.arange(ym,yM+1))
        X_, Y_ = X_.flatten(), Y_.flatten()
        wedge_pts = []
        for i in range(len(X_)):
            xi = X_[i]
            yi = Y_[i]
            r = np.sqrt(xi**2 + yi**2)
            theta = np.arctan2(yi,xi)
            if r >= r_exts[0] and r <= r_exts[1] and theta >= theta_exts[0] and theta <= theta_exts[1]:
                wedge_pts.append((int(xi+x0),int(yi+y0))) #wedge points in original coordinate system
        
        wedge_pts = np.array(wedge_pts)
        return wedge_pts
        
    @staticmethod
    def hammer_projection(phi,lam):
        x = 2*np.sqrt(2)*np.cos(phi)*np.sin(lam/2)/(1+np.cos(phi)*np.cos(lam/2))**0.5
        y = np.sqrt(2)*np.sin(phi)/(1+np.cos(phi)*np.cos(lam/2))**0.5
        return x,y

    @staticmethod
    def process_slice(im_segmented,im_intensity):
        imtype = itk.template(im_segmented)
        cte = itk.ContourExtractor2DImageFilter[imtype].New(im_segmented)
        cte.Update()
        contours = cte.GetOutput()
        n = contours.GetNumberOfContours()
        if n != 1:
            raise ValueError('Number of segmented objects  must be 1')
        contour = contours.GetContour(0)


