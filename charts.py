import numpy as np
import scipy as sp
import itk

class Volume:
    def __init__(self,data_intensity=None,data_segmented=None,projection_type="hammer",intensity_method="mean"):
        self.data_intensity = data_intensity
        self.data_segmented = data_segmented
        self.projection_type = projection_type
        self.intensity_method = intensity_method

    def chart(self):
        for z,(im,seg) in enumerate(zip(self.data_intensity,self.data_segmented)):
            sl = Slice(im,seg,intensity_method=self.intensity_method)
            f,s = sl.process()
            
class Slice:
    def __init__(self,intensity_im,segmented_im,intensity_method="mean"):
        self.intensity_im = intensity_im
        self.segmented_im = segmented_im
        self.intensity_method = intensity_method

    def process(self):
        pass

class Boundary:
    def __init__(self,x=None,y=None,dr=5,dtheta=5,eps=1e-6):
        self.x = x
        self.y = y
        self.dr = dr
        self.dtheta = dtheta
        self.eps = eps # Tolerance for checking if a number is zero 

    def process(self):
        pass
    
    def get_arclength(self):
        arclength = np.zeros(len(self.x))
        for i in range(1,len(self.x)):
            arclength[i] = arclength[i-1] + np.sqrt((self.x[i]-self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2)
        return arclength
    
    def generate_wedge_points(self):
        midpt = int(len(self.x)/2)
        x0 = self.x[midpt]
        y0 = self.y[midpt]
        
        if len(x)==2:
            dx = x[1]-x[0]
            dy = y[1]-y[0]
        else:
            # Use spline interpolation to calculate tangent
            x = np.array(x)
            y = np.array(y)
            arclength = xy_to_arclength(x,y)
            spx = sp.interpolate.UnivariateSpline(arclength,x)
            spy = sp.interpolate.UnivariateSpline(arclength,y)
            dx = spx.derivative()(arclength[midpt])
            dy = spy.derivative()(arclength[midpt])

        norm = np.sqrt(dx**2 + dy**2)
        dx = dx/norm
        dy = dy/norm
        
        # Coefficients of the normal line: c11*x + c12*y + c00 = 0
        c11 = -dy
        c12 = dx
        c00 = -x0*c11 - y0*c12

        # Get polar coordinates from new coordinate system where the normal line passes through the origin
        if np.abs(c11) < eps:
            # Line is horizontal
            x_ = x0
            y_ = y0 - c00/c12
        elif np.abs(c12) < eps:
            # Line is vertical
            y_ = y0
            x_ = x0 - c00/c11
        else:
            y_ = 0
            x_ = -c00/c11
        
        r_ = np.sqrt(x_**2 + y_**2)
        theta_ = np.arctan2(y_,x_)
        
        def g(dr,dtheta=0,ndr=10,ndtheta=1):
            R_ = np.linspace(r_-dr/2,r_+dr/2,ndr)
            Th_ = np.linspace(theta_-dtheta/2,theta_+dtheta/2,ndtheta)
            X_ = R_[:,None]*np.cos(Th_)[None,:]
            Y_ = R_[:,None]*np.sin(Th_)[None,:]
            X = X_ + x_
            Y = Y_ + y_
            return X,Y

        return g

        
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


