import numpy as np
import scipy as sp
import itk
import matplotlib.pyplot as plt
import argparse
import logging
from time import sleep

logfmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(format=logfmt)
logger = logging.getLogger('charts')
logger.setLevel(logging.DEBUG)

class Volume:
    def __init__(self,data_intensity=None,data_segmented=None,projection_type="hammer",intensity_method="max",nS=100):
        self.data_intensity = data_intensity
        self.data_segmented = data_segmented
        self.projection_type = projection_type
        self.intensity_method = intensity_method
        self.nS = nS

        self.centroid = self.get_centroid()
        self.chart_vals = {}

    def chart(self,show_all=True,chart_kwargs={}):
        VALS = self.get_chart_vals(**chart_kwargs)
        if VALS['X'] is None:
            nplots = 2
        else:
            nplots = 3
        if not show_all:
            nplots = 1
        
        fig0,ax0 = plt.subplots(1,nplots)
        fig1,ax1 = plt.subplots(1,nplots)
        plt.inferno()
        if nplots>1:
            self.plot_chart(ax0[nplots-2],X=VALS['S'],Y=VALS['Z'],F=VALS['F'],xlabel='arclength',ylabel='slice',title='Intensity')
            self.plot_chart(ax1[nplots-2],X=VALS['S'],Y=VALS['Z'],F=VALS['dV'],xlabel='arclength',ylabel='slice',title='Surface Area')
            self.plot_chart(ax0[nplots-1],X=VALS['L'],Y=VALS['P'],F=VALS['F'],xlabel='lambda',ylabel='phi',title='Intensity')
            self.plot_chart(ax1[nplots-1],X=VALS['L'],Y=VALS['P'],F=VALS['dV'],xlabel='lambda',ylabel='phi',title='Surface Area')
        else:
            ax0 = [ax0]
            ax1 = [ax1]

        self.plot_chart(ax0[0],X=VALS['X'],Y=VALS['Y'],F=VALS['F'],xlabel=f'x_{self.projection_type}',ylabel=f'y_{self.projection_type}',title='Intensity')
        self.plot_chart(ax1[0],X=VALS['X'],Y=VALS['Y'],F=VALS['dV'],xlabel=f'x_{self.projection_type}',ylabel=f'y_{self.projection_type}',title='Surface Area') 
        plt.show()

    @staticmethod
    def plot_chart(ax,X,Y,F,xlabel='x',ylabel='y',title=None):
        ax.pcolormesh(X,Y,F)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    def get_chart_vals(self,z0=0,z1=None):
        if z1 is None:
            z1 = self.data_intensity.shape[0]
        L = []
        F = []
        Z = []
        S = []
        P = []
        dV = []
        for z in range(z0,z1):
            logger.info('Processing slice %d',z)
            im = self.intensity_slice(z)
            seg = self.segmented_slice(z)
            sl = Slice(im,seg,intensity_method=self.intensity_method,xy_centroid=self.centroid,nS=self.nS)
            s,f,r,dv = sl.process()
            if s is None:
                continue
            else:
                lam = self.s_to_lamda(s)
                phi,z_ = self.zr_to_phi(z,r)
                L.append(lam)
                Z.append(z_*np.ones(len(lam)))
                F.append(f)
                P.append(phi)
                dV.append(dv)
                S.append(s)

        S = self.center_s(S)
        if self.projection_type.lower()=="hammer":
            X,Y = self.hammer_projection(P,L)
        else:
            X,Y = None,None
    
        D = {'X':X,'Y':Y,'F':F,'Z':Z,'P':P,'L':L,'S':S,'dV':dV}
                
        return D

    def intensity_slice(self,z):
        h,l,w = self.data_intensity.shape
        im = itk.Image[itk.UC,2].New()
        start = itk.Index[2]([0,0])
        size = itk.Size[2]([w,l])
        region = itk.ImageRegion[2](start,size)
        im.SetRegions(region)
        im.Allocate()
        for i in range(w):
            for j in range(l):
                im.SetPixel(itk.Index[2]([i,j]),self.data_intensity.GetPixel(itk.Index[3]([i,j,z])))
        return im

    def segmented_slice(self,z):
        h,l,w = self.data_segmented.shape
        im = itk.Image[itk.UC,2].New()
        start = itk.Index[2]([0,0])
        size = itk.Size[2]([w,l])
        region = itk.ImageRegion[2](start,size)
        im.SetRegions(region)
        im.Allocate()
        for i in range(w):
            for j in range(l):
                im.SetPixel(itk.Index[2]([i,j]),self.data_segmented.GetPixel(itk.Index[3]([i,j,z])))
        return im
    
    def zr_to_phi(self,z,R):
        z_ = z - self.centroid[2]
        phi = self._zr_to_phi(z_,R)
        return phi,z_

    def get_centroid(self):
        centroid = self.get_shape_statistic("GetCentroid")
        return centroid

    def get_shape_statistic(self,attr="GetCentroid",all_labels=False):
        return self._get_shape_statistic(self.data_segmented,attr=attr,im=self.data_intensity,all_labels=all_labels)
    @staticmethod
    def _get_shape_statistic(seg,attr="GetCentroid",im=None,all_labels=False):
        if im is None:
            im = seg
        bsf = itk.BinaryImageToStatisticsLabelMapFilter.New(seg,im)
        bsf.Update()
        slm = bsf.GetOutput()

        areas = []
        for i in slm.GetLabels():
            areas.append(slm[i].GetPhysicalSize())
        
        #Flag to return all label attributes or only that belonging to the largest object
        if all_labels:
            ids = slm.GetLabels()
        else:
            ids = [ slm.GetLabels()[np.argmax(areas)] ]
        
        stats = []
        for i in ids:
            stats.append( getattr(slm[i],attr)() )

        if len(stats)==1:
            stats = stats[0]
        stats = np.array(stats)
        return stats
            
    @staticmethod
    def hammer_projection(phi,lam):
        if type(phi) == list:
            phi = np.array(phi)
        if type(lam) == list:
            lam = np.array(lam)
        x = 2*np.sqrt(2)*np.cos(phi)*np.sin(lam/2)/(1+np.cos(phi)*np.cos(lam/2))**0.5
        y = np.sqrt(2)*np.sin(phi)/(1+np.cos(phi)*np.cos(lam/2))**0.5
        return x,y

    @staticmethod
    def s_to_lamda(s):
        lam = 2*np.pi*(s/s[-1]-0.5)
        return lam
    
    @staticmethod
    def center_s(S):
        S_ = []
        for s in S:
            midpt = int(len(s)/2)
            s_ = s - s[midpt]
            S_.append(s_)
        return S_

    @staticmethod
    def _zr_to_phi(Z,R=None):
        if R is None:
            zm,zM = Z.min(),Z.max()
            phi = np.pi*((Z-zm)/(zM-zm)-0.5)
        else:
            phi = np.arctan2(Z,R)
        return phi

class Slice:
    def __init__(self,im_intensity,im_segmented,xy_centroid=(0,0),intensity_method="max",nS=100):
        self.im_intensity = im_intensity
        self.im_segmented = im_segmented
        self.centroid = xy_centroid
        self.intensity_method = intensity_method
        self.nS = nS

        self.boundary = self.get_boundary()

    def process(self):
        if len(x)==0:
            S,F,R,dV = None,None,None,None

        else:
            logger.debug('Processing boundary')
            S, R, wedge_pts = self.boundary.process()
            F = []
            dV = []
            for i in range(len(wedge_pts)):
                dV.append(len(wedge_pts[i]))
                F.append(self.get_intensity(wedge_pts[i]))
            S_ = S
            S,F = self.resample(S_,F)
            _,R = self.resample(S_,R)
            _,dV = self.resample(S_,dV)

        return S,F,R,dV

    def get_boundary(self):
        x,y = self.get_contour()
        b = Boundary(x,y,self.centroid)
        return b

    def resample(self,S,F):
        if type(S) == list:
            S = np.array(S)
        if type(F) == list:
            F = np.array(F)
        S_interp = np.linspace(S[0],S[-1],self.nS)
        F_interp = np.interp(S_interp,S,F)
        return S_interp,F_interp

    def prepocess(self):
        pass

    @staticmethod
    def standardize_boundary_index(x,y,centroid=(0,0)):
        mux,muy = centroid[0],centroid[1] 
        x_ = np.abs(x-mux)
        y_ = y-muy
        L = np.where(y_>0,x_,np.inf)
        id0 = np.where(L==L.min())[0][0]
        x = np.roll(x,-id0)
        y = np.roll(y,-id0)
        return x,y

    def get_contour(self):
        logger.debug('Extracting contours')
        mmf = itk.MinimumMaximumImageFilter.New(self.im_segmented)
        mmf.Update()
        max_val = mmf.GetMaximum()
        if max_val == 0:
            x,y = [],[]
            logger.debug('All zero image: skipping slice')
        else:
            logger.debug('Extracting contours')
            imtype = itk.template(self.im_segmented)
            cte = itk.ContourExtractor2DImageFilter.New(self.im_segmented)
            cte.SetContourValue(254)
            cte.Update()
            contours = cte.GetOutput()
            if contours is None:
                x,y = [],[]
                logger.debug('No contours found')
            else:
                vertices = contours.GetVertexList()
                x = []
                y = []
                for i in range(vertices.Size()):
                    xy = vertices.GetElement(i)
                    x.append(int(xy[0]))
                    y.append(int(xy[1]))
            logger.debug('Extracted %d contour points',len(x))
        return x,y

    def get_intensity(self,pts):
        if len(pts)==0:
            return 0
        else:
            f = np.zeros(len(pts))
            for i in range(len(pts)):
                x,y = pts[i]
                f[i] = self.im_intensity.GetPixel(itk.Index[2]([int(x),int(y)]))
            
            if self.intensity_method.lower() == "mean":
                f = np.mean(f)
            elif self.intensity_method.lower() == "max":
                f = np.max(f)
            else:
                raise ValueError('Intensity method not recognized')

        return f

class Boundary:
    def __init__(self,x=(),y=(),centroid=(0,0),dr=2,dtheta=2,eps=1e-6,di=2):
        x_,y_ = self.unique_points(x,y)
        self.x = x_
        self.y = y_
        self.centroid = centroid
        self.dr = dr
        self.dtheta = dtheta
        self.eps = eps # Tolerance for checking if a number is zero 
        self.di = di # \pm di points to use around the point of interest for tangent calculation
        self.n = len(self.x)
    
    @staticmethod
    def unique_points(x,y):
        pts = np.array(list(zip(x,y)))
        ids = np.unique(pts,axis=0,return_index=True)[1]
        pts = np.array([pts[i] for i in sorted(ids)])
        return pts[:,0],pts[:,1]

    def process(self):
        S = self.arclength()
        R = self.magnitudeR()
        wedge_pts = []
        for i in range(self.n):
            idx = self.get_circ_idx(i)
            xs = self.x[idx]
            ys = self.y[idx]
            wedge_pts.append(self.generate_wedge_points(xs,ys))
        return S, R, wedge_pts
        
    def get_circ_idx(self,i):
        idx = []
        for j in range(i-self.di,i+self.di+1):
            idx.append(j % self.n)
        return idx
    
    def arclength(self):
        S = self.xy_to_arclength(self.x,self.y)
        return S

    def magnitudeR(self):
        R = np.sqrt( (self.x-self.centroid[0])**2 + (self.y-self.centroid[1])**2 )
        return R

    @staticmethod
    def xy_to_arclength(x_,y_):
        arclength = np.zeros(len(x_))
        for i in range(1,len(x_)):
            arclength[i] = arclength[i-1] + np.sqrt((x_[i]-x_[i-1])**2 + (y_[i]-y_[i-1])**2)
        return arclength

    def generate_wedge_points(self,xs=None,ys=None):
        dr = self.dr
        dtheta = self.dtheta
        eps = self.eps
        centroid = self.centroid

        #Put centroid at origin 
        xs = xs - centroid[0]
        ys = ys - centroid[1]

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
            s = self.xy_to_arclength(xs,ys)
            spx = sp.interpolate.UnivariateSpline(s,xs)
            spy = sp.interpolate.UnivariateSpline(s,ys)
            dxs = spx(s[midpt],1)
            dys = spy(s[midpt],1)
        
        norm = np.sqrt(dxs**2 + dys**2)
        dxs = dxs/norm
        dys = dys/norm
        
        # Coefficients of the normal line: c11*x + c12*y + c00 = 0
        c11 = dys
        c12 = -dxs
        c00 = -xmid*c11 - ymid*c12

        # Origin of second coordinate system translated so that the normal passes through (0,0)
        if np.abs(c12) < eps: #Normal line is vertical: move line along x to origin
            x0 = -c00/c11
            y0 = 0
        else:
            x0 = 0
            y0 = -c00/c12
        
        x_ = xmid - x0
        y_ = ymid - y0
        
        # Polar coordinates of the midpoint in second coordinate system
        r_ = np.sqrt(x_**2 + y_**2)
        theta_ = np.arctan2(x_,y_)

        # Get extents of wedge and query x_,y_ points to see if contained in wedge
        dtheta_rad = np.deg2rad(dtheta)
        theta_exts = theta_ + np.array([-dtheta_rad/2,dtheta_rad/2])
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
                wedge_pts.append((int(xi+x0+centroid[0]),int(yi+y0+centroid[1]))) #wedge points in original coordinate system
        
        wedge_pts = np.array(wedge_pts).astype(int)
        return wedge_pts
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create a chart of a volume')
    parser.add_argument('data_intensity', type=str, help='Path to the intensity data')
    parser.add_argument('data_segmented', type=str, help='Path to the segmented data')
    parser.add_argument('--projection_type', type=str, default='hammer', help='Type of projection to use')
    parser.add_argument('--intensity_method', type=str, default='mean', help='Method to calculate intensity')
    parser.add_argument('--nS', type=int, default=100, help='Number of arclength points to use for the chart')
    parser.add_argument('--z0', type=int, default=0, help='Starting slice')
    parser.add_argument('--z1', type=int, default=None, help='Ending slice')
    args = parser.parse_args()

    data_intensity = itk.imread(args.data_intensity)
    data_segmented = itk.imread(args.data_segmented)
    v = Volume(data_intensity=data_intensity,data_segmented=data_segmented,projection_type=args.projection_type,intensity_method=args.intensity_method,nS=args.nS)
    v.chart(chart_kwargs={'z0':args.z0,'z1':args.z1})
