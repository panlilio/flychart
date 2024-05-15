import numpy as np
import scipy as sp
import itk
import matplotlib.pyplot as plt
import argparse
import logging
from time import sleep
import pickle

logfmt = '%(asctime)s [%(levelname)s] %(funcName)s %(message)s'
logging.basicConfig(format=logfmt)
logger = logging.getLogger('charts')
logger.setLevel(logging.DEBUG)

class Volume:
    def __init__(self,data_intensity=None,data_segmented=None,projection_type="hammer",intensity_method="max",nS=100,
                 boundary_kwargs={}):
        self.data_intensity = data_intensity
        self.imtype = itk.template(self.data_intensity)
        self.data_segmented = self.recast(data_segmented)
        self.segtype = itk.template(self.data_segmented)
        self.projection_type = projection_type
        self.intensity_method = intensity_method
        self.nS = nS
        self.boundary_kwargs = boundary_kwargs
        
        self.centroid = self.get_centroid()

    def chart(self,show_all=True,z0=0,z1=None,return_chart_values=True):
        VALS = self.get_chart_vals(z0=z0,z1=z1)
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
            c0 = self.plot_chart(ax0[nplots-2],X=VALS['S'],Y=VALS['Z'],F=VALS['F'],xlabel='arclength',ylabel='slice',title='Intensity')
            c1 = self.plot_chart(ax1[nplots-2],X=VALS['S'],Y=VALS['Z'],F=VALS['dV'],xlabel='arclength',ylabel='slice',title='Surface Area')
            self.plot_chart(ax0[nplots-1],X=VALS['L'],Y=VALS['P'],F=VALS['F'],xlabel='lambda',ylabel='phi',title='Intensity')
            self.plot_chart(ax1[nplots-1],X=VALS['L'],Y=VALS['P'],F=VALS['dV'],xlabel='lambda',ylabel='phi',title='Surface Area')
        else:
            ax0 = [ax0]
            ax1 = [ax1]
        
        if VALS['X'] is not None:
            c0 = self.plot_chart(ax0[0],X=VALS['X'],Y=VALS['Y'],F=VALS['F'],xlabel=f'x_{self.projection_type}',ylabel=f'y_{self.projection_type}',title='Intensity')
            c1 = self.plot_chart(ax1[0],X=VALS['X'],Y=VALS['Y'],F=VALS['dV'],xlabel=f'x_{self.projection_type}',ylabel=f'y_{self.projection_type}',title='Surface Area') 
        
        fig0.subplots_adjust(left=0.05,right=0.9)
        fig1.subplots_adjust(left=0.05,right=0.9)
        cax0 = fig0.add_axes([0.92,0.15,0.02,0.7])
        cax1 = fig1.add_axes([0.92,0.15,0.02,0.7])
        fig0.colorbar(c0,cax=cax0)
        fig1.colorbar(c1,cax=cax1)
        
        plt.show()
        if not return_chart_values:
            return ax0,ax1
        else:
            return VALS

    def recast(self,seg):
        segtype = itk.template(seg)
        if segtype[1][0] != self.imtype[1][0]:
            rf = itk.RescaleIntensityImageFilter[itk.Image[segtype[1]],itk.Image[self.imtype[1]]].New(seg)
            rf.SetOutputMinimum(0)
            rf.SetOutputMaximum(itk.NumericTraits[self.imtype[1][0]].max())
            rf.Update()
            out = rf.GetOutput()
        else:
            out = seg
        return out

    @staticmethod
    def plot_chart(ax,X,Y,F,xlabel='x',ylabel='y',title=None):
        c = ax.pcolormesh(X,Y,F)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        return c

    def get_chart_vals(self,z0=0,z1=None):
        if z1 is None:
            z1 = self.data_intensity.shape[0]
        
        nonempty = np.zeros(z1-z0).astype(bool)
        L = np.ndarray((z1-z0,self.nS),dtype=np.float64)
        F = np.ndarray((z1-z0,self.nS),dtype=np.float32)
        Z = np.ndarray((z1-z0,self.nS),dtype=np.int16)
        S = np.ndarray((z1-z0,self.nS),dtype=np.float64)
        P = np.ndarray((z1-z0,self.nS),dtype=np.float64)
        dV = np.ndarray((z1-z0,self.nS),dtype=np.uint8)
        x = np.ndarray((z1-z0,self.nS),dtype=np.float64)
        y = np.ndarray((z1-z0,self.nS),dtype=np.float64)

        for i,z in enumerate(range(z0,z1)):
            logger.info('PROCESSING SLICE %d ================',z)
            im = self.intensity_slice(z)
            seg = self.segmented_slice(z)
            sl = Slice(im,seg,intensity_method=self.intensity_method,xy_centroid=self.centroid,nS=self.nS,
                       boundary_kwargs=self.boundary_kwargs)
            s,f,r,dv,x_,y_ = sl.process()
            if s is None:
                continue
            else:
                nonempty[i] = True
                lam = self.s_to_lamda(s)
                phi,z_ = self.zr_to_phi(z,r)
                L[i] = lam
                F[i] = f
                Z[i] = z_*np.ones(self.nS)
                S[i] = s
                P[i] = phi
                dV[i] = dv
                x[i] = x_
                y[i] = y_

            logger.info('SLICE %d DONE ----------------------',z)

        L = L[nonempty]
        F = F[nonempty]
        Z = Z[nonempty]
        S = S[nonempty]
        P = P[nonempty]
        dV = dV[nonempty]
        x = x[nonempty]
        y = y[nonempty]

        S = self.center_s(S)
        if self.projection_type.lower()=="hammer":
            X,Y = self.hammer_projection(P,L)
        else:
            X,Y = None,None
    
        D = {'X':X,'Y':Y,'F':F,'Z':Z,'P':P,'L':L,'S':S,'dV':dV,'x':x,'y':y}
        return D

    def intensity_slice(self,z):
        pixtype = itk.template(self.data_intensity)[1][0]
        exf = itk.ExtractImageFilter[itk.Image[pixtype,3],itk.Image[pixtype,2]].New(self.data_intensity)
        exf.SetDirectionCollapseToSubmatrix()

        inregion = self.data_intensity.GetBufferedRegion()
        size = inregion.GetSize()
        size[2] = 0
        start = inregion.GetIndex()
        start[2] = z
        targetregion = inregion
        targetregion.SetSize(size)
        targetregion.SetIndex(start)
        exf.SetExtractionRegion(targetregion)
        exf.Update()

        outimage = exf.GetOutput()
        return outimage
    
    def segmented_slice(self,z):
        pixtype = itk.template(self.data_segmented)[1][0]
        exf = itk.ExtractImageFilter[itk.Image[pixtype,3],itk.Image[pixtype,2]].New(self.data_segmented)
        exf.SetDirectionCollapseToSubmatrix()

        inregion = self.data_segmented.GetBufferedRegion()
        size = inregion.GetSize()
        size[2] = 0
        start = inregion.GetIndex()
        start[2] = z
        targetregion = inregion
        targetregion.SetSize(size)
        targetregion.SetIndex(start)
        exf.SetExtractionRegion(targetregion)
        exf.Update()

        outimage = exf.GetOutput()
        return outimage

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
    def __init__(self,im_intensity,im_segmented,xy_centroid=(0,0),intensity_method="max",nS=100,boundary_kwargs={}):
        self.im_intensity = im_intensity
        self.im_segmented = im_segmented
        self.centroid = xy_centroid
        self.intensity_method = intensity_method
        self.nS = nS

        self.boundary_kwargs = boundary_kwargs
        self.boundary = self.get_boundary(boundary_kwargs)

    def process(self):
        if len(self.boundary.x)==0:
            S,F,R,dV,x,y = None,None,None,None,None,None
        else:
            S, R, wedge_pts = self.boundary.process()
            wedge_pts = self.clean_wedge_pts(wedge_pts)
            F = []
            dV = []
            for i in range(len(wedge_pts)):
                dV.append(len(wedge_pts[i]))
                F.append(self.get_intensity(wedge_pts[i]))
            S_ = S
            S,F = self.resample(S_,F)
            _,R = self.resample(S_,R)
            _,dV = self.resample(S_,dV)
            _,x = self.resample(S_,self.boundary.x)
            _,y = self.resample(S_,self.boundary.y)
        return S,F,R,dV,x,y
    
    def clean_wedge_pts(self,wedge_pts):
        wedge_pts_ = []
        for i in range(len(wedge_pts)):
            wp = []
            for j in range(len(wedge_pts[i])):
                x,y = wedge_pts[i][j]
                in_object = self.im_segmented.GetPixel(itk.Index[2]([int(x),int(y)]))
                if in_object:
                    wp.append((x,y))
            if len(wp)>0:
                wedge_pts_.append(wp)
        return wedge_pts_

    def get_boundary(self,boundary_kwargs):
        x,y = self.get_contour()
        x,y = self.standardize_contour_index(x,y,self.centroid)
        npts = int(self.nS*1.5)
        b = Boundary(x,y,self.centroid,npts=npts,**boundary_kwargs)
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
    def standardize_contour_index(x,y,centroid=(0,0)):
        if len(x)!=0:
            mux,muy = centroid[0],centroid[1] 
            x_ = np.abs(x-mux)
            y_ = y-muy
            L = np.where(y_>0,x_,np.inf)
            id0 = np.where(L==L.min())[0][0]
            x = np.roll(x,-id0)
            y = np.roll(y,-id0)
        return x,y

    def get_contour(self):
        mmf = itk.MinimumMaximumImageFilter.New(self.im_segmented)
        mmf.Update()
        max_val = mmf.GetMaximum()
        if max_val == 0:
            x,y = [],[]
            logger.debug('\tAll zero image: skipping slice')
        else:
            imtype = itk.template(self.im_segmented)
            cte = itk.ContourExtractor2DImageFilter.New(self.im_segmented)
            cte.SetContourValue(int(max_val-1))
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
                    x.append(xy[0])
                    y.append(xy[1])
            logger.debug('\tExtracted %d contour points',len(x))
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
    def __init__(self,x=(),y=(),centroid=(0,0),dN=4,dT=6,eps=1e-6,di=2,npts=200):
        self.npts = npts 
        x_,y_ = self.unique_points(x,y)
        x_,y_ = self.trim_points(x_,y_)
        self.x = x_
        self.y = y_
        self.centroid = centroid
        self.dN = dN
        self.dT = dT
        self.eps = eps # Tolerance for checking if a number is zero 
        self.di = di # \pm di points to use around the point of interest for tangent calculation
        self.n = len(self.x)

    @staticmethod
    def unique_points(x,y):
        if len(x)==0:
            return x,y
        else:
            pts = np.array(list(zip(x,y)))
            ids = np.unique(pts,axis=0,return_index=True)[1]
            pts = np.array([pts[i] for i in sorted(ids)])
            return pts[:,0],pts[:,1]

    def trim_points(self,x,y):
        if len(x)!=0:
            npts = np.minimum(self.npts,len(x))
            ids = np.linspace(0,len(x)-1,npts).astype(int)
            x = x[ids]
            y = y[ids]
        return x,y

    def process(self,test_pt=None):
        if test_pt is not None:
            ids = [test_pt]
        else:
            ids = np.arange(len(self.x))

        S = self.arclength()
        R = self.magnitudeR()
        wedge_pts = []
        logger.debug('Processing %d points',len(ids))
        for c,i in enumerate(ids):
            idx = self.get_circ_idx(i)
            xs = self.x[idx]
            ys = self.y[idx]
            wedge_pts.append(self.generate_wedge_points(xs,ys))
            if c % 10 == 0:
                logger.debug('Processed point %d',c)
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
        dN = self.dN
        dT = self.dT
        eps = self.eps
        centroid = self.centroid

        #Put centroid at origin 
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
       
        dxs,dys = (0,1) if np.abs(dxs)<eps else (dxs,dys)
        dxs,dys = (1,0) if np.abs(dys)<eps else (dxs,dys)

        # Normal and tangent vectors
        nv = np.array([-dys,dxs])
        tv = np.array([dxs,dys])
        
        if nv[1] < 0:
            nv = -nv
        if tv[1] < 0: 
            tv = -tv

        N_pos = self.linear_fun(xmid+dT/2*tv[0],ymid+dT/2*tv[1],nv[0],nv[1],pos=True)
        N_neg = self.linear_fun(xmid-dT/2*tv[0],ymid-dT/2*tv[1],nv[0],nv[1],pos=False)
        T_pos = self.linear_fun(xmid+dN/2*nv[0],ymid+dN/2*nv[1],tv[0],tv[1],pos=True)
        T_neg = self.linear_fun(xmid-dN/2*nv[0],ymid-dN/2*nv[1],tv[0],tv[1],pos=False)

        # Get extents of wedge
        v0 = np.array([xmid,ymid])
        V = v0[:,None]*np.ones((4))[None,:] + nv[:,None]*np.array([1,-1,1,-1])[None,:]*dN/2 + tv[:,None]*np.array([1,1,-1,-1])[None,:]*dT/2
        
        xm, xM = int(V[0,:].min()), int(V[0,:].max())
        ym, yM = int(V[1,:].min()), int(V[1,:].max())
        
        X_, Y_ = np.meshgrid(np.arange(xm,xM+1),np.arange(ym,yM+1))
        X_, Y_ = X_.flatten(), Y_.flatten()
        wedge_pts = []
        for i in range(len(X_)):
            xi = X_[i]
            yi = Y_[i]
            if N_neg(xi) <= yi <= N_pos(xi) and T_neg(xi) <= yi <= T_pos(xi):
                wedge_pts.append((xi,yi))
        wedge_pts = np.array(wedge_pts).astype(int)
        return wedge_pts

    def linear_fun(self,x0,y0,dx,dy,pos=True):
        if dy < self.eps:
            return lambda x: y0
        if dx <  self.eps:
            if pos:
                return lambda x: np.inf
            else:
                return lambda x: -np.inf
        else:
            return lambda x: dy*(x-x0)/dx + y0
        

class BoundaryStandardizer:
    def __init__(self,origin_key="01",centroid=(0,0,0)):
        self.origin_key = origin_key
        self.centroid = centroid
        self.lambda0 = 0
        self.set_lambda0()

    def set_lambda0(self):
        if self.origin_key == "10":
            self.lambda0 = 0
        elif self.origin_key == "01":
            self.lambda0 = np.pi/2
        elif self.origin_key == "-10":
            self.lambda0 = np.pi
        elif self.origin_key == "0-1":
            self.lambda0 = 3*np.pi/2
        else:
            logger.error('Origin key not recognized: leaving lambda0 as 0')

    def standardize_xy(self,x,y):
        x,y = self.unique_points(x,y)
        x,y = self.make_ccw(x,y)
        mux,muy = self.centroid[0],self.centroid[1] 
        x_ = np.abs(x-mux)
        y_ = y-muy
        theta = np.arctan2(y_,x_)

        x = np.roll(x,-id0)
        y = np.roll(y,-id0)
        return x,y
    
    def make_ccw(self,x,y):
        x_ = np.array(x)-self.centroid[0]
        y_ = np.array(y)-self.centroid[1]
        theta = np.arctan2(y_,x_)
        dtheta = np.diff(theta)
        if np.mean(dtheta) < 0:
            # x,y are clockwise --> reverse order
            x = x[::-1]
            y = y[::-1]
        return x,y
    
    @staticmethod
    def unique_points(x,y):
        if len(x)==0:
            return x,y
        else:
            pts = np.array(list(zip(x,y)))
            ids = np.unique(pts,axis=0,return_index=True)[1]
            pts = np.array([pts[i] for i in sorted(ids)])
            return pts[:,0],pts[:,1]

    def xyz_to_lambdaphi(self,xyz):
        x,y,z = xyz
        r = np.sqrt( (x-self.centroid[0])**2 + (y-self.centroid[1])**2 )
        z_ = z - self.centroid[2]

        phi = np.arctan2(z_,r)
        lam = np.arctan2(y-self.centroid[1],x-self.centroid[0]) - self.lambda0
        
        return phi,lam


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create a chart of a volume')
    parser.add_argument('data_intensity', type=str, help='Path to the intensity data')
    parser.add_argument('data_segmented', type=str, help='Path to the segmented data')
    parser.add_argument('--projection_type', type=str, default='hammer', help='Type of projection to use')
    parser.add_argument('--intensity_method', type=str, default='max', help='Method to calculate intensity')
    parser.add_argument('--nS', type=int, default=200, help='Number of arclength points to use for the chart')
    parser.add_argument('--z0', type=int, default=0, help='Starting slice')
    parser.add_argument('--z1', type=int, default=None, help='Ending slice')
    parser.add_argument('--dN', type=int, default=100, help='Extent of the wedge in the normal direction')
    parser.add_argument('--dT', type=int, default=20, help='Extent of the wedge in the tangent direction')
    parser.add_argument('--eps', type=float, default=1e-6, help='Tolerance for checking if a number is zero')
    parser.add_argument('--di', type=int, default=4, help='Number of points to use around the point of interest for tangent calculation')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages') 
    parser.add_argument('--do_save', action='store_true', help='Pickle the values needed to create the chart')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    data_intensity = itk.imread(args.data_intensity)
    data_segmented = itk.imread(args.data_segmented)
    v = Volume(data_intensity=data_intensity,data_segmented=data_segmented,projection_type=args.projection_type,intensity_method=args.intensity_method,nS=args.nS,
               boundary_kwargs={'di':args.di,'eps':args.eps,'dN':args.dN,'dT':args.dT})
    if args.do_save:
        vals = v.chart(z0=args.z0,z1=args.z1,return_chart_values=True)
        with open('chart_vals.pkl','wb') as f:
            pickle.dump(vals,f)
        with open('centroid.pkl','wb') as f:
            pickle.dump(v.centroid,f)
