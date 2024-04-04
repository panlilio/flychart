import numpy as np
import scipy as sp

def xy_to_arclength(x,y):
    arclength = np.zeros(len(x))
    for i in range(1,len(x)):
        arclength[i] = arclength[i-1] + np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
    return arclength

def normal_fun(x,y,eps=1e-6):
    if len(x) != len(y):
        raise ValueError('x and y must have the same length')

    midpt = int(len(x)/2)
    x0 = x[midpt]
    y0 = y[midpt]
    
    if len(x)==2:
        dx = x[1]-x[0]
        dy = y[1]-y[0]
    else:
        # Use spline interpolation to calculate tangent
        x = np.array(x)
        y = np.array(y)
        arclength = xy_to_arclength(x,y)
        spl = sp.interpolate.SmoothBivariateSpline(x,y,arclength)
        dx,dy = spl(x0,y0,dx=1,dy=1)

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
    
    def g(dr,dtheta,ndr=10,ndtheta=10):
        R_ = np.linspace(r_-dr/2,r_+dr/2,ndr)
        Th_ = np.linspace(theta_-dtheta/2,theta_+dtheta/2,ndtheta)
        X_ = R_[:,None]*np.cos(Th_)[None,:]
        Y_ = R_[:,None]*np.sin(Th_)[None,:]
        X = X_ + x_
        Y = Y_ + y_
        return X,Y

    return g

        
     
