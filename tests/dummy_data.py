import imageio as io
from skimage.morphology import ball, binary_opening, binary_closing
import numpy as np
import argparse

def make_dummy_volume(a=256,b=128,c=512,nu=1):
    # Create a dummy ellipsoid volume with x,y,z semi-axes a,b,c
    V = np.zeros((2*c+1,2*(b+nu),2*(a+nu)),dtype=np.uint8)
    for k in range(2*c+1):
        z = k-c
        Q = np.zeros((2*(b+nu),2*(a+nu)),dtype=np.uint8)
        for i,j,idir,jdir in zip([0,0,1,1],[0,1,0,1],[-1,-1,1,1],[-1,1,-1,1]):
            q = np.zeros((b+nu,a+nu))
            x_max = a*np.sqrt(max(0,1-(z/c)**2))
            x_max = int(x_max+nu*np.random.randn())
            for x in range(x_max):
                y = b*np.sqrt(max(0,1-(x/a)**2-(z/c)**2))
                if x > 0:
                    y = int(y+nu*np.random.randn())
                else:
                    y = int(y)
                y = max(0,y)
                q[:y,:x] = 255
            q = q[::jdir,::idir]
            Q[j*(b+nu):(j+1)*(b+nu),i*(a+nu):(i+1)*(a+nu)] = q
            V[k] = Q
    return V

def pad_volume(V,padding=3,smrad=2):
    # Pad the volume with a border of size padding and smooth edges by binary opening using structuring element ball of radius smrad
    V = np.pad(V,((padding,padding),(padding,padding),(padding,padding)),'constant')
    se = ball(smrad)
    V = binary_closing(V,se)
    V = binary_opening(V,se)
    V = V.astype(np.uint8)*255
    return V

def make_intensity_volume(V,nspots=100,sig=None,nsig=3,doinvert=False):
    # Create a volume with nspots random spots with Gaussian intensity profiles with standard deviation rad
    if sig is None:
        sig = int(max(1,(min(V.shape)//nspots**(1/3))))
    z,y,x = np.meshgrid(np.arange(-nsig*sig,nsig*sig+1),np.arange(-nsig*sig,nsig*sig+1),np.arange(-nsig*sig,nsig*sig+1))
    spot = np.exp(-(x**2+y**2+z**2)/(2*sig**2))
    spot = (255*(spot-spot.min())/(spot.max()-spot.min())).astype(np.uint8)
    F = np.zeros(V.shape,dtype=np.uint8)
    for i in range(nspots):
        x = np.random.randint(0,V.shape[2])
        y = np.random.randint(0,V.shape[1])
        z = np.random.randint(0,V.shape[0])
        xm, xM = int(max(0,x-nsig*sig)), int(min(V.shape[2],x+nsig*sig+1))
        ym, yM = int(max(0,y-nsig*sig)), int(min(V.shape[1],y+nsig*sig+1))
        zm, zM = int(max(0,z-nsig*sig)), int(min(V.shape[0],z+nsig*sig+1))
        dx, dy, dz = xM-xm, yM-ym, zM-zm
        
        offset_x = 0 if xm!=0 else 2*nsig*sig+1-dx
        offset_y = 0 if ym!=0 else 2*nsig*sig+1-dy
        offset_z = 0 if zm!=0 else 2*nsig*sig+1-dz

        F[zm:zM,ym:yM,xm:xM] = np.maximum(F[zm:zM,ym:yM,xm:xM],spot[offset_z:offset_z+dz,offset_y:offset_y+dy,offset_x:offset_x+dx])
    
    if doinvert:     
        F = (255 - F) 
    
    F = (F*(V/V.max())).astype(np.uint8)

    return F

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog="Dummy data",description="Create a dummy ellipsoid volume")
    argparser.add_argument('--a',type=int,default=50,help='x semi-axis')
    argparser.add_argument('--b',type=int,default=35,help='y semi-axis')
    argparser.add_argument('--c',type=int,default=80,help='z semi-axis')
    argparser.add_argument('--nu',type=int,default=1,help='noise level')
    argparser.add_argument('--nspots',type=int,default=100,help='number of spots in intensity volume')
    argparser.add_argument('--sig',type=int,default=None,help='standard deviation of spots')
    argparser.add_argument('--nsig',type=int,default=3,help='number of standard deviations represented for each spot')
    argparser.add_argument('--padding',type=int,default=3,help='zero padding of around the volume to avoid edge effects')
    argparser.add_argument('--smrad',type=int,default=2,help='radius of structuring element for smoothing edges')
    argparser.add_argument('--doinvert',type=bool,default=True,help='invert the intensity spots')
    args = argparser.parse_args()
    V = make_dummy_volume(args.a,args.b,args.c,args.nu)
    V = pad_volume(V,args.padding,args.smrad)
    F = make_intensity_volume(V,args.nspots,args.sig,args.nsig,args.doinvert)

    io.volwrite('dummy_segmentation.tif',V)
    io.volwrite('dummy_intensity.tif',F)
