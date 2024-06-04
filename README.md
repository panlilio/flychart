# Flychart: mapping 3D surfaces to 2D representations
Flychart simplifies the visualization of 3D image volumes by reparameterizing the surface to obtain a 2D representation. This is especially useful when visualizing volumetric time series. While the package was developed with the study of fluorescent molecule patterning during fly embryo development, it can be applied to any 3D volume provided that the following criteria are met: (i) the features of interest occur near the object's surface, and (ii) cross-sections along at least one axis of the object yield a single, continguous area [see details [here](#chartspy)].

## Core dependencies
All available using `pip install`:
- numpy
- scipy
- itk
- matplotlib
- h5py
- hdf5plugin

## General usage
Flychart has 2 primary modules:
- **preprocess** : aligns object principal axes to physical xyz-axes. This step is optional 
- **charts** : processes an aligned image volume by z-slices to find a flattened surface representation, plotted as a heat map  

The class responsible for image reading and its associated engines are contained in **charts_io.py**. Currently, Flychart supports intake through ITK, or h5py for HDF5 files that follow the Bitplane Imaris hierarchical convention. The Flychart Reader was written to be extensible if other image formats are required. That said, the ITK engine is capable of reading the standard TIF(F), PNG, JPEG, and most biomedical image formats.  

The **vis.py** script maps 3D tracks from Imaris' Spot detector to the corresponding 2D representation.

### preprocess.py
The preprocessing module calculates the object's moments of inertia and transforms the image to align the principal axes with physical xyz-axes. Flychart analyzes 2D slices along z and flattens them to a 1D representation. While the use of Flychart's preprocessing module is entirely optional, it is advisable to perform some kind alignment to orient oneself in 2D. In the case of the fly embryo, which is near-ellipsoidal for much of its development, the longitudinal axis is the natural choice for z, with left/right and dorsal/ventral axes aligned to x and y (or y and x) respectively.  

The default workflow when run as `__main__`:
1. Read in the image volume and, if available, its mask image
2. If the mask image does not exist, create it by segmenting the image using thresholding + binary morphological operators  
3. Calculate the moments of inertia and the corresponding transformation from principal to physical axes
4. Resample the image and its mask using the principal axes transformation
5. Write the image and its mask to file  

#### Command-line usage
$ python preprocess.py <path-to-my-image-volume> [--mask <path-to-my-mask-image>]

### charts.py

