# Flychart: mapping 3D surfaces to 2D representations
Flychart simplifies the visualization of 3D image volumes by reparameterizing the surface to obtain a 2D representation. This is especially useful when visualizing volumetric time series. While the package was developed with the study of fluorescent molecule patterning during fly embryo development, it can be applied to any 3D volume provided that the following criteria are met: (i) the features of interest occur near the object's surface, and (ii) cross-sections along at least one axis of the object yield a single, continguous area [see details [here](#markdown-header-charts.py)].

## Core dependencies
All available using `pip install`:
- numpy
- scipy
- itk
- matplotlib
- h5py
- hdf5plugin

## General usage
Flychart has 3 primary scripts to perform 

### charts.py
