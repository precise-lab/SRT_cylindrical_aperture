# Copyright (c) 2026, The University of Texas at Austin
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SRT for Cylindrical Apertures Library. For more information and source code
# availability see https://github.com/precise-lab/SRT_cylindrical_aperture.
#
# SRT for Cylindrical Apertures is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import numpy as np
import scipy.sparse.linalg as ssla
import matplotlib.pyplot as plt
import skimage.transform as skitr
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from caSRT import *

if __name__ == "__main__":
    """
    Example implementation of the cylindrical Radon transform (CRT)
    for a circular aperture.
    
    """


    #Number of voxels in the x and y direction
    Nz = 128
    Nr = 256

    H = 1
    min_radius = 0.1
    max_radius = 1.1

    #Angles in the x,y direction for an assumed 
    Nh = 512

    #Heights at which SRT meaurements are computed
    heights = np.linspace(-H, H, Nh)


    numCircles = int(np.sqrt(Nz*Nz + Nr*Nr))*3
    F = skitr.resize(shepp_logan_phantom(), [Nz,Nr])
    f = F.flatten()
    print("Image size: {}".format(F.shape))



    crt = CircularRadonTransform_ZR(Nz, H, Nr, min_radius, max_radius, heights=heights, numCircles= numCircles)

    #Forward computation
    measurements = crt.fwd(F.flatten()).reshape((Nh,numCircles))

    print("Measurements size: {}".format(measurements.shape))
    print("     Number of radii: {}".format(numCircles))
    print("     Number of heights: {}".format(Nh))

    #Adjoint computation
    adj = crt.bwd(measurements.flatten()).reshape((Nz,Nr))

    print("Adjoint size: {}".format(adj.shape))

    sol = ssla.lsqr(crt.A, measurements.flatten(), damp=1e-2, show=True)

    print( np.linalg.norm(f-sol[0])/np.linalg.norm(f))

    hatF=sol[0].reshape(Nz,Nr)

    plt.subplot(1,3,1)
    plt.imshow(F)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(hatF)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(np.abs(F-hatF))
    plt.colorbar()
    plt.show()
   

