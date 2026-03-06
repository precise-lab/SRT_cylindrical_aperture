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
    Nx = 256
    L = 1
    R = 2 #.5*np.sqrt(2)*L

    #Angles in the x,y direction for an assumed 
    Na = 360
    angles = 2*np.pi*np.arange(Na)/Na
    #Heights at which SRT meaurements are computed

    F = skitr.resize(shepp_logan_phantom(), [Nx,Nx])
    f = F.flatten()
    print("Image size: {}".format(F.shape))

    dr = L/(np.sqrt(2.)*Nx)

    crt = CircularRadonTransform(Nx, L, R, angles = angles, dr = dr)
    numCircles = crt.numCircles

    #Forward computation
    measurements = crt.fwd(F.flatten()).reshape((Na,numCircles))

    print("Measurements size: {}".format(measurements.shape))
    print("     Number of radii: {}".format(numCircles))
    print("     Number of angles: {}".format(Na))

    #Adjoint computation
    adj = crt.bwd(measurements.flatten()).reshape((Nx,Nx))

    print("Adjoint size: {}".format(adj.shape))

    sol = ssla.lsqr(crt.A, measurements.flatten(), damp=1e-4, show=True)

    print( np.linalg.norm(f-sol[0])/np.linalg.norm(f))

    hatF=sol[0].reshape(Nx,Nx)

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
   

