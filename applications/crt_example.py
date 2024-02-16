# Copyright (c) 2022, Washington University in St. Louis.
#
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
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom

import sys
sys.path.append('../')
from caSRT import *

if __name__ == "__main__":
    """
    Example implementation of the cylindrical Radon transform (CRT)
    for a circular aperture.
    
    """


    #Number of voxels in the x and y direction
    Nx = 400

    #Angles in the x,y direction for an assumed 
    Na = 180
    angles = 2*np.pi*np.arange(Na)/Na
    #Heights at which SRT meaurements are computed

    F = shepp_logan_phantom()
    print("Image size: {}".format(F.shape))

    numCircles = int(np.sqrt(2)*Nx)

    crt = CircularRadonTransform(Nx, angles = angles, numCircles= numCircles)

    #Forward computation
    measurements = crt.fwd(F.flatten()).reshape((Na,numCircles))

    print("Measurements size: {}".format(measurements.shape))
    print("     Number of radii: {}".format(numCircles))
    print("     Number of angles: {}".format(Na))

    #Adjoint computation
    adj = crt.bwd(measurements.flatten()).reshape((Nx,Nx))

    print("Adjoint size: {}".format(adj.shape))



