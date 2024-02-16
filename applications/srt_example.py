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
from phantominator import shepp_logan

import sys
sys.path.append('../')
from caSRT import *

if __name__ == "__main__":
    """
    Example implementation of the spherical Radon transform (SRT)
    for a cylindrical aperture.
    
    """


    #Number of voxels in the x and y direction
    Nx = 256
    #Number of voxels in the z direction with isotropic pixel sizes
    Nz = 128

    #Angles in the x,y direction for an assumed 
    angles = 2*np.pi*np.arange(0,360,2)/360
    #Heights at which SRT meaurements are computed
    heights = np.linspace(-1/2,1/2,64)

    F = shepp_logan((Nx, Nx, Nz))
    print("Image size: {}".format(F.shape))

    srt = SphericalRadonTransform(Nx, Nz, angles= angles, heights = heights)

    #Forward computation
    measurements = srt.fwd(F)

    print("Measurements size: {}".format(measurements.shape))
    print("     Number of heights: {}".format(measurements.shape[0]))
    print("     Number of radii: {}".format(measurements.shape[1]))
    print("     Number of angles: {}".format(measurements.shape[2]))

    #Adjoint computation
    adj = srt.bwd(measurements)

    print("Adjoint size: {}".format(adj.shape))



