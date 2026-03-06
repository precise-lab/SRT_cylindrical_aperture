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
    im_shape = [128, 256, 256]

    # FOV size
    Lx = Ly = 1.
    H = .5

    #Imager
    R = 1.
    Na = 360
    angles = np.linspace(0, 2*np.pi, Na, endpoint=False)
    Nh = im_shape[0]
    heights = np.linspace(-H, H, Nh)

    dr = Lx/(3*im_shape[2])
    
    F = shepp_logan(im_shape)
    print("Image size: {}".format(F.shape))

    Ayx = CircularRadonTransform(im_shape[2], Lx, R, angles = angles, dr = dr)
    Azr = CircularRadonTransform_ZR(im_shape[0], H, Ayx.numCircles,Ayx.min_radius,
                                    Ayx.max_radius, heights=heights)
    data_shape = [Na, Nh, Azr.numCircles]

    srt = SphericalRadonTransform(im_shape, data_shape, Ayx.A, Azr.A)

    #Forward computation
    measurements = srt.fwd(F)
    measurements2 = np.random.randn(*data_shape)

    print("Measurements size: {}".format(measurements.shape))
    print("     Number of angles: {}".format(measurements.shape[0]))
    print("     Number of heights: {}".format(measurements.shape[1]))
    print("     Number of radii: {}".format(measurements.shape[2]))
   

    #Adjoint computation
    adj = srt.bwd(measurements)
    adj2 = srt.bwd(measurements2)

    print("Adjoint size: {}".format(adj.shape))

    #Inner product test:
    m2_AF = np.sum(measurements2*measurements)
    adj2_F = np.sum(adj2*F)

    print("Adjoint test: ", 2*(m2_AF - adj2_F)/(np.abs(m2_AF) + np.abs(adj2_F)))

