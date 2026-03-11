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
import cupy as cp
import matplotlib.pyplot as plt
import pylops
from phantominator import shepp_logan

import sys
sys.path.append('../')
from caSRT import *

def test_adj(srt):
    im_shape = srt.im_shape
    data_shape = srt.data_shape

    F_np = shepp_logan(im_shape)
    print("Image size: {}".format(F_np.shape))
    F = cp.array(F_np, dtype=np.float32)

    #Forward computation
    measurements = srt.fwd(F)
    measurements2 = cp.random.randn(*data_shape, dtype=np.float32)

    print("Measurements size: {}".format(measurements.shape))
    print("     Number of angles: {}".format(measurements.shape[0]))
    print("     Number of heights: {}".format(measurements.shape[1]))
    print("     Number of radii: {}".format(measurements.shape[2]))
   

    #Adjoint computation
    adj = srt.bwd(measurements)
    adj2 = srt.bwd(measurements2)

    print("Adjoint size: {}".format(adj.shape))

    #Inner product test:
    m2_AF = cp.sum(measurements2*measurements)
    adj2_F = cp.sum(adj2*F)

    print("Adjoint test: ", 2*(m2_AF - adj2_F)/(np.abs(m2_AF) + np.abs(adj2_F)))

def test_recon(srt):
    F_np = shepp_logan(im_shape)
    F = cp.array(F_np, dtype = np.float32)
    Y = srt.fwd(F)

    sol = pylops.optimization.basic.lsqr(srt, Y.flatten(), cp.zeros_like(F.flatten()), damp=1e-4, niter = 10000, show=True)

    f_hat = sol[0]
    F_hat = cp.reshape(f_hat, srt.im_shape)
    F_hat_np = cp.asnumpy(F_hat)

    plt.subplot(2,3,1)
    plt.imshow(F_np[64,:,:])
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.imshow(F_hat_np[64,:,:])
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.imshow(np.abs(F_np[64,:,:] -F_hat_np[64,:,:] ))
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.imshow(F_np[:,128,:])
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.imshow(F_hat_np[:,128,:])
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.imshow(np.abs(F_np[:,128,:] -F_hat_np[:,128,:] ))
    plt.colorbar()
    plt.savefig("3d_slice.png")

    print( cp.linalg.norm(F.flatten()-sol[0])/cp.linalg.norm(F.flatten()))









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
    R = 0.75
    Na = 360
    angles = np.linspace(0, 2*np.pi, Na, endpoint=False)
    Nh = 2*im_shape[0]
    heights = np.linspace(-H, H, Nh)

    dr = Lx/(2*im_shape[2])


    Ayx = CircularRadonTransform(im_shape[2], Lx, R, angles = angles, dr = dr)
    Azr = CircularRadonTransform_ZR(im_shape[0], H, Ayx.numCircles,Ayx.min_radius,
                                    Ayx.max_radius, heights=heights)
    data_shape = [Na, Nh, Azr.numCircles]

    srt = SphericalRadonTransformGPU(im_shape, data_shape, Ayx.tocuda(), Azr.tocuda())

    test_adj(srt)
    test_recon(srt)

