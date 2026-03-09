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
import cupyx.scipy.sparse.linalg as cpla

from .CRT import *

class SphericalRadonTransform(cpla.LinearOperator):
    """
    %SphericalRadonTransform  Creates a 3D spherical Radon tomography (SRT) test problem
    %
    % This function genetates a tomography test problem based on the spherical
    % Radon tranform where data consists of integrals along spherical shells.  This type
    % of problem arises, e.g., in photoacoustic imaging.
    %
    % The image domain is a cube centered at the origin.  The centers for the
    % integration circles are placed on a cylindrical measurement aperture just
    % outside the image domain.

    % For each circle center we integrate along a number of concentric circles
    % with equidistant radii, using the periodic trapezoidal rule. 
    %
    % Assumes isotropic voxel shape.
    % 
    % This implementation leverages the fact that for a columnated measurement
    % aperture, the SRT can be computed as the composition of two circular Radon
    % transforms as proposed by Haltmeier and Moon
    %
    %   M. Haltmeier, S. Moon, "The spherical Radon transform with centers on
    %   cylindrical surfaces", Journal of Mathematical Analysis and Applications
    %   448.1 (2017): 567-579.
    %
    % Input:
    %       im_shape: Shape of the image Nz x Ny x Nx. We assume Ny = Nx
    %.      data_shape: Shape of the data Nviews x Nheights x Ncircles
    %       Axy: Circular Radon transform matrix in the xy plane. Number of rows:
    %            Nviews*Ncircles; number of columns Nx*Ny
    %       Azr: Circular Radon transform matrix in the zr plane. Number of rows: 
    %            Nheights*Ncircles; Number of columns Ny*Ncircles     
    %
    % Output:
    %   A         The matrix is never formed explicitly, thus saving memory.
    %
    %
    % Based on Matlab code written: Per Christian Hansen, Jakob Sauer Jorgensen, and 
    % Maria Saxild-Hansen, 2010-2017 & Juergen Frikel, OTH Regensburg.
    """
    def __init__(self, im_shape, data_shape, Ayx, Azr):
        self.im_shape = tuple(im_shape)
        self.data_shape = tuple(data_shape)
        self.Ayx = cpla.aslinearoperator(Ayx)
        self.Azr = cpla.linalg.aslinearoperator(Azr)

        self.shape = tuple([np.prod(self.data_shape), np.prod(self.im_shape)])
        self.dtype = self.Ayx.dtype

    def _matvec(self,x):
        return self.fwd(x.reshape(self.im_shape)).flatten()
    
    def _rmatvec(self,y):
        return self.bwd(y.reshape(self.data_shape)).flatten()
    
    def _adjoint(self,y):
        return self.bwd(y.reshape(self.data_shape)).flatten()

    def fwd(self,x):
        assert( x.shape == self.im_shape)
        
        Na = self.data_shape[0]
        Nh = self.data_shape[1]
        numCircles = self.data_shape[2]
        Nr = self.Ayx.shape[0]//Na
        assert( Nr*Na == self.Ayx.shape[0])

        Nx = self.im_shape[2]
        Ny = self.im_shape[1]
        Nz = self.im_shape[0]
        
        b = cp.transpose( x.reshape(Nz,Nx*Ny), [1,0])
        b = self.Ayx.matmat(b) #NaNr times NZ
        b = b.reshape((Na,Nr, Nz)) #Na times Nr times Nz
        b = cp.transpose(b, axes = [2, 1, 0]) #Nz times Nr times Na
        b = b.reshape((Nz*Nr, Na)) #NzNr times Na

        bzr = self.Azr.matmat(b) #NhNr times Na
        
        return cp.transpose( bzr.reshape((Nh, numCircles, Na)), [2,0,1])
    
    def bwd(self,y):
        assert (y.shape==self.data_shape) #Nviews x Nheights x Ncircles
        Na = self.data_shape[0]
        Nh = self.data_shape[1]
        numCircles = self.data_shape[2]
        Nr = self.Ayx.shape[0]//Na
        assert( Nr*Na == self.Ayx.shape[0])

        Nx = self.im_shape[2]
        Ny = self.im_shape[1]
        Nz = self.im_shape[0]


        x = cp.transpose( y.reshape((Na, Nh*numCircles)), [1,0]) #NheightsNcircles x Nviews
        x = self.Azr.rmatmat(x) #NzNr x Nviews
        x = x.reshape((Nz, Nr, Na))  #Nz x Nr x Nviews
        x = cp.transpose(x, axes = [2, 1, 0]) #Nviews x Ncircles x Nz

        x = x.reshape((Na*Nr,Nz))
        x = self.Ayx.rmatmat(x) #NyNx x Nz

        return cp.transpose(x, [1,0]).reshape(self.im_shape)