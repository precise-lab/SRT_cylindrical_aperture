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
import scipy as sc
import scipy.sparse as scs

from .CRT import *

class SphericalRadonTransform:
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
    %   N           Scalar denoting the number of pixels in the x and y dimesion, such
    %               that the each axial slice consists of N^2 cells.
    %   Nz          Scalar denoting the number of pixels in the z dimesion, such
    %               that the image domain consists of N^2*Nz cells.
    %   angles      Vector containing the angles to the circle centers in the 
    %               (x,y)-plane in radians. Default: angles = 2*pi*(0:2:358)/360.
    %   heights     Vector containing the heights in the z-plane for measurements.
    %               Default: heights = 2*pi*(0:2:358)/360.
    %   numCircles  Number of concentric integration circles for each center.
    %               Default: numCircles = round(sqrt(2)*N).
    %   asMatrix    If True, a sparse matrix is returned in A (default).
    %               If False, instead a function handle is returned.
    %
    % Output:
    %   A           If input isMatrix is True (default): coefficient matrix with
    %               N^2*Nz columns and length(angles)*len(heights)*numCircles rows.
    %               If input isMatrix is False: A function handle representing a
    %               matrix-free version of A in which the forward and backward
    %               operations are computed as A.fwd(x) and A.bwd(y),
    %               respectively, for column vectors x and y of appropriate size.
    %               The matrix is never formed explicitly, thus saving memory.
    %
    %
    % Based on Matlab code written: Per Christian Hansen, Jakob Sauer Jorgensen, and 
    % Maria Saxild-Hansen, 2010-2017 & Juergen Frikel, OTH Regensburg.
    """
    def __init__(self, N, Nz,angles=None,heights = None,numCircles=None):
        self.N = N
        self.Nz = Nz
        if angles is None:
            self.angles = np.arange(0, 359, 2)*np.pi/180.
        else:
            self.angles = angles
        if numCircles is None:
            self.numCircles = int( np.round(np.sqrt(3.)*N) )
        else:
            self.numCircles = int( numCircles )
        if heights is None:
            self.heights = np.arange(N)[::10]
        else:
            self.heights = heights

        self.Axy = CircularRadonTransform(N, angles = self.angles, numCircles= self.numCircles)
        self.Azr = CircularRadonTransform_ZR(Nz, N, heights = self.heights, numCircles=self.numCircles)
    def fwd(self,x):
        Na = len(self.angles)
        Nz = len(self.heights)
        
        x = x.reshape((self.N**2,self.Nz))
        bxy = self.Axy.fwd(x)
        bxy = bxy.reshape((Na,self.numCircles, self.Nz))
        bxy = np.transpose(bxy, axes = [2, 1, 0])
        bxy = bxy.reshape((self.numCircles*self.Nz, Na))

        bzr = self.Azr.fwd(bxy)

        return bzr.reshape((Nz, self.numCircles, Na))
    def bwd(self,y):
        Na = len(self.angles)
        Nz = len(self.heights)


        y = y.reshape((Nz*self.numCircles, Na))
        y = self.Azr.bwd(y)
        y = y.reshape((self.Nz, self.numCircles,Na))
        y = np.transpose(y, axes = [2, 1, 0])

        y = y.reshape((self.numCircles*Na,self.Nz))
        y = self.Axy.bwd(y)

        return y.reshape((self.N,self.N,self.Nz))