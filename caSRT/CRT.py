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
import scipy as sc
import scipy.sparse as scs

class CircularRadonTransform:

    """
    %CircularRadonTransform  Creates a 2D spherical Radon tomography test problem
    %
    % This function genetates a tomography test problem based on the circular
    % Radon tranform where data consists of integrals along circles.  This type
    % of problem arises, e.g., in photoacoustic imaging.
    %
    % The image domain is a square centered at the origin with length L.  The centers for the
    % integration circles are placed on a circle of radius R
    % For each circle center we integrate along a number of concentric circles
    % with equidistant radii, using the periodic trapezoidal rule.
    %
    % Assumes isotropic pixel size.
    %
    % Input:
    %   N           Scalar denoting the number of pixels in each dimesion, such
    %               that the image domain consists of N^2 cells.
    %   L           Dimension of FOV
    %   R           Radius of detection geometry. Default = sqrt(2)/2*L
    %   angles      Vector containing the angles to the circle centers in
    %               radians. Default: angles = 2*pi*(0:2:358)/360.
    %   numCircles  Number of concentric integration circles for each center.
    %               Default: numCircles = round(sqrt(2)*N).
    % Output:
    %   A           If input isMatrix is True (default): coefficient matrix with
    %               N^2 columns and length(angles)*numCircles rows.
    %               The matrix assumes that the image is vectorized with the fastest
    %               running index corresponding to the x-axis.
    %               
    %
    %
    % Based on Matlab code written: Per Christian Hansen, Jakob Sauer Jorgensen, and 
    % Maria Saxild-Hansen, 2010-2017 & Juergen Frikel, OTH Regensburg.
    """
    def __init__(self,N,L,R,angles=None,numCircles=None):
        self.N = N
        self.L = L
        self.R = R
        # Default value of the angles to the circle centers.
        if angles is None:
            self.angles = np.arange(0, 359, 2)*np.pi/180.
        else:
            self.angles = angles
            
        if numCircles is None:
            self.numCircles = int( np.round(np.sqrt(3.)*N) )
        else:
            self.numCircles = int( numCircles )
            
        self.A = self._get_system_matrix(self.N, L, R, self.angles, self.numCircles)

            
        self.shape = [self.numCircles*self.angles.shape[0], self.N*self.N]
            
    def fwd(self, x):
            return self.A*(x)

        
    def bwd(self, y):
            return self.A.T*y

            
    def _get_system_matrix(self, N, L, R, angles, numCircles):
        # Define the number of angles.
        nA = angles.shape[0]
        
        # Radii for the circles.
        min_radius = R - 0.5*np.sqrt(2)*L
        max_radius = R + 0.5*np.sqrt(2)*L
        radii  = np.linspace(min_radius, max_radius,numCircles+1)
        radii  = radii[1:]

        # Image coordinates.
        centerImg = np.floor(N/2)

        # Determine the quarature parameters.
        dx = L/N
        nPhi = np.ceil((4.*np.pi/dx)*radii)
        dPhi = 2*np.pi/nPhi
        
        II = np.arange(nA)
        JJ = np.arange(numCircles)


        # Initialize vectors that contains the row numbers, the column numbers
        # and the values for creating the matrix A effiecently.
        nnz = int(2*N*nA*numCircles)
        rows = np.zeros( nnz, dtype=np.int32)
        cols = np.zeros( nnz, dtype=np.int32)
        vals = np.zeros( nnz, dtype=np.float64)
            
        idxend = 0

        # Loop over angles.
        for m in II: 
            # Angular position of source.
            xix = R*np.cos(angles[m])
            xiy = R*np.sin(angles[m])
    
            # Loop over the circles.
            for n in JJ:
                # (x,y) coordinates of circle.
                k = np.arange(nPhi[n])*dPhi[n]
                xx = (xix + radii[n]*np.cos(k))
                yy = (xiy + radii[n]*np.sin(k))
        
                # Round to get pixel index.
                pixel_x_index = np.round( xx/dx )+centerImg-1
                pixel_y_index = np.round( yy/dx )+centerImg-1
        
                # Discard if outside box domain.
                IInew = np.logical_and(pixel_x_index>=0, pixel_x_index<N) & np.logical_and(pixel_y_index>=0, pixel_y_index<N)
                pixel_x_index = pixel_x_index[IInew]
                pixel_y_index = pixel_y_index[IInew]
                J = pixel_x_index + pixel_y_index*N     # Ordering in contiguous wrt x
        
                # Convert to linear index and bin
                Ju, w = np.unique(J, return_counts=True)
        
                # Determine rows, columns and weights.
                i = m*numCircles + n
                ii = np.array([i]*Ju.shape[0])
                jj = Ju
                aa = (2.*np.pi*radii[n]/nPhi[n])*w
        
                # Store the values, if any.
                if jj.shape[0] > 0:
                    # Create the indices to store the values to vector for
                    # later creation of A matrix.
                    idxstart = idxend
                    idxend = idxstart + jj.shape[0]
                    idx = np.arange(idxstart, idxend)
                
                    # Store row numbers, column numbers and values.
                    rows[idx] = ii
                    cols[idx] = jj
                    vals[idx] = aa

                            
        # Truncate excess zeros.
        rows = rows[:idxend]
        cols = cols[:idxend]
        vals = vals[:idxend]
    
        # Create sparse matrix A from the stored values.
        A = scs.csr_matrix((vals, (rows,cols)), (self.numCircles*nA,self.N*self.N) )
        return A


class CircularRadonTransform_ZR:

    """
    %CircularRadonTransform_ZR  Helper circular radon test problem
    %
    % This function genetates a tomography test problem based on the circular
    % Radon transform where data consists of integrals along circles.  This type
    % of problem arises, e.g., in photoacoustic imaging.
    %
    % The image domain is a rectangle with z in [-0.5H, .5H] and r in [min_radius max_radius].
    % The centers for the integration circles are placed at r = 0 and z = heights
    % For each circle center we integrate along a number of concentric circles
    % with equidistant radii, using the periodic trapezoidal rule.
    %
    % Assumes isotropic pixel size.
    %
    %
    % Input:
    %   Nz           Scalar denoting the number of pixels in the z-dimension
    %   Nr          Scalar denoting the number of pixels in the radial 
    %               directions such that the image domain consists of Nz*Nr
    %   heights     Vector containing the normalized column heights 
    %   numCircles  Number of concentric integration circles for each center.
    %               Default: numCircles = Nr.
    %
    %
    % Based on Matlab code written: Per Christian Hansen, Jakob Sauer Jorgensen, and 
    % Maria Saxild-Hansen, 2010-2017 & Juergen Frikel, OTH Regensburg.
    """
    def __init__(self, Nz, H, Nr, min_radius, max_radius, heights=None, numCircles=None):
        self.Nz = Nz
        self.H=H
        self.Nr = Nr
        self.min_radius  = min_radius
        self.max_radius  = max_radius
        # Default value of the angles to the circle centers.
        if heights is None:
            self.heights = np.linspace(-0.5*H, 0.5*H, Nz)
        else:
            self.heights = heights
            
        if numCircles is None:
            self.numCircles = np.ceil( np.sqrt(Nr*Nr+Nz*Nz) )
        else:
            self.numCircles = int( numCircles )

        self.A = self._get_system_matrix(self.Nz, self.H, self.Nr, self.min_radius, self.max_radius,
                                         self.heights, self.numCircles)

            
        self.shape = [self.numCircles*len(self.heights), self.Nz*self.Nr]
            
    def fwd(self, x):
            return self.A*(x)

        
    def bwd(self, y):
            return self.A.T*y

            
    def _get_system_matrix(self, Nz, H, Nr, min_radius, max_radius, heights, numCircles):
        # Define the number of angles.
        nH = len(heights)
        h_max = np.max(heights)
        
        # Radii for the circles.
        radii = np.linspace(min_radius,np.sqrt(max_radius**2+0.25*(H+h_max)**2),numCircles + 1)
        radii  = radii[1:]

        # Image coordinates.

        # Determine the quarature parameters.
        dz = H/Nz
        dr = (max_radius-min_radius)/Nr
        dzr_sqrt = np.sqrt(dz*dr)
        nPhi = np.ceil((4.*np.pi/dzr_sqrt)*radii)
        dPhi = 2*np.pi/nPhi
        
        II = np.arange(nH)
        JJ = np.arange(numCircles)

        # Initialize vectors that contains the row numbers, the column numbers
        # and the values for creating the matrix A effiecently.
        nnz = int(np.sqrt(Nz*Nr)*nH*numCircles)
        rows = np.zeros( nnz, dtype=np.int32)
        cols = np.zeros( nnz, dtype=np.int32)
        vals = np.zeros( nnz, dtype=np.float64)
            
        idxend = 0

    
        # Loop over angles.
        for m in II: 
            # Angular position of source.
            ziz = heights[m]
    
            # Loop over the circles.
            for n in JJ:
                # (x,y) coordinates of circle.
                k = np.arange(nPhi[n])*dPhi[n]
                zz =  Nz/2 + (ziz + radii[n]*np.cos(k))/dz
                rr = (radii[n]*np.sin(k) - min_radius)/dr# + centerImg
        
                # Round to get pixel index.
                z_index = np.round( zz )-1
                r_index = np.round( rr )-1
        
                # Discard if outside box domain.
                IInew = np.logical_and(z_index>=0, z_index<Nz) & np.logical_and(r_index>=0, r_index<Nr)
                r_index = r_index[IInew]
                z_index = z_index[IInew]
                J = r_index + z_index*Nr        #r_index is the fastest running index
        
                # Convert to linear index and bin
                Ju, w = np.unique(J, return_counts=True)
        
                # Determine rows, columns and weights.
                i = m*numCircles + n
                ii = np.array([i]*Ju.shape[0])
                jj = Ju
                aa = (2.*np.pi*radii[n]/nPhi[n])*w
        
                # Store the values, if any.
                if jj.shape[0] > 0:
                    # Create the indices to store the values to vector for
                    # later creation of A matrix.
                    idxstart = idxend
                    idxend = idxstart + jj.shape[0]
                    idx = np.arange(idxstart, idxend)
                
                    # Store row numbers, column numbers and values.
                    rows[idx] = ii
                    cols[idx] = jj
                    vals[idx] = aa

        # Truncate excess zeros.
        rows = rows[:idxend]
        cols = cols[:idxend]
        vals = vals[:idxend]
    
        # Create sparse matrix A from the stored values.
        A = scs.csr_matrix((vals, (rows,cols)), (self.numCircles*nH,self.Nz*self.Nr) )
        return A

