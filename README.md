# SRT for Cylindrical Apertures
Implementation of the spherical Radon transform (SRT) that exploits a cylindrical measurement aperture geometry for computational efficiency.

Companion code for technical note

> L. Lozenski, R. Cam, M. Anastasio, and U. Villa, “Technical note: An efficient implementation of the spherical radon transform with cylindrical apertures,” arXiv, 2024 ([preprint](https://arxiv.org/abs/2205.05585?context=eess))

and journal article

> L. Lozenski, R. Mert Cam, M. Anastasio, U. Villa. _ProxNF: Neural Field Proximal Training for
High-Resolution 4D Dynamic Image Reconstruction_, (2024), Submitted to IEEE Transactions on Computational Imaging


Based on formulation proposed by Haltmeier and Moon in 

> M. Haltmeier, S. Moon, _The spherical Radon transform with centers on cylindrical surfaces_, Journal of Mathematical Analysis and Applications 448.1 (2017): 567-579.





An efficient implementation of the spherical radon transform with cylindrical apertures

The spherical Radon transform (SRT) is an integral transform that maps a function to its integrals over concentric spherical shells centered at specified sensor locations. It has several imaging applications, including synthetic aperture radar and photoacoustic computed tomography. However, computation of the SRT can be expensive and its efficient implementation on general purpose graphic processing units (GPGPUs) often utilized non-matched implementation of its adjoint, leading to inconsistent gradients during iterative reconstruction methods. This work details an efficient implementation of the SRT and its adjoint for the case of a cylindrical measurement aperture. Exploiting symmetry of the cylindrical geometry, the SRT can then be expressed as the composition of two circular Radon transforms (CRT).  Utilizing this formulation then allows for an efficient implementation of the SRT as a discrete-to-discrete operator utilizing sparse matrix representation.

In this repository we provide our implementaiton of the SRT with an example application.


# Dependencies 

`scikit-image`: collection of algorithms for image processing
```bash
conda install scikit-image
```

`phantominator`: package for easy generation of numerical phantoms
```bash
pip install phantominator
```
