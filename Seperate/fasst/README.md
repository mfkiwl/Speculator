This repository contains versions 2.2.2 and 1 of FASST, previously available at http://fasst.gforge.inria.fr/ and http://bass-db.gforge.inria.fr/fasst/, respectively.


# Purpose

FASST aims to **speed up the conception** and **automate the implementation** of model-based audio source separation algorithms.

Featured models:
- LGM, NMF, GMM, GSMM, HMM, HSMM (NMF is the only model available in the C++ version of the toolbox)
- source-filter models
- rank-1 and full-rank spatial models
- any combination of the models above


# Version 3.0.0 (August 2019, by request)

Language: core in C++ and user scripts in Matlab and Python

Authors: Yann Salaün, Emmanuel Vincent, Ewen Camberlein, Romain Lebarbenchon, Rémi Gribonval, and Nancy Bertin

License: For academic research activities only, this software is freely available under the terms of the following [license agreement](FASST_V3.0_academic_license_2019-08-29.pdf). To obtain the source code, please fill in the license agreement (items in blue boxes) and sign it. Send two signed copies by postal mail to:

	 Inria Rennes Bretagne Atlantique
	 Attn Ana-Bela LECONTE
	 Campus de Beaulieu
	 35042 Rennes Cedex
	 France

or scan it and email it to stip-rba@inria.fr.

For all other uses, the software is available under a commercial license. Please contact stip-rba@inria.fr.


# Earlier versions (this repository)

[Version 2.2.2](v2.2.2) (May 2018)  
Language: core in C++ and user scripts in Matlab and Python  
Authors: Yann Salaün, Emmanuel Vincent, Ewen Camberlein, Romain Lebarbenchon, and Nancy Bertin  
License: [Q Public License Version 1.0](v2.2.2/src_dir/LICENSE.txt) 

[Version 1](v1) (March 2013)  
Language: Matlab  
Authors: Alexey Ozerov, Emmanuel Vincent and Frédéric Bimbot  
License: [GNU General Public License (GPL) version 3](https://www.gnu.org/licenses/gpl-3.0.en.html).


# Usage
Version 2.2.2: see [documentation](v2.2.2/doc/index.html).

Version 1: see [user guide](v1/doc/FASST_UserGuide_v1.pdf) and [examples](v1/examples.html).


# References
[1] Y. Salaün, E. Vincent, N. Bertin, N. Souviraà-Labastie, X. Jaureguiberry, D. T. Tran, and F. Bimbot, ["The Flexible Audio Source Separation Toolbox Version 2.0"](https://hal.inria.fr/hal-00957412/document), in *Show & Tell, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2014.  
[2] A. Ozerov, E. Vincent, and F. Bimbot, ["A general flexible framework for the handling of prior information in audio source separation"](https://hal.archives-ouvertes.fr/hal-00626962v4/document), *IEEE Transactions on Audio, Speech and Signal Processing*, 20(4), pp. 1118-1133 (2012).