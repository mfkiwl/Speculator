Release note (v2.2.2) : updates since v2.0.0 {#release_note}
---------------------------------
[TOC]
# v2.2.2
## Bugs fixed
* Apply a pre-EM model parameters normalization in model-estimation

# v2.2.1
## Bugs fixed
* ERB wiener filtering does not depend anymore on spatial covariance matrix rank (we have the correct number of channels in the output files now)
* Update fasst_loadRx.m and fasst_writeRx.m Matlab scripts to handle Rx_en.bin temporary file introduced in v2.2.0

# v2.2.0
## Bugs fixed
* Process a pre-EM normalization step to adjust global energy to mix (based on [FASST v1](http://bass-db.gforge.inria.fr/fasst/))
* Add a controlled noise level to inverse Sigma_x in Wiener filtering (based on [FASST v1](http://bass-db.gforge.inria.fr/fasst/)) for better stability in case of ill-conditioned matrices.
* Compiler-specific C++11 activation for Mac OS
* Fix the path for Eigen libraries on Windows "Program Files" -> "Program Files (x86)"
* sndfile lib is now correctly found on Windows
* Fasst version is now set at the beginning of the file and correctly propagated for every installation process (cmake / install Visual project / and Windows installer generated using Visual)
* Doxygen documentation (visual project and output files) can now be built on Windows. 'make doc' command may not be needed anymore on Linux side. 
* Specific warnings disabled for visual project (release and debug) are now correctly set
* ADD_TEST() function is now correctly used under Windows OS.
* Load "nbin" elements is now well executed when loading xml file from /dev/src_dir/scripts/MATLAB/fasst_loadXML.m
* wavread() is replaced by audioread() in matlab scripts

## Deployment / ergonomics updates
* Create src_dir and build_dir directories (avoid commit of built files for developers / simplify installation process)
* Update windows installer to configure the path of example folder
* Add a macOS installer
* IDE projects are stored in dedicated folders to increase IDE ergonomics
* example1, example2, matlab and python scripts are no longer configured and added to installation process depending of the version of Microsoft Visual C compiler (simply detection of Microsoft Visual usage instead)
* Add documentation files to build on Linux and Mac platforms : build-linux.markdown / build-mac.markdown

## Examples updates 
* Example 1 : minor updates (variable naming, comments)
* Example 2 : replaced by an example of source separation on simulated anechoic mixture (in Matlab)
* Example 3 : new example of source separation on reverberated mixture (in Matlab and Python)

## Third party library related updates
* gtest 1.6.0 updated to 1.8.0
* Eigen versions supported up to 3.3.4
* Cmake minimum version set to 3.9.3
* [tinyXML2](http://leethomason.github.io/tinyxml2/index.html) is now used for XML read/write instead of Qt framework.

## IDE / Compiler updates
* Visual Studio 2013/2015/2017 (and associated compiler version) are now supported