Build on MacOs {#build-macOS}
===
[TOC]
# Install Homebrew package manager

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	
# Install FASST dependencies

    brew install wget libsndfile python cmake gcc eigen
    pip2 install numpy
    pip2 install spicy
    
# Install FASST example dependencies (optional)

**PEASS v2.0.1**

Source separation measurements backend (in Matlab) used in exemple 2.

Detection of PEASS on your computer is automatically done with cmake but PEASS has to be previously installed.
In the following steps, we assume that PEASS will be installed in the home (~), but you can replace it by your own directory
To do so, please refer to the followings steps in a new terminal :

    cd ~
    mkdir peass
    cd peass
    wget http://bass-db.gforge.inria.fr/peass/PEASS-Software-v2.0.1.zip
    unzip PEASS-Software-v2.0.1.zip
    cd PEASS-Software-v2.0.1
    wget http://medi.uni-oldenburg.de/download/demo/adaption-loops/adapt_loop.zip
    unzip adapt_loop.zip
    Open Matlab and compile the MEX files by running compile.m (this is optional but leads to much faster computation).
    Edit your .bash_profile file and add following line to export PEASS to the environment variable "PATH":
        export PATH=:"~/peass/PEASS-Software-v2.0.1:$PATH"
    Launch a new terminal to take into account the new environment variable "PATH"

# Build FASST (FASST directory)

## OpenMP status and compiler choice

OpenMP is used by FASST to accelerate some parts of code (parallellization). 
LLVM-Clang compiler does not support natively OpenMP but you can activate it following the guidelines on https://llvm.org .
GCC compiler supports natively OpenMP.

Current implementation for packaging was realised with g++-7. 
Using another compiler or compiler version implies modifying the code to fullfill your specific requirements.

If you don't need packaging tools, any compiler can be used (keeping in mind OpenMP specific requirements if cpu consumption optimization is one of your concerns)

##Build steps
\note Replace the $COMPILER variable by using "clang" or "g++-7"


    mkdir build_dir
    cd build_dir
    cmake ../src_dir -DCMAKE_CXX_COMPILER=$COMPILER
    make

Optional flags can be activated to build unit test projects and select the build type (Debug/Release):
    
    mkdir build_dir
    cd build_dir
    cmake ../src_dir -DCMAKE_CXX_COMPILER=$COMPILER -DTEST=1 -DCMAKE_BUILD_TYPE="Release"
    make
    
# Run test

If the optional flag to build the unit test projects has been activated at the previous step, then execute in the build_dir directory:

    ctest
    
# Packaging
\note For g++-7 compiler only

    ./generate_osx_installer.sh