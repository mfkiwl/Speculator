Build on Linux {#build-linux}
===
[TOC]
# Install FASST dependencies

**Install libsndfile**

    sudo apt-get install libsndfile1-dev 

**Install Python**

    sudo apt-get install python-dev
    sudo pip2 install scipy  

**Build and install Cmake (recommended 3.9.5 version)**  
    
    mkdir ./cmake_temp
    cd ./cmake_temp
    wget https://cmake.org/files/v3.9/cmake-3.9.5.tar.gz
    tar -xzvf cmake-3.9.5.tar.gz
    cd cmake-3.9.5/
    ./bootstrap
    make -j4
    sudo make install
    cd ../..
    rm -Rf cmake_temp

**Build and install Eigen (recommended 3.3.4 version)**
    
    wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
    tar xjf 3.3.4.tar.bz2
    cd eigen-eigen-*
    mkdir build_dir
    cd build_dir
    cmake ..
    sudo make install

# Install FASST example dependencies (optional)

**PEASS v2.0.1**

Source separation measurements backend (in Matlab) used in exemple 2.

Detection of PEASS on your computer is automatically done with cmake but PEASS has to be previously installed.
In the following steps, we assume that PEASS will be installed in the home (~), but you can replace it by your own  directory
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
    Edit your .bashrc file and add following line to export PEASS path to the environment variable "PATH":
        export PATH=$PATH:~/peass/PEASS-Software-v2.0.1
    Launch a new terminal to take into account the new environment variable "PATH"   
    
# Build FASST (FASST directory)

Build FASST project in Release mode:

    mkdir build_dir
    cd build_dir/
    cmake ../src_dir
    make
    
Optional flags can be activated to build unit test projects and select the build type (Debug/Release): 
    
    mkdir build_dir
    cd build_dir/
    cmake ../src_dir -DTEST=1 -DCMAKE_BUILD_TYPE="Release"
    make
    
# Run test

If the optional flag to build the unit test projects has been activated at the previous step, then execute in the build_dir directory:

    ctest

