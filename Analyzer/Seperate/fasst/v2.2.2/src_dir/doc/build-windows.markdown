Build on Windows {#build-windows}
===
[TOC]
# Install FAAST mandatory dependencies

**CMake (version 3.10.0)**
1. Visit [http://www.cmake.org/](http://cmake.org/)
2. Download the Win32 installer
3. Run the installer

**libsndfile (version 1.0.28-w64)**
1. Visit [http://www.mega-nerd.com/libsndfile/](http://www.mega-nerd.com/libsndfile/)
2. Download the Windows installer that corresponds to your hardware
3. Run the installer

**Eigen (version 3.3.4)**
1. Visit [http://eigen.tuxfamily.org/](http://eigen.tuxfamily.org/)
2. Download the 3.3.4 release as zip file and unzip it.
3. Create a build_dir in Eigen directory, then cd to this directory:
    > mkdir build_dir
    > cd build_dir
3. Run CMake in the Eigen build directory:
    > cmake ..
4. Open a Visual Studio command prompt as an admin, cd to Eigen build directory and enter the following command: `msbuild INSTALL.vcxproj`

**Python 2.7 + numpy + scipy**

In order to run python scripts, you should install a python environment
1. Download and install python 2.7.13 (https://www.python.org/downloads/release/python-2713/)
2. In a windows command prompt, execute the following command to install numpy and scipy packages:

    pip install numpy
    pip install scipy
 
# Install other tools (optional) {#tools}
  
**NSIS (version 2.46 for windows up to windows 7 - windows 8 and 8.1 require v3xx):** 

In order to build windows installer 
1. Visit http://nsis.sourceforge.net/Download
2. Run the installer

**Doxygen (version 1.8.13):**

In order to extract code documentation 
1. Visit http://www.stack.nl/~dimitri/doxygen/download.html
2. Run the installer

**GraphViz (version 2.38)**:

Used by doxygen for graphs
1. Visit http://www.graphviz.org/Download_windows.php
2. Run .msi installer
3. Add path to dot.exe to PATH environment variable (ex: C:\Program Files (x86)\Graphviz2.38\bin).

# Install FASST example dependencies (optional)

**PEASS v2.0.1**

Source separation measurements backend (in Matlab) used in exemple 2.
1. Visit http://bass-db.gforge.inria.fr/peass/PEASS-Software.html
2. Download the version 2.0.1 zip file
3. Unzip the content of the file in a directory (e.g. C:\peass)
4. Download and unzip needed third-party http://medi.uni-oldenburg.de/download/demo/adaption-loops/adapt_loop.zip in PEASS folder (e.g. C:\peass)
5. Compile the MEX files by running compile.m under Matlab (this is optional but leads to much faster computation)
6. Add a new entry in your PATH environment variable pointing to PEASS directory (e.g. C:\peass).

# Build and install FASST

1. Run CMake in FASST directory (here the generator is chosen for Visual 2017 and 64 bits hardware but you can use Visual 2013 or 2015 as well):

	    mkdir build_dir
	    cd build_dir
	    cmake -G "Visual Studio 15 2017 Win64" -LA -Wno-dev ../src_dir

    Optional flag can be activated to build unit tests projects: 
    
	    cmake -G "Visual Studio 15 2017 Win64" -LA -Wno-dev ../src_dir -DTEST=ON

2. Build and installation:

 - with cmd (administrator privileges required) : 
 
	    cmake --build . --config Release --target install

 - with Visual Studio: 
    - Open the fasst.sln file in the build directory with Visual Studio and launch the build.
    - INSTALL project install FASST (administrator privileges required)
    - PACKAGE project generate a windows installer for FASST

# Run tests

 - In build_dir execute:  

		ctest -C Debug
	
	or

		ctest -C Release

    in order to run test cases.

# Packaging
\note Require you installed the [NSIS tool](\ref tools)

    cpack -G NSIS
    
# Debug mode prerequisites
Visual 2013/2015/2017 : allow Visual Studio to automatically download the correct symbols for debugging your Visual Studio project :

 Tools -> Options -> Debugging -> Symbols : check the box "Microsoft symbols server" and give a path to store the symbols
