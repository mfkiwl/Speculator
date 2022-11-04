FASST Documentation {#mainpage}
===
[TOC]
Welcome to the home page for the C++ implementation of the FASST toolbox. 
This page is a quickstart guide for users. Other pages of this documentation describe the internal structure of FASST and the build process.

Have a look to the related [release note](\ref release_note) for change logs.

# Download installers (Windows or MacOs) {#download}
* [Windows installer (64-bit)](files/FASST-2.2.2-win64.exe): for users only (binaries + example scripts)
* [MAC OS-X installer (64-bit)](files/FASST_2.2.2-OSX.pkg): for users only (binaries + example scripts)

MacOS and windows installers are designed to install FASST binaries on your computer and copy a folder containing three example scripts to call FASST. These example scripts are described in the [dedicated](\ref run-the-examples) section.

# Download source code {#download_sources}
* [Source in zip format](files/fasst-2.2.2.zip): for users and developers

# License {#license}
This software is released under the Q Public License Version 1.0. Please find a copy of the license at http://opensource.org/licenses/QPL-1.0.

# References {#reference}
If you use FASST and obtain scientific results that you publish, please acknowledge the usage of FASST by referencing the following articles:

\htmlonly
<iframe src="https://haltools.inria.fr/Public/afficheRequetePubli.php?titre_exp=A+General+Flexible+Framework+for+the+Handling+of+Prior+Information+in+Audio+Source+Separation&annee_publideb=2012&CB_auteur=oui&CB_titre=oui&CB_article=oui&langue=Anglais&tri_exp=annee_publi&tri_exp2=typdoc&tri_exp3=date_publi&ordre_aff=TA&Fen=Aff&css=../css/styles_publicationsHAL_frame.css"  frameborder="0" width="800" height="200">
</iframe>
<br>
<iframe src="https://haltools.inria.fr/Public/afficheRequetePubli.php?titre_exp=The+Flexible+Audio+Source+Separation+Toolbox+Version+2.0&CB_auteur=oui&CB_titre=oui&CB_article=oui&langue=Anglais&tri_exp=annee_publi&tri_exp2=typdoc&tri_exp3=date_publi&ordre_aff=TA&Fen=Aff&css=../css/styles_publicationsHAL_frame.css" frameborder="0" width="800" height="220">
</iframe>
\endhtmlonly

# FASST compilation {#build-fasst}
The compilation has been tested and is known to succeed on the following systems:

* Ubuntu 16.04 (gcc)
* OSX-10.9 (Clang and gcc)
* Windows 7 (Visual Studio 2017)
* Windows 10 (Visual Studio 2013, 2015, 2017)

Please find a documentation of how to build FASST on your platform at the following links:

* [Build on Windows platform](\ref build-windows)
* [Build on Linux platform](\ref build-linux)
* [Build on MacOS platform](\ref build-macOS)

# Run the examples {#run-the-examples}
The typical workflow is to use some scripts (Python or Matlab) to initialize the source model, write it to an XML file and call FASST binaries.

This FASST version provides three example scripts to show you how to use FASST in different contexts:
* Example 1 ([Python](\ref run-the-examples-in-python) and [Matlab](\ref run-the-examples-in-matlab)): FASST source separation script dealing with an instantaneous mixture of 3 tracks (piano, voice and drums).
* Example 2 ([Matlab](\ref run-the-examples-in-matlab) only): FASST source separation script dealing with a simulated 8-channel anechoic mixture of 2 speakers.
* Example 3 ([Python](\ref run-the-examples-in-python) and [Matlab](\ref run-the-examples-in-matlab)): FASST source separation script dealing with a real 8-channel reverberated mixture of 2 speakers.
* (optional) [PEASS](http://bass-db.gforge.inria.fr/peass/PEASS-Software.html) for source separation quality measuments

These example scripts are located under the example path requested during the installation step (on subfolder per example).

## In Python {#run-the-examples-in-python}
\note Examples have been tested with Python 2.7

Both examples 1 and 3 are provided in Python. In order to run them, you have to install a python 2 environment on your computer. If you already have
installed a python 2 environment, please make sure numpy and scipy packages are available. Otherwise, you can use guidelines provided
below.

On Windows we recommend that you install one of the following python environment:

* [pythonxy](http://python-xy.github.io/)
* [python2.7](https://www.python.org/downloads/release/python-2713/)

On Linux you can install Python and dependencies with the terminal as follow:

    sudo apt-get install python-numpy
    sudo apt-get install python-dev
    sudo pip2 install scipy    
    
On macOS you can install Python with the terminal as follow (with brew):

If brew package manager is not installed, install it:

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then install Python and dependencies:

    brew install python
    pip2 install numpy
    pip2 install spicy

Once your Python environment installed, execute the .py file located in the example1 or example3 folder.

## In Matlab {#run-the-examples-in-matlab}

\note Examples scripts have been tested with Matlab R2017a

In order to run example script in Matlab, just open a Matlab session, 'cd' to the example folder and 
execute the .m file.

To run the MATLAB example, simply `cd` to the example2 directory and type `example2`

## Run the third example {#run-the-third-example}

The third example script is provided in MATLAB. It illustrates how to use FASST to seperate two speakers recorded in an reverberated condition with a 8-microphones array.

Further explanations are given in example2 source code (i.e. example3.m).

To run the MATLAB example, simply `cd` to the example3 directory and type `example3`

# Write your own script {#write-your-own-script}
As a prerequisite, you need to be familiar with the FASST source model. If not, you can refer to [this paper](http://hal.inria.fr/hal-00626962/) where it is fully described.

\note You have to be aware that some features that are present in the MATLAB version of FASST are still not implemented, here is a quick list:
* GMM and HMM spectral models
* Partially-adaptive spectral parameters
* Only one additive noise model

FASST is decomposed in 3 different executables. Here are their descriptions and usages:

## Compute mixture covariance matrix

    Usage:  comp-rx input-wav-file input-xml-file output-bin-dir

## Estimate source parameters

    Usage:  model-estimation input-xml-file input-bin-dir output-xml-file

## Separate sources

    Usage:  source-estimation input-wav-file input-xml-file input-bin-dir output-wav-dir

# Contact {#contact}
In case you have any trouble and want to contact a human being, you can send an email to fasst-support@inria.fr. We will do our best to answer quickly (we like feedback too).