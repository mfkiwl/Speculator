Binary file format {#binfileformat}
===

Binary files are used for two purposes:

* To store mixture covariance matrices of some multichannel audio signals (Rx.bin)
* To store the average energy per frequency of some multichannel audio signals (Rx_en.bin)

For each purpose, one format is adopted and described below.

# Rx.bin

We need to store one complex square matrix per frequency bin \f$F\f$ per time frame \f$N\f$. The dimension of each matrix is the number of audio channels \f$I\f$. Given the fact that each matrix is Hermitian (\f$M_{i,i}\in \mathbb{R}\f$ and \f$M_{i,j} = \overline{M_{j,i}}\f$), we designed the file format so that we store each matrix as a real vector of size \f$I \times I\f$.

## Header

In the header, the data is stored as integers, which size is equal to 4 bytes on most computers.

The first integer is the number of dimensions, which is always 3 in our case (\f$I \times I\f$, \f$F\f$ and \f$N\f$).

Then we store each dimension as an integer.

## Data

The data is stored as floats, which size is also equal to 4 bytes.

For each time frame and for each frequency bin, we store \f$I \times I\f$ floats. The \f$I\f$ first floats are the real diagonal elements.

Then we store the upper triangular part of the matrix. The real and imaginary parts of the complex elements are always stored side by side: the real part of \f$M_{i,j}\f$ is always followed by the imaginary part. These elements are stored in the following order: we store the first row (\f$M_{1,2}\f$, \f$M_{1,3}\f$ until \f$M_{1,I}\f$) then the second row (\f$M_{2,3}\f$, \f$M_{2,4}\f$ until \f$M_{2,I}\f$) and so on until the last element \f$M_{I-1,I}\f$.

# Rx_en.bin

We need to store one double value per frequency bin \f$F\f$.

## Header

In the header, the data is stored as integers, which size is equal to 4 bytes on most computers.

The first integer is the number of dimensions, which is always 1 in our case (\f$F\f$).

## Data

Data are stored as double values, which size also equal to 8 bytes.

For each frequency bin, we store one double value.