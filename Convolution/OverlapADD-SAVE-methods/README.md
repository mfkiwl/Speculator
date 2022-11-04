# OverlapADD-SAVE-methods
## Overlap ADD method
The overlap-add method allows us to use the DFT-based method when calculating the convolution of very long sequences.

In the first part of this series, we discussed the DFT-based method to calculate the time-domain convolution of two finite-duration signals. In practice, we generally need to calculate the convolution of very long sequences. We may even need to apply a Finite Impulse Response (FIR) filter on a real-time input sequence. In these cases, it is necessary to break the input sequence into finite-duration signals of an appropriate length.

There are two methods to perform DFT-based linear filtering on long sequences: overlap-add method and overlap-save method. In this article, we will review the overlap-add method. Then, we will compare the computational complexity of an FIR filter based on the DFT method with that of the direct-form realization.
## Overlap save method
In the overlap save algorithm, the first M-1 elements of the current x_i[n] interval are 'saved' from the overlap last of the previous x_(i-1)[n] interval.  This is where the name 'overlap save' comes from.  The initial 'saved' values are simply set to zero.
The calculation step is quite similar to that found in the overlap add algorithm.  One notable difference from the overlap add method is in overlap add, the zero padding that occurs on the end of each x_i[n] interval ensures that the circular convolution is equivalent to the linear convolution.  This means that with overlap add, the matrix calculation need not actually have non-zero elements in the upper right-hand corner of the filter matrix since they will always be multiplied against zero elements.  However, in overlap save this is not the case, and circular convolution must be used.
## Why Overlap ADD/SLAVE method is employed?
Suppose, the input sequence x[n] of long duration is to be processed with a system having finite duration impulse response by convolving the two sequences. Since, the linear filtering performed via DFT involves operation on a fixed size data block, the input sequence is divided into different fixed size data block before processing.

The successive blocks are then processed one at a time and the results are combined to produce the net result.

As the convolution is performed by dividing the long input sequence into different fixed size sections, it is called sectioned convolution. A long input sequence is segmented to fixed size blocks, prior to FIR filter processing.

Two methods are used to evaluate the discrete convolution −

    Overlap-save method

    Overlap-add method
