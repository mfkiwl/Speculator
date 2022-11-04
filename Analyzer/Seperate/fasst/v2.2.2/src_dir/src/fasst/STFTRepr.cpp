// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "STFTRepr.h"
#include "Audio.h"
#include "Sources.h"
#include <stdexcept>
#include <unsupported/Eigen/FFT>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

namespace fasst {

    STFTRepr::STFTRepr(const int nOrigSamples, const int wlen) : Abstract_TFRepr(nOrigSamples,wlen)
    {
        // Check window length
        if (m_wlen % 4 != 0 || m_wlen == 0) {
            stringstream s;
            s << "Error:\twlen is " << m_wlen << " and should be multiple of 4.\n";
            throw runtime_error(s.str());
        }

        // Init private members
        m_frames = static_cast<int>(std::ceil(static_cast<double>(m_origSamples) / m_wlen * 2));
        m_samples = (m_frames + 1) * m_wlen / 2;
        m_bins = m_wlen / 2 + 1;
    }

    ArrayMatrixXcd STFTRepr::directFraming(const Audio &x) const
    {
        int I = x.channels();

        // Define sine window
        ArrayXd win =
            Eigen::sin(ArrayXd::LinSpaced(m_wlen, 0.5, m_wlen - 0.5) / m_wlen * M_PI);

        // Zero-padding
        ArrayXXd xx = ArrayXXd::Zero(m_samples, I);
        xx.block(m_wlen / 4, 0, m_origSamples, I) = x;

        // Pre-processing for edges
        ArrayXd swin = ArrayXd::Zero(m_samples);
        for (int n = 0; n < m_frames; n++) 
        {
            swin.segment(n * m_wlen / 2, m_wlen) += (win * win);
        }

        swin = Eigen::sqrt(m_wlen * swin);
        VectorMatrixXcd X(I);
        FFT<double> fft;
        for (int i = 0; i < I; i++) 
        {
            X(i) = ArrayXXcd(m_bins, m_frames);
            for (int n = 0; n < m_frames; n++) 
            {
                // Framing
                VectorXd frame = xx.col(i).segment(n * m_wlen / 2, m_wlen) * win /
                       swin.segment(n * m_wlen / 2, m_wlen);
                // FFT
                VectorXcd fframe;
                fft.fwd(fframe, frame);
                X(i).col(n) = fframe.segment(0, m_bins);
            }
        }

        // See data as a (m_bins x m_frames) array of (1 x I) matrices
        ArrayMatrixXcd XPrim = ArrayMatrixXcd(m_bins, m_frames);
        for (int n = 0; n < m_frames; n++) 
        {
            for (int f = 0; f < m_bins; f++) 
            {
                XPrim(f, n) = MatrixXcd(1,I);
                for (int i = 0; i < I; i++) 
                {
                    XPrim(f, n)(0,i) = X(i)(f, n);
                }
            }
        }
        return XPrim;
    }

	void STFTRepr::Filter(const Audio & x, Wiener_Interface * wiener, const std::string & dirName) {
    
        // Checking window length
        if (m_wlen % 4 != 0 || m_wlen == 0) 
        {
            stringstream s;
            s << "Error:\twlen is " << m_wlen << " and should be multiple of 4.\n";
            throw runtime_error(s.str());
        }
      
        // Computing TF representation
        ArrayMatrixXcd Xfram = directFraming(x);
        int F = bins();
        int N = frames();
        int J = wiener->sources();
    
        // Checking if dimensions are consistent
        if (F != wiener->bins()) {
            stringstream s;
            s << "Error:\tnumber of bins is not consistent:\n";
            s << "F = " << F << " in wavfile\n";
            s << "F = " << wiener->bins() << " in xml file\n";
            throw runtime_error(s.str());
        }
        if (N != wiener->frames()) {
            stringstream s;
            s << "Error:\tnumber of frames is not consistent:\n";
            s << "N = " << N << " in wavfile\n";
            s << "N = " << wiener->frames() << " in xml file\n";
            throw runtime_error(s.str());
        }
         
        // Source estimation: Eq. 31
        // This parfor reduce time computation but increase amout of memory
        //#pragma omp parallel for 
        for (int j = 0; j < J; j++) 
        {
            cout << "      Wiener filtering of source "<< j+1 << "/"<< J << endl;
            ArrayMatrixXcd Y = ArrayMatrixXcd(F, N);
            for (int n = 0; n < N; n++) 
            {
                for (int f = 0; f < F; f++) 
                {
					MatrixXcd W = wiener->computeW(n, f, j);
    				Y(f, n) = Xfram(f, n) * W.transpose(); // 
            }
        }
        
        // Computing TF inverse + write out audio for source j
        Audio yj(inverse(Y),x.samplerate());
        yj.write(dirName + wiener->name(j) + ".wav", x.samplerate());
        }
    }

    ArrayXXd STFTRepr::inverse(const ArrayMatrixXcd & X ) const
    {
        int I = static_cast<int>(X(0,0).cols());
        int F = bins();
        int N = frames();
    
        // Defining sine window
        ArrayXd win = Eigen::sin(ArrayXd::LinSpaced(m_wlen, 0.5, m_wlen - 0.5) / m_wlen * M_PI);
    
        // Pre-processing for edges
        ArrayXd swin = ArrayXd::Zero(m_samples);
        for (int n = 0; n < N; n++) 
        {
            swin.segment(n * m_wlen / 2, m_wlen) += (win * win);
        }
        swin = Eigen::sqrt(swin / m_wlen);
    
        // See data as a I-vector of F-by-N-arrays
        VectorMatrixXcd XPrim(I);
        for (int i = 0; i < I; i++) 
        {
            XPrim(i) = ArrayXXcd(F, N);
            for (int n = 0; n < N; n++) 
            {
                for (int f = 0; f < F; f++) 
                {
                    XPrim(i)(f, n) = X(f, n)(0,i);
                }
            }
        }
    
        ArrayXXd x = ArrayXXd::Zero(m_samples, I);
        FFT<double> fft;
        for (int i = 0; i < I; i++) 
        {
            for (int n = 0; n < N; n++) 
            {
                // IFFT
                VectorXcd fframe(m_wlen);
                fframe.segment(0, F) = XPrim(i).col(n);
                fframe.segment(F, F - 2) = XPrim(i).col(n).segment(1, F - 2).reverse();
                VectorXd frame;
                fft.inv(frame, fframe);
    
                // Overlap-add
                x.col(i).segment(n * m_wlen / 2, m_wlen) +=
                    frame.array() * win / swin.segment(n * m_wlen / 2, m_wlen);
            }
        }    
        // Truncation
        ArrayXXd xx = x.block(m_wlen / 4, 0, m_origSamples, I);
        return xx;
    }
}
