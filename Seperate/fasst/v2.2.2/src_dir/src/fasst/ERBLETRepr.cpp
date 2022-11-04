// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "ERBLETRepr.h"
#include <unsupported/Eigen/FFT>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace fasst {

	ERBLETRepr::ERBLETRepr(const int nOrigSamples, const int wlen, const int fs, const int binPerERB)
		: Abstract_TFRepr(nOrigSamples,wlen)
	{

		if (binPerERB <= 0)
			throw runtime_error("Number of bin per ERB must be a stricly positive number");

		// Fixed parameters
		int qVar = 1;
		int mFac = 1;

		// Compute m_frames
		m_frames = static_cast<int>(std::ceil(static_cast<double>(nOrigSamples * 2.) / wlen));

		// Compute m_samples : m_OrigSamples + zeros padded samples
		m_samples = (m_frames + 1)*wlen / 2;

		double df = static_cast<double>(fs) / m_samples;
		Vector2d fLims;
		double fmin = 0.;
		double fmax = static_cast<double>(fs) / 2;
		fLims << fmin , fmax; // Frequency limits in Hz
		Vector2d erbLims = 9.2645*sign(fLims.array().sign()).cwiseProduct((1 + fLims.array().abs()*0.00437).log());

		// Number of reliable freq channels
		m_bins = binPerERB * static_cast<int>(std::ceil(erbLims(1) - erbLims(0)));
		
		// Full number of bins (with the miroted part)
		m_fullBins = 2 * m_bins - 1; 

		// compute central frequencies
		m_fc = VectorXd::LinSpaced(m_bins, erbLims(0), erbLims(1));
		m_fc = (1. / 0.00437)*(m_fc.array().sign().cwiseProduct((m_fc.array().abs()/ 9.2645).exp() - 1));
		
		// Set the endpoints to be exactly what the user specified, instead of the computed values
		m_fc(0) = fmin;
		m_fc(m_fc.rows() - 1) = fmax;

		// Compute m_gamma
		m_gamma = (m_fc / 9.265).array() + 24.7;

		// compute posit and m_shift
		 m_posit = VectorXi(m_fullBins);
		VectorXi fc_rounded = (m_fc.array() / df).round().cast<int>();
        m_posit.head(m_bins) = fc_rounded;
        m_posit.tail(m_bins - 1) = m_samples - fc_rounded.head(m_bins -1).reverse().array();
		
		m_shift = VectorXi(m_fullBins);
		m_shift(0) = m_samples - m_posit(m_posit.size() - 1);
		m_shift.tail(2* m_bins - 2) = m_posit.tail(2* m_bins - 2) - m_posit.head(2*m_bins - 2);

		// Compute desired essential (gaussian) support for each filter (lwin) and M
		VectorXi lwin = 4 * ((m_gamma.array() / df).round().cast<int>());
		m_M = (qVar* lwin.array().cast<double>() / 1.1).round().cast<int>();

		// adjust M to be a multiple of (nFrame + 1)
		int multiple = m_frames + 1;
		VectorXi tmp = modulo(m_M, multiple);
		m_M = m_M.array() +  ((tmp.array() != 0).cast<int>()).cwiseProduct(multiple - tmp.array());

		// Compute the Analysis filter m_g_an
		m_g_an = vector<VectorXd>(m_bins);
		for (int i = 0; i < m_bins; i++) {
			m_g_an[i] = (firwin(m_M(i)).array() / sqrt(m_M(i)));
		}
		m_g_an[0] = (1. / sqrt(2))*m_g_an[0].array();

		m_M = mFac * (m_M.array() / mFac).ceil();
		
		// compute m_g_syn 
		nsgabdual_painless();
	}


	VectorXd ERBLETRepr::firwin(int M) {
		double step = 1. / M;
		VectorXd x;
		if ((M % 2) == 0) {
			//VectorXi::LinSpaced(((hi - low) / step) + 1,low, low + step * (size - 1))  
			//  low:step:hi
			int size1 = static_cast<int>(round((0.5-step-0.) / step) + 1);
			int size2 = static_cast<int>(round((-step + 0.5) / step) + 1);
			x = VectorXd(size1 + size2);
			x.head(size1) = VectorXd::LinSpaced(size1, 0., 0. + step*(size1-1)).array();
			x.tail(size2) = VectorXd::LinSpaced(size2, -0.5, -0.5 + step * (size2 - 1)).array();
		}
		else {
			int size1 = static_cast<int>(round((0.5-0.5*step - 0) / step) + 1);
			int size2 = static_cast<int>(round((-step -(-0.5+0.5*step)) / step) + 1);
			x = VectorXd(size1 + size2);
			x.head(size1) = VectorXd::LinSpaced(size1, 0., 0. + step * (size1 - 1)).array();
			x.tail(size2) = VectorXd::LinSpaced(size2, -0.5 + 0.5*step, -0.5 + 0.5*step + step * (size2 - 1)).array();

		}
		// 'nuttall' window
		VectorXd g = 0.355768 + 0.487396*(x.array()*2.*M_PI).cos() + 0.144232*(x.array() * 4. * M_PI).cos() + 0.012604*(x.array() * 6. * M_PI).cos();

		// Force the window to be 0 outside(-.5, .5)
		g = g.array().cwiseProduct((x.array().abs() < 0.5).cast<double>());

		return g;
	}


	void ERBLETRepr::nsgabdual_painless() {
		// The painless case is considered in this implementation !

		VectorXi timepos = cumsum(m_shift).array() - m_shift(0);

		// Compute the diagonal of the frame operator
		VectorXd f = nsgabframediag(timepos);

		// Initialize m_g_syn
		m_g_syn = vector<VectorXd>(m_fullBins);

		// Correct each window to ensure perfect reconstrution
		for (int i = 0; i < m_fullBins; i++) {
			int flipInd;
			if (i >= m_bins) {
				flipInd = static_cast<int>(m_bins - 1 - (i - (m_bins - 1)));
			}
			else {
				flipInd = i;
			}
			int shift = static_cast<int>(std::floor(static_cast<double>(m_g_an[flipInd].rows()) / 2.));
			int start = 1;
			int end = static_cast<int>(m_g_an[flipInd].rows());
			int size = end - start + 1;
			VectorXi linInd = VectorXi::LinSpaced(size, start, end).array() + timepos(i) - shift - 1;
			VectorXi tempInd = modulo(linInd, m_samples).array();
			VectorXd currentf = VectorXd(size);
			for (int ii = 0; ii < size; ii++) {
				currentf(ii) = f(tempInd(ii));
			}
			VectorXd tmp = circshift(m_g_an[flipInd], shift).cwiseQuotient(currentf);
			m_g_syn[i] = circshift(tmp, -shift);
		}
	}


	VectorXd ERBLETRepr::nsgabframediag(const VectorXi timepos) {
		// L = m_samples
		VectorXd f = VectorXd::Zero(m_samples);

		for (int i = 0; i < m_fullBins; i++) {
			
			// Compute the flipped index to access M
			int flipInd;
			if (i >= m_bins) {
				flipInd = static_cast<int>(m_bins - 1 - (i - (m_bins-1)));
			} else{
				flipInd = i;
			}
			int shift = static_cast<int>(std::floor(static_cast<double>(m_g_an[flipInd].rows()) / 2.));
			VectorXd temp = (circshift(m_g_an[flipInd], shift).array().abs2())*m_M(flipInd);
			int start = 1;
			int end = static_cast<int>(m_g_an[flipInd].rows());
			int size = end - start + 1;
			VectorXi linInd = VectorXi::LinSpaced(size,start,end).array() + timepos(i) - shift - 1;
			VectorXi tempInd = modulo(linInd, m_samples).array();
			for (int ii = 0; ii < size; ii++) {
				f(tempInd(ii)) = f(tempInd(ii)) + temp(ii);
			}
		}
		return f;
	}
	

	template <typename DerivedIn>
	DerivedIn ERBLETRepr::cumsum(const DerivedIn in) const {
		DerivedIn out = DerivedIn(in.rows(),in.cols());
			for (int j = 0; j < in.cols(); j++) {
				out(0, j) = in(0, j);
				for (int i = 1; i < in.rows(); i++) {
					out(i,j) = out(i - 1,j) + in(i,j);
				}
			}
		return out;
	}


	template <typename DerivedIn>
	DerivedIn ERBLETRepr::circshift(const DerivedIn & in, int k) const{
		if (!k) return in;
		DerivedIn out(static_cast<int>(in.rows()), static_cast<int>(in.cols()));
		if (k > 0) k = k % in.rows();
		else k = static_cast<int>(in.rows()) - (-k % static_cast<int>(in.rows()));
		// We avoid the implementation-defined sign of modulus with negative arg. 
		int rest = static_cast<int>(in.rows()) - k;
		out.topRows(k) = in.bottomRows(k);
		out.bottomRows(rest) = in.topRows(rest);
		return out;
	}


    template <typename DerivedIn>
    DerivedIn ERBLETRepr::modulo(const DerivedIn & in, int k) const {
		//auto in2 = in.array().cast<double>();
		//auto in2 = ;
		//DerivedIn out = in.array() - (k *((in.template cast<double>() / k).floor()).cast<int>());
        //DerivedIn out = in.array() - (k *(Eigen::floor((in2 / k))).cast<int>());
		DerivedIn out = in.array() - (k * ((Eigen::floor(( (in.template cast<double>()).array() / k))).template cast<int>()).array() );
        return out;
    }
	
	ArrayMatrixXcd ERBLETRepr::direct(const Audio & x) const
    {
        // check audio length vs m_OrigSamples
        if (x.rows() != m_origSamples)
        {
            throw runtime_error(
                "Check your parameters: The number of samples used to initialize ERBLET filter banks is not equal "
                "to the length of audio signal sent to the analysis transform");
        }

        int I = x.channels(); // number of channels
        
		ArrayMatrixXcd output;
        output = ArrayMatrixXcd(m_bins,1); // vector of size m_bins
        for (int binId = 0; binId < m_bins; binId++)
        {
            output(binId,0) = ArrayXXcd::Zero(m_M(binId), I);
        }

        FFT<double> fft;
        VectorXd xPadded = VectorXd::Zero(m_samples);

        for (int channelId=0 ; channelId<I ; channelId++)
        {
            // Fill xPadded with samples of current chan
            xPadded.head(m_origSamples) = x.col(channelId);

            // FFT direct
            VectorXcd X;
            fft.fwd(X, xPadded);

            for (int binId=0 ; binId<m_bins ; binId++)
            {
                int Lg = static_cast<int>(m_g_an[binId].size());

                // idx 
                // idx = [ceil(Lg / 2) + 1:Lg, 1 : ceil(Lg / 2)];
                int start1  = static_cast<int>(ceil(Lg / 2.0));
                int end1    = Lg - 1;
                int size1   = end1 - start1 + 1;
                int start2  = 0;
                int end2    = static_cast<int>(ceil(Lg / 2.0)) -1;
                int size2   = end2 - start2 + 1;
                VectorXi idx = VectorXi(size1+size2);
                idx.head(size1) = VectorXi::LinSpaced(size1, start1, end1);
                idx.tail(size2) = VectorXi::LinSpaced(size2, start2, end2);

                // winrange 
                // win_range = mod(posit(ii)+(-floor(Lg/2):ceil(Lg/2)-1),nSamples)+1;
                int start   = -1*static_cast<int>(floor(Lg / 2.0));
                int end     = static_cast<int>(ceil(Lg / 2.0)) - 1;
                int size    = end - start + 1;
                VectorXi tempVectorXi = VectorXi::LinSpaced(size, start, end).array() + m_posit(binId);                
                VectorXi win_range = modulo(tempVectorXi,m_samples).array();

                if (m_M(binId) < Lg) // if the number of frequency channels is too small
                {
                    // Not sure that this use case is a real one
                    cout << "The number of frequency bins / subbands is too small" << endl;
                }
                else
                {

                    // temp
                    // temp([end-floor(Lg/2)+1:end,1:ceil(Lg/2)],:) = bsxfun(@times,X(win_range, :), g{ ii }(idx));
                    VectorXcd temp = VectorXcd::Zero(m_M(binId));

                    // should be optimized? win_range is the concatenation of 2 linespace ?
                    VectorXcd tempBis = VectorXcd::Zero(win_range.size());
                    for (int i = 0; i < win_range.size(); i++)
                    {
                        tempBis(i) = X(win_range(i))*m_g_an[binId](idx(i));
                    }

                    temp.head(static_cast<int>(ceil(Lg / 2.0))) = tempBis.tail(static_cast<int>(ceil(Lg / 2.0)));
                    temp.tail(static_cast<int>(floor(Lg / 2.0))) = tempBis.head(static_cast<int>(floor(Lg / 2.0)));
                
                    // should be real values: complex part is near 0 ?
                    // FFT inverse
                    VectorXcd xBin;
                    fft.inv(xBin, temp);
                    output(binId,0).col(channelId) = xBin;
                }
            }
        }
        
        return output;
    }

    ArrayMatrixXcd ERBLETRepr::directFraming(const Audio & x) const 
    {
        // Apply direct transform on x
		ArrayMatrixXcd X = direct(x);

        ArrayMatrixXcd Xfram = ArrayMatrixXcd(m_frames, m_bins);

        int I = static_cast<int>(X(0,0).cols());

        for (int f = 0; f < m_bins; f++)
        {
            int wlenBand = 2 * static_cast<int>(X(f,0).rows()) / (m_frames + 1);

            // Define the time integration window
            ArrayXd win =
                Eigen::sin(ArrayXd::LinSpaced(wlenBand, 0.5, wlenBand - 0.5) / wlenBand * M_PI);
            ArrayXd swin = ArrayXd::Zero((m_frames + 1) * wlenBand / 2);

            for (int n = 0; n < m_frames; n++) {
                swin.segment(n * wlenBand / 2, wlenBand) += (win * win);
            }
            swin = Eigen::sqrt(swin);

            for (int n = 0; n < m_frames; n++)
            {
                Xfram(n, f) = MatrixXcd(wlenBand, I);
                for (int i = 0; i < I; i++)
                {
                    Xfram(n, f).col(i) = X(f,0).block(n * wlenBand / 2, i, wlenBand, 1).array() * win / swin.segment(n * wlenBand / 2, wlenBand);
                }
            }
        }
        return Xfram;
    }

    ArrayXXd ERBLETRepr::inverse(const ArrayMatrixXcd & X) const
    {
        // check number of bins
        if (X.rows() != m_bins)
        {
            throw runtime_error(
                "Check your parameters: The number of frequency bins / subbands "
                "was modified between analysis and synthesis filterbank");
        }

        // check length of bins
        for (int binId = 0; binId < m_bins; binId++)
        {
            if (X(binId,0).rows() != m_g_syn[binId].size())
            {
                throw runtime_error(
                    "Check your parameters: The length of at least one bin / subband "
                    "was modified between analysis and synthesis filterbank");
            }
        }


        // Number of channels
        int I = static_cast<int>(X(0,0).cols()); 

        // Init the output
        ArrayXXd output(m_origSamples, I); 

        VectorXi timepos = cumsum(m_shift).array();
        timepos = timepos.array() - m_shift(0);    

        FFT<double> fft;
        for (int channelId=0; channelId<I; channelId++)
        {
            VectorXcd fr = VectorXcd::Zero(m_samples);

            for (int binId=0 ; binId<m_bins ; binId++)
            {
                // FFT direct
                VectorXcd XPrim = X(binId,0).col(channelId);
                VectorXcd XTer;
                fft.fwd(XTer, XPrim);

                // temp = fft(c{ii})*M(ii);
                VectorXcd temp = XTer.array()*(m_M(binId));

                // Lg = length(g{ii});
                int Lg = static_cast<int>(m_g_syn[binId].size());

                // win_range = mod(timepos(ii)+(-floor(Lg/2):ceil(Lg/2)-1),NN)+1;
                int start   = -1 * static_cast<int>(floor(Lg / 2.0));
                int end     = static_cast<int>(ceil(Lg / 2.0)) - 1;
                int size    = end - start + 1;
                VectorXi indexVectorXi1 = VectorXi::LinSpaced(size, start, end);

                VectorXi tempVectorXi = timepos(binId) + indexVectorXi1.array();
                VectorXi win_range = modulo(tempVectorXi, m_samples);

                // temp2 = temp(mod([end-floor(Lg/2)+1:end,1:ceil(Lg/2)]-1,M(ii))+1,:);
                int start1  = static_cast<int>(temp.size()) - 1 * static_cast<int>(floor(Lg / 2.0));
                int end1    = static_cast<int>(temp.size()) - 1;
                int size1   = end1 - start1 + 1;
                VectorXi tempVectLinSpa = VectorXi::LinSpaced(size1, start1, end1);
                VectorXi tempVectorXi1 = modulo(tempVectLinSpa, m_M(binId));

                int start2  = 0;
                int end2    = static_cast<int>(ceil(Lg / 2.0)) - 1;;
                int size2   = end2 - start2 + 1;
                tempVectLinSpa = VectorXi::LinSpaced(size2, start2, end2);
                VectorXi tempVectorXi2 = modulo(tempVectLinSpa, m_M(binId));

                // concatenate tempVectorXi1 and tempVectorXi2 => indexVectorXi2
                VectorXi indexVectorXi2 = VectorXi::Zero(size1 + size2);
                indexVectorXi2.head(size1) = tempVectorXi1;
                indexVectorXi2.tail(size2) = tempVectorXi2;

                VectorXcd temp2 = VectorXcd::Zero(size1 + size2);
                for (int i = 0; i < size1 + size2; i++)
                {
                    temp2(i) = temp(indexVectorXi2(i));
                }

                // fr(win_range,:) = fr(win_range,:) + bsxfun(@times,temp2, g{ ii }([Lg - floor(Lg / 2) + 1:Lg, 1 : ceil(Lg / 2)]));
                start1   = Lg - 1 * static_cast<int>(floor(Lg / 2.0));
                end1     = Lg-1;
                size1    = end1 - start1 + 1;
                tempVectorXi1 = VectorXi::LinSpaced(size1, start1, end1).array();

                start2 = 0;
                end2 = static_cast<int>(ceil(Lg / 2.0)) - 1;
                size2 = end2 - start2 + 1;
                tempVectorXi2 = VectorXi::LinSpaced(size2, start2, end2).array();

                // concatenate tempVectorXi1 and tempVectorXi2
                VectorXi indexVectorXi3 = VectorXi::Zero(size1 + size2);
                indexVectorXi3.head(size1) = tempVectorXi1;
                indexVectorXi3.tail(size2) = tempVectorXi2;

                for (int i=0 ; i<temp2.size() ; i++)
                {
                    fr(win_range(i)) = fr(win_range(i)) + temp2(i)*m_g_syn[binId](indexVectorXi3(i));
                }

                if (binId < m_bins-1)
                {
                    // Lg = length(g{Ntot-ii+1}) == length(g{ii}); // Do not compute it again
                    // win_range = mod(timepos(Ntot - ii + 1) + (-floor(Lg / 2) :ceil(Lg / 2) - 1), NN) + 1;                   
                    tempVectorXi = timepos(m_fullBins - binId - 1) + indexVectorXi1.array();
                    win_range = modulo(tempVectorXi, m_samples);

                    //temp(1:end) = conj([temp(1);temp(end:-1:2)]);
                    VectorXcd tempFlip = temp.reverse(); 
                    temp.tail(tempFlip.size() - 1) = tempFlip.head(tempFlip.size() - 1); // temp(1) is not modified !
					temp = temp.conjugate();

                    // temp2 = temp(mod([end - floor(Lg / 2) + 1:end, 1 : ceil(Lg / 2)] - 1, M(Ntot - ii + 1)) + 1, :);
                    // fr(win_range,:) = fr(win_range,:) + bsxfun(@times,temp2, g{ Ntot - ii + 1 }([Lg - floor(Lg / 2) + 1:Lg, 1 : ceil(Lg / 2)]));
                    for (int i = 0; i<temp2.size(); i++)
                    {
                        fr(win_range(i)) = fr(win_range(i)) + temp(indexVectorXi2(i))*m_g_syn[m_fullBins - binId -1](indexVectorXi3(i));
                    }
                }
            }

            // Fill the output
            VectorXcd xPadded_currentChan;
            fft.inv(xPadded_currentChan, fr);
            output.col(channelId) = xPadded_currentChan.real().head(m_origSamples); // Only keep the real part
        }

        return output;
    }

	void ERBLETRepr::Filter(const Audio & x, Wiener_Interface * wiener, const std::string & dirName) {

		int F = bins();
		int N = frames();
		int J = wiener->sources();
		int I = wiener->channels();

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

		// Compute the framed TF representation
		ArrayMatrixXcd Xfram = directFraming(x);

		// Source estimation: Eq. 31
		// This parfor reduce time computation but increase amout of memory
		//#pragma omp parallel for 
		for (int j = 0; j < J; j++) {
			cout << "      Wiener filtering of source " << j + 1 << "/" << J << endl;
			ArrayMatrixXcd Xj = ArrayMatrixXcd(F,1);
			
				for (int f = 0; f < F; f++) {
						int wlenBand = 2 * static_cast<int>(filterSize(f)) / (frames() + 1);
						ArrayXd win =
							Eigen::sin(ArrayXd::LinSpaced(wlenBand, 0.5, wlenBand - 0.5) / wlenBand * M_PI);
						
						ArrayXd swin = ArrayXd::Zero((frames() + 1) * wlenBand / 2);

						for (int n = 0; n < frames(); n++) {
							swin.segment(n * wlenBand / 2, wlenBand) += (win * win);
						}

						swin = Eigen::sqrt(swin);
						MatrixXcd oband = MatrixXcd::Zero((frames() + 1)*wlenBand / 2, I);
						for (int n = 0; n < N; n++) {
							MatrixXcd W = wiener->computeW(n, f, j);
							oband.block(n * wlenBand / 2, 0, wlenBand, I) = 
								oband.block(n * wlenBand / 2, 0, wlenBand, I).array() + ((Xfram(n, f).array() * (win.array() / swin.segment(n * wlenBand / 2, wlenBand)).replicate(1, I).array()).matrix() * W.transpose()).array();
						}
				Xj(f,0) = oband;
				}
				// Computing TF inverse + write out audio
				Audio xj(inverse(Xj),x.samplerate());
				xj.write(dirName + wiener->name(j) + ".wav", x.samplerate());
				
			
		}
	}
}

