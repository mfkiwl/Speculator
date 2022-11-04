/*F***************************************************************************
 * This file is part of openSMILE.
 * 
 * Copyright (c) audEERING GmbH. All rights reserved.
 * See the file COPYING for details on license terms.
 ***************************************************************************E*/


/*  openSMILE component:

example vectorProcessor descendant

*/


#ifndef __CPITCHACF_HPP
#define __CPITCHACF_HPP

#include <core/smileCommon.hpp>
#include <core/vectorProcessor.hpp>

#define COMPONENT_DESCRIPTION_CPITCHACF "This component computes the fundamental frequency and the probability of voicing via an acf and cepstrum based method. The input must be an acf field and a cepstrum field (both generated by a cAcf component)."
#define COMPONENT_NAME_CPITCHACF "cPitchACF"

class cPitchACF : public cVectorProcessor {
  private:
    int HNR;
    int HNRdB, linHNR;
	  int F0, F0raw;
	  int F0env;
    int voiceProb, voiceQual;
	  int onsFlag;
    double maxPitch;
	  double voicingCutoff;
	  FLOAT_DMEM lastPitch, lastlastPitch, glMeanPitch, pitchEnv;
	  float fsSec;

  protected:
    SMILECOMPONENT_STATIC_DECL_PR

    virtual void myFetchConfig() override;
    //virtual int myConfigureInstance() override;
    //virtual int myFinaliseInstance() override;
    //virtual eTickResult myTick(long long t) override;

    //virtual int configureWriter(const sDmLevelConfig *c) override;

    //virtual void configureField(int idxi, long __N, long nOut) override;
    //virtual int setupNamesForField(int i, const char*name, long nEl) override;
	  virtual int setupNewNames(long nEl) override;
    virtual int processVector(const FLOAT_DMEM *src, FLOAT_DMEM *dst, long Nsrc, long Ndst, int idxi) override;
    double computeHNR(const FLOAT_DMEM *a, int f0Idx);
    double computeHNR_dB(const FLOAT_DMEM *a, int f0Idx);
    double computeHNR_lin(const FLOAT_DMEM *a, int f0Idx);

  public:
    SMILECOMPONENT_STATIC_DECL
    
    cPitchACF(const char *_name);
    
    double voicingProb(const FLOAT_DMEM *a, int n, int skip, double *Zcr);
    long pitchPeak(const FLOAT_DMEM *a, long n, long skip);

    virtual ~cPitchACF();
};




#endif // __CPITCHACF_HPP
