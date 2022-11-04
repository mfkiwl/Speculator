/***************************************************************************
 *   Copyright (C) 2009 by Christian Borss                                 *
 *   christian.borss@rub.de                                                *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#ifndef __HybridConvolverTripple__
#define __HybridConvolverTripple__


#include "HybridConvolver.h"
#include "SampleData.h"
#include "JuceHeader.h"

extern "C" {
#include "libHybridConv/libHybridConv.h"
}


class HybridConvolverTripple : public HybridConvolver
{
public:
    HybridConvolverTripple(int lenShort,
                           int lenMedium,
                           int lenLong,
                           SampleData *impulses);
    ~HybridConvolverTripple(void);

    void process(const float** inputs,
                 float** outputs,
                 int numOutput,
                 int numSamples) override;

private:
    std::unique_ptr<HConvTripple[]> m_filter;

private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HybridConvolverTripple)
};


#endif // __HybridConvolverTripple__
