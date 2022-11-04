//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
//                          Felix Weninger <felix@weninger.de>
//                          Bjoern Schuller <schuller@tum.de>
//
// Institute for Human-Machine Communication
// Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
//
// openBliSSART is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 2 of the License, or (at your option) any later
// version.
//
// openBliSSART is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
//


#include "NMFTest.h"
#include <blissart/nmf/Deconvolver.h>
#include <blissart/linalg/generators/generators.h>
#include <iostream>
#include <ctime>
#include <cstdlib>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


namespace Testing {


bool NMFTest::outputAndCheck(Matrix& x, nmf::Deconvolver& d)
{
    cout << "W = " << endl;
    cout << d.getW(0) << endl;
    cout << "H = " << endl;
    cout << d.getH() << endl;
    cout << "Approx = " << endl;
    d.computeApprox();
    Matrix l(d.getApprox());
    cout << l << endl;
    
    for (unsigned int i = 0; i < x.rows(); i++) {
        for (unsigned int j = 0; j < x.cols(); j++) {
            if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                return false;
        }
    }

    return true;
}


bool NMFTest::performTest()
{
    srand(1);

    cout << "Creating 10x5 random matrix:" << endl;
    Matrix x(10, 5, nmf::gaussianRandomGenerator);
    cout << x;
    cout << "---" << endl;

    {
        cout << "Performing NMF using IS divergence" << endl;

        nmf::Deconvolver d(x, 10, 1);
        d.decompose(nmf::Deconvolver::ISDivergence, 5000, 1e-5);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;

        if (!outputAndCheck(x, d)) return false;
    }

    srand(1);

    {
        cout << "Performing NMF using KL divergence" << endl;

        nmf::Deconvolver d(x, 10, 1);
        d.decompose(nmf::Deconvolver::KLDivergence, 5000, 1e-5);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;

        if (!outputAndCheck(x, d)) return false;
    }

    {
        cout << "Performing NMF using Euclidean distance" << endl;

        {
            srand(2);
            cout << endl << "--- Overcomplete" << endl;
            nmf::Deconvolver d(x, 10, 1);
            //d.setAlgorithm(nmf::Deconvolver::Overcomplete);
            d.decompose(nmf::Deconvolver::EuclideanDistance, 5000, 1e-5);
            cout << "# steps: " << d.numSteps() << endl;
            cout << "absolute error: " << d.absoluteError() << endl;
            cout << "relative error: " << d.relativeError() << endl;
            cout << endl;

            /*cout << d.getW(0) << endl;
            cout << d.getH() << endl;*/
            if (!outputAndCheck(x, d)) return false;
        }
        {
            srand(2);
            cout << endl << "--- Incomplete" << endl;
            nmf::Deconvolver d(x, 10, 1);
            //d.setAlgorithm(nmf::Deconvolver::Incomplete);
            d.factorizeNMFEDIncomplete(5000, 1e-5);
            //d.decompose(nmf::Deconvolver::EuclideanDistance, 5000, 1e-5);
            cout << "# steps: " << d.numSteps() << endl;
            cout << "absolute error: " << d.absoluteError() << endl;
            cout << "relative error: " << d.relativeError() << endl;
            cout << endl;
            /*cout << d.getW(0) << endl;
            cout << d.getH() << endl;*/

            if (!outputAndCheck(x, d)) return false;
        }
    }

    return true;
}


} // namespace Testing
