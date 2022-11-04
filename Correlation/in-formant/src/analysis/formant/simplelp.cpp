#include "formant.h"
#include "../util/util.h"

using namespace Analysis::Formant;
using Analysis::FormantResult;

FormantResult SimpleLP::solve(const double *lpc, int lpcOrder, double sampleRate)
{
    rpm::vector<double> polynomial(lpcOrder + 1);
    polynomial[0] = 1.0;
    std::copy(lpc, lpc + lpcOrder, std::next(polynomial.begin()));
    
    rpm::vector<std::complex<double>> roots = findRoots(polynomial);

    FormantResult result;

    const double phiDelta = 2.0 * 50.0 * M_PI / sampleRate;

    for (const auto& z : roots) {
        if (z.imag() < 0) continue;

        double r = std::abs(z);
        double phi = std::arg(z);

        if (r < 0.7 || r > 1.0 || phi < phiDelta || phi > M_PI - phiDelta) {
            continue;
        }

        result.formants.push_back(calculateFormant(r, phi, sampleRate));
    }

    sortFormants(result.formants);

    return result;
}
