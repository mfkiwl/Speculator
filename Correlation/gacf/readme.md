![Python Test Status](https://github.com/joshbriegal/gacf/workflows/Python%20tests/badge.svg)

# Generalised Autocorrelation Function

(credit Lars Kreutzer, c++ implementation by Josh Briegal jtb34@cam.ac.uk)

## Installation

Requirements:

* CMAKE (https://cmake.org) > 3.8.
* C++14

From above top level directory run

```
pip install ./gacf
```

in python:


GACF follows Astropy LombScargle implementation:

```python
from gacf import GACF

lag_timeseries, correlations = GACF(timeseries, values, errors=None).autocorrelation()
```

with options:

```python
gacf.autocorrelation(max_lag=None, lag_resolution=None, selection_function='natural', weight_function='fast', alpha=None)
```

NOTE: If users specify `selection_function="fast"` or `weight_function="gaussian"`, a python implementation of the GACF will be invoked which is considerably slower than the default C++ option.

### Tests

From root directory run:

```python
tox
```
