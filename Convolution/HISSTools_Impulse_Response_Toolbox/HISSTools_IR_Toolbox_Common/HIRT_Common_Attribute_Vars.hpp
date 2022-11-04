
#ifndef __HIRT_COMMON_ATTRIBUTE_VARS__
#define __HIRT_COMMON_ATTRIBUTE_VARS__

// Buffer Write Attributes

#ifdef OBJ_USES_HIRT_WRITE_ATTR
#define HIRT_WRITE_ATTR \
long resize; \
t_atom_long write_chan;
#else
#define HIRT_WRITE_ATTR
#endif

// Buffer Read Attributes

#ifdef OBJ_USES_HIRT_READ_ATTR
#define HIRT_READ_ATTR \
t_atom_long read_chan;
#else
#define HIRT_READ_ATTR
#endif

// Deconvolution Attributes

#if defined OBJ_USES_HIRT_DECONVOLUTION_ATTR && !defined OBJ_DOES_NOT_USE_HIRT_DECONVOLUTION_DELAY
#define HIRT_DECONVOLUTION_DELAY \
t_atom deconvolve_delay;
#else
#define HIRT_DECONVOLUTION_DELAY
#endif

#ifdef OBJ_USES_HIRT_DECONVOLUTION_ATTR
#define HIRT_DECONVOLUTION_ATTR \
t_atom *deconvolve_filter_specifier; \
t_atom *deconvolve_range_specifier; \
long deconvolve_num_filter_specifiers; \
long deconvolve_num_range_specifiers; \
long deconvolve_mode; \
t_atom deconvolve_phase;
#else
#define HIRT_DECONVOLUTION_ATTR
#endif

// Output Phase Attributes

#ifdef OBJ_USES_HIRT_OUT_PHASE_ATTR
#define HIRT_OUT_PHASE_ATTR \
t_atom out_phase;
#else
#define HIRT_OUT_PHASE_ATTR
#endif

// Smoothing Attributes

#ifdef OBJ_USES_HIRT_SMOOTH_ATTR
#define HIRT_SMOOTH_ATTR \
long smooth_mode; \
long num_smooth; \
double smooth[2];
#else
#define HIRT_SMOOTH_ATTR
#endif

#ifdef OBJ_USES_HIRT_SWEEP_AMP_CURVE_ATTR
#define HIRT_SWEEP_AMP_CURVE_ATTR \
t_atom amp_curve_specifier[32]; \
long amp_curve_num_specifiers;
#else
#define HIRT_SWEEP_AMP_CURVE_ATTR
#endif

// Common Atrributes (ALL)

#define HIRT_COMMON_ATTR \
HIRT_WRITE_ATTR \
HIRT_READ_ATTR \
HIRT_DECONVOLUTION_DELAY \
HIRT_DECONVOLUTION_ATTR \
HIRT_OUT_PHASE_ATTR \
HIRT_SMOOTH_ATTR \
HIRT_SWEEP_AMP_CURVE_ATTR

#endif /* __HIRT_COMMON_ATTRIBUTE_VARS__ */
