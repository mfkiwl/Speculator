
#include <ext.h>
#include <ext_obex.h>
#include <z_dsp.h>

#include <HIRT_Core_Functions.hpp>
#include <HIRT_Buffer_Access.hpp>

// Define common attributes and the class name (for the common attributes file)

#define OBJ_CLASSNAME t_bufconvolve
#define OBJ_USES_HIRT_WRITE_ATTR
#define OBJ_USES_HIRT_READ_ATTR
#define OBJ_USES_HIRT_DECONVOLUTION_ATTR

#include <HIRT_Common_Attribute_Vars.hpp>

// Object class and structure

t_class *this_class;

struct t_bufconvolve
{
    t_pxobject x_obj;

    // Attributes

    HIRT_COMMON_ATTR

    // Bang Out

    void *process_done;
};


// This include deals with setup of common attributes - requires the object structure to be defined

#include <HIRT_Common_Attribute_Setup.hpp>


// Function prototypes

void *bufconvolve_new(t_symbol *s, short argc, t_atom *argv);
void bufconvolve_free(t_bufconvolve *x);
void bufconvolve_assist(t_bufconvolve *x, void *b, long m, long a, char *s);

void bufconvolve_process(t_bufconvolve *x, t_symbol *sym, long argc, t_atom *argv);
void bufconvolve_process_internal(t_bufconvolve *x, t_symbol *sym, short argc, t_atom *argv);


//////////////////////////////////////////////////////////////////////////
/////////////////////// Main / New / Free / Assist ///////////////////////
//////////////////////////////////////////////////////////////////////////


int C74_EXPORT main()
{
    this_class = class_new("bufconvolve~",
                          (method) bufconvolve_new,
                          (method)bufconvolve_free,
                          sizeof(t_bufconvolve),
                          0L,
                          A_GIMME,
                          0);

    class_addmethod(this_class, (method)bufconvolve_process, "convolve", A_GIMME, 0L);
    class_addmethod(this_class, (method)bufconvolve_process, "deconvolve", A_GIMME, 0L);

    class_addmethod(this_class, (method)bufconvolve_assist, "assist", A_CANT, 0L);
    class_register(CLASS_BOX, this_class);

    declare_HIRT_common_attributes(this_class);

    return 0;
}


void *bufconvolve_new(t_symbol *s, short argc, t_atom *argv)
{
    t_bufconvolve *x = reinterpret_cast<t_bufconvolve *>(object_alloc(this_class));

    x->process_done = bangout(x);

    init_HIRT_common_attributes(x);
    attr_args_process(x, argc, argv);

    return x;
}


void bufconvolve_free(t_bufconvolve *x)
{
    free_HIRT_common_attributes(x);
}


void bufconvolve_assist(t_bufconvolve *x, void *b, long m, long a, char *s)
{
    if (m == ASSIST_INLET)
        sprintf(s,"Instructions In");
    else
        sprintf(s,"Bang on Success");
}


//////////////////////////////////////////////////////////////////////////
///////////////////////////// User Messages //////////////////////////////
//////////////////////////////////////////////////////////////////////////


void bufconvolve_process(t_bufconvolve *x, t_symbol *sym, long argc, t_atom *argv)
{
    t_atom temp_argv[4];
    double time_mul = 1.;

    // Load and check arguments

    if (argc < 3)
    {
        object_error((t_object *) x, "not enough arguments to message %s", sym->s_name);
        return;
    }

    if (sym == gensym("deconvolve") && argc > 3)
    {
        time_mul = atom_getfloat(argv + 3);

        if (time_mul < 1.)
        {
            object_warn((t_object *) x, " time multiplier cannot be less than 1 (using 1)");
            time_mul = 1;
        }
    }

    temp_argv[0] = *argv++;
    temp_argv[1] = *argv++;
    temp_argv[2] = *argv++;
    atom_setfloat(temp_argv + 3, time_mul);

    defer(x, (method) bufconvolve_process_internal, sym, 4, temp_argv);
}


void bufconvolve_process_internal(t_bufconvolve *x, t_symbol *sym, short argc, t_atom *argv)
{
    FFT_SPLIT_COMPLEX_D spectrum_1;
    FFT_SPLIT_COMPLEX_D spectrum_2;
    FFT_SPLIT_COMPLEX_D spectrum_3;
    
    bool convolve_mode = sym == gensym("convolve") ? true : false;

    t_symbol *target = atom_getsym(argv++);
    t_symbol *source_1 = atom_getsym(argv++);
    t_symbol *source_2 = atom_getsym(argv++);
    t_symbol *filter = filter_retriever(x->deconvolve_filter_specifier);

    double filter_specifier[HIRT_MAX_SPECIFIER_ITEMS];
    double range_specifier[HIRT_MAX_SPECIFIER_ITEMS];

    double time_mul = atom_getfloat(argv++);
    double sample_rate = buffer_sample_rate(source_1);
    double deconvolve_phase = phase_retriever(x->deconvolve_phase);
    double deconvolve_delay;

    intptr_t source_length_1 = buffer_length(source_1);
    intptr_t source_length_2 = buffer_length(source_2);
    intptr_t filter_length = buffer_length(filter);

    uintptr_t fft_size;
    uintptr_t fft_size_log2;

    t_filter_type deconvolve_mode = static_cast<t_filter_type>(x->deconvolve_mode);
    t_atom_long read_chan = x->read_chan - 1;

    // Check input buffers

    if (buffer_check((t_object *) x, source_1) || buffer_check((t_object *) x, source_2))
        return;

    // Check sample rates

    if (sample_rate != buffer_sample_rate(source_2))
        object_warn((t_object *) x, "sample rates do not match");

    // Check and calculate lengths

    if (convolve_mode)
        fft_size = static_cast<uintptr_t>((source_length_1 + source_length_2) * time_mul);
    else
        fft_size = static_cast<uintptr_t>(source_length_1 < source_length_2 ? source_length_2 * time_mul : source_length_1 * time_mul);

    fft_size = calculate_fft_size(fft_size, fft_size_log2);
    deconvolve_delay = delay_retriever(x->deconvolve_delay, fft_size, sample_rate);

    if (fft_size < 8)
    {
        object_error((t_object *) x, "input buffers are too short, or have no length");
        return;
    }

    // Allocate Memory (use pointer aliasing where possible for efficiency)

    temp_fft_setup fft_setup(fft_size_log2);
    
    temp_ptr<double> temp(fft_size * (convolve_mode ? 3 : 4));
    temp_ptr<float> filter_in(filter_length);

    spectrum_1.realp = temp.get();
    spectrum_1.imagp = spectrum_1.realp + (fft_size >> 1);
    spectrum_2.realp = spectrum_1.imagp + (fft_size >> 1);
    spectrum_2.imagp = spectrum_2.realp + (fft_size >> 1);
    spectrum_3.realp = spectrum_2.imagp + (fft_size >> 1);
    spectrum_3.imagp = convolve_mode ? 0 : spectrum_3.realp + fft_size;

    double *out_buf = spectrum_2.realp;
    float *in_temp = reinterpret_cast<float *>(spectrum_3.realp);

    // Check memory allocations

    if (!fft_setup || !spectrum_1.realp || (filter_length && !filter_in))
    {
        object_error((t_object *) x, "could not allocate temporary memory for processing");
        return;
    }

    // Get inputs - convert to frequency domain

    buffer_read(source_1, read_chan, in_temp, source_length_1);
    time_to_halfspectrum_float(fft_setup, in_temp, source_length_1, spectrum_1, fft_size);
    buffer_read(source_2, read_chan, in_temp, source_length_2);
    time_to_halfspectrum_float(fft_setup, in_temp, source_length_2, spectrum_2, fft_size);

    // Do deconvolution or convolution

    if (convolve_mode)
        convolve(spectrum_1, spectrum_2, fft_size, SPECTRUM_REAL);
    else
    {
        // Fill deconvolution filter specifiers - load filter from buffer (if specified) - deconvolve

        fill_power_array_specifier(filter_specifier, x->deconvolve_filter_specifier, x->deconvolve_num_filter_specifiers);
        fill_power_array_specifier(range_specifier, x->deconvolve_range_specifier, x->deconvolve_num_range_specifiers);
        buffer_read(filter, 0, filter_in.get(), fft_size);
        deconvolve(fft_setup, spectrum_1, spectrum_2, spectrum_3, filter_specifier, range_specifier, 0.0, filter_in.get(), filter_length, fft_size, SPECTRUM_REAL, deconvolve_mode, deconvolve_phase, deconvolve_delay, sample_rate);
    }

    // Convert to time domain - copy out to buffer

    spectrum_to_time(fft_setup, out_buf, spectrum_1, fft_size, SPECTRUM_REAL);
    auto error = buffer_write((t_object *) x, target, out_buf, (convolve_mode? source_length_1 + source_length_2 - 1 : fft_size), x->write_chan - 1, x->resize, sample_rate, 1.);

    if (!error)
        outlet_bang(x->process_done);
}
