
#include <ext.h>
#include <ext_obex.h>
#include <z_dsp.h>

#include <HIRT_Core_Functions.hpp>
#include <HIRT_Buffer_Access.hpp>

// Define common attributes and the class name (for the common attributes file)

#define OBJ_CLASSNAME t_irtransaural
#define OBJ_USES_HIRT_WRITE_ATTR
#define OBJ_USES_HIRT_READ_ATTR
#define OBJ_USES_HIRT_DECONVOLUTION_ATTR

#include <HIRT_Common_Attribute_Vars.hpp>


// Object class and structure

t_class *this_class;

struct t_irtransaural
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

void *irtransaural_new(t_symbol *s, short argc, t_atom *argv);
void irtransaural_free(t_irtransaural *x);
void irtransaural_assist(t_irtransaural *x, void *b, long m, long a, char *s);

void irtransaural_process(t_irtransaural *x, t_symbol *sym, long argc, t_atom *argv);
void irtransaural_process_internal(t_irtransaural *x, t_symbol *sym, short argc, t_atom *argv);


//////////////////////////////////////////////////////////////////////////
/////////////////////// Main / New / Free / Assist ///////////////////////
//////////////////////////////////////////////////////////////////////////


int C74_EXPORT main()
{
    this_class = class_new("irtransaural~",
                          (method) irtransaural_new,
                          (method)irtransaural_free,
                          sizeof(t_irtransaural),
                          0L,
                          A_GIMME,
                          0);

    class_addmethod(this_class, (method)irtransaural_process, "lattice", A_GIMME, 0L);
    class_addmethod(this_class, (method)irtransaural_process, "shuffler", A_GIMME, 0L);

    class_addmethod(this_class, (method)irtransaural_assist, "assist", A_CANT, 0L);
    class_register(CLASS_BOX, this_class);

    declare_HIRT_common_attributes(this_class);

    return 0;
}


void *irtransaural_new(t_symbol *s, short argc, t_atom *argv)
{
    t_irtransaural *x = reinterpret_cast<t_irtransaural *>(object_alloc(this_class));

    x->process_done = bangout(x);

    init_HIRT_common_attributes(x);
    attr_args_process(x, argc, argv);

    return x;
}


void irtransaural_free(t_irtransaural *x)
{
    free_HIRT_common_attributes(x);
}


void irtransaural_assist(t_irtransaural *x, void *b, long m, long a, char *s)
{
    if (m == ASSIST_INLET)
        sprintf(s,"Instructions In");
    else
        sprintf(s,"Bang on Success");
}


//////////////////////////////////////////////////////////////////////////
///////////////////////////// User Messages //////////////////////////////
//////////////////////////////////////////////////////////////////////////


void irtransaural_process(t_irtransaural *x, t_symbol *sym, long argc, t_atom *argv)
{
    t_atom temp_argv[5];
    double time_mul = 1.0;

    // Load and check arguments

    if (argc < 4)
    {
        object_error((t_object *) x, "not enough arguments to message %s", sym->s_name);
        return;
    }

    if (argc > 4)
    {
        time_mul = atom_getfloat(argv + 4);

        if (time_mul < 1)
        {
            object_warn((t_object *) x, " time multiplier cannot be less than 1 (using 1)");
            time_mul = 1;
        }
    }

    temp_argv[0] = *argv++;
    temp_argv[1] = *argv++;
    temp_argv[2] = *argv++;
    temp_argv[3] = *argv++;
    atom_setfloat(temp_argv + 4, time_mul);

    defer(x, (method) irtransaural_process_internal, sym, 5, temp_argv);
}


void irtransaural_process_internal(t_irtransaural *x, t_symbol *sym, short argc, t_atom *argv)
{
    using complex = std::complex<double>;
    
    FFT_SPLIT_COMPLEX_D spectrum_1;
    FFT_SPLIT_COMPLEX_D spectrum_2;
    FFT_SPLIT_COMPLEX_D spectrum_3;
    FFT_SPLIT_COMPLEX_D spectrum_4;

    t_symbol *target_1 = atom_getsym(argv++);
    t_symbol *target_2 = atom_getsym(argv++);
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

    t_filter_type deconvolve_mode = static_cast<t_filter_type>(x->deconvolve_mode);
    t_atom_long read_chan = x->read_chan - 1;

    t_buffer_write_error error;
    bool overall_error = false;
    
    // Check input buffers

    if (buffer_check((t_object *) x, source_1) || buffer_check((t_object *) x, source_2))
        return;

    // Check and calculate length

    uintptr_t fft_size_log2;
    uintptr_t fft_size = calculate_fft_size(static_cast<uintptr_t>((source_length_1 + source_length_2) * time_mul), fft_size_log2);
    uintptr_t fft_size_halved = fft_size >> 1;
    deconvolve_delay = delay_retriever(x->deconvolve_delay, fft_size, sample_rate);

    if (fft_size < 8)
    {
        object_error((t_object *) x, "buffers are too short, or have no length");
        return;
    }

    // Allocate Memory

    temp_fft_setup fft_setup(fft_size_log2);

    temp_ptr<double> temp(fft_size * 5);
    temp_ptr<float> filter_in(filter_length);

    spectrum_1.realp = temp.get();
    spectrum_1.imagp = spectrum_1.realp + fft_size_halved;
    spectrum_2.realp = spectrum_1.imagp + fft_size_halved;
    spectrum_2.imagp = spectrum_2.realp + fft_size_halved;
    spectrum_3.realp = spectrum_2.imagp + fft_size_halved;
    spectrum_3.imagp = spectrum_3.realp + fft_size_halved;
    spectrum_4.realp = spectrum_3.imagp + fft_size_halved;
    spectrum_4.imagp = spectrum_4.realp + fft_size;

    double *out_buf = spectrum_4.realp;
    float *in_temp = reinterpret_cast<float *>(spectrum_4.realp);

    // Check memory allocations

    if (!fft_setup || !temp || (filter_length & !filter_in))
    {
        object_error((t_object *) x, "could not allocate temporary memory for processing");
        return;
    }

    // Fill deconvolution filter specifiers - read filter from buffer if specified

    fill_power_array_specifier(filter_specifier, x->deconvolve_filter_specifier, x->deconvolve_num_filter_specifiers);
    fill_power_array_specifier(range_specifier, x->deconvolve_range_specifier, x->deconvolve_num_range_specifiers);
    buffer_read(filter, 0, filter_in.get(), fft_size);

    // Get inputs - convert to frequency domain

    buffer_read(source_1, read_chan, in_temp, source_length_1);
    time_to_halfspectrum_float(fft_setup, in_temp, source_length_1, spectrum_1, fft_size);
    buffer_read(source_2, read_chan, in_temp, source_length_2);
    time_to_halfspectrum_float(fft_setup, in_temp, source_length_2, spectrum_2, fft_size);

    if (sym == gensym("shuffler"))
    {
        // Shuffle matrix

        for (uintptr_t i = 0; i < fft_size_halved; i++)
        {
            const double a = spectrum_1.realp[i];
            const double b = spectrum_1.imagp[i];
            const double c = spectrum_2.realp[i];
            const double d = spectrum_2.imagp[i];

            spectrum_2.realp[i] = (a + c);
            spectrum_2.imagp[i] = (b + d);
            spectrum_3.realp[i] = (a - c);
            spectrum_3.imagp[i] = (b - d);
        }

        // Deconvolve - convert to time domain - copy out to buffer

        spike_spectrum(spectrum_1, fft_size, SPECTRUM_REAL, deconvolve_delay);
        deconvolve(fft_setup, spectrum_1, spectrum_2, spectrum_4, filter_specifier, range_specifier, 0, filter_in.get(), filter_length, fft_size, SPECTRUM_REAL, deconvolve_mode, deconvolve_phase, 0, sample_rate);
        spectrum_to_time(fft_setup, out_buf, spectrum_1, fft_size, SPECTRUM_REAL);
        error = buffer_write((t_object *)x, target_1, out_buf, fft_size, x->write_chan - 1, x->resize, sample_rate, 1.0);
        if (error)
            overall_error = true;
        
        // Deconvolve - convert to time domain - copy out to buffer

        spike_spectrum(spectrum_1, fft_size, SPECTRUM_REAL, deconvolve_delay);
        deconvolve(fft_setup, spectrum_1, spectrum_3, spectrum_4, filter_specifier, range_specifier, 0, filter_in.get(), filter_length, fft_size, SPECTRUM_REAL, deconvolve_mode, deconvolve_phase, 0, sample_rate);
        spectrum_to_time(fft_setup, out_buf, spectrum_1, fft_size, SPECTRUM_REAL);
        error = buffer_write((t_object *)x, target_2, out_buf, fft_size, x->write_chan - 1, x->resize, sample_rate, 1.0);
        if (error)
            overall_error = true;
    }
    else
    {
        // Lattice - calculate divisor and invert the relevant channel

        const double a = spectrum_1.realp[0];
        const double b = spectrum_1.imagp[0];
        const double c = spectrum_2.realp[0];
        const double d = spectrum_2.imagp[0];

        spectrum_2.realp[0] = -c;
        spectrum_2.imagp[0] = -d;
        spectrum_3.realp[0] = a * a - c * c;
        spectrum_3.imagp[0] = b * b - d * d;

        for (uintptr_t i = 1; i < fft_size_halved; i++)
        {
            complex t1, t2, t3;

            const double a = spectrum_1.realp[i];
            const double b = spectrum_1.imagp[i];
            const double c = spectrum_2.realp[i];
            const double d = spectrum_2.imagp[i];

            t1 = complex(a, b);
            t2 = complex(c, d);
            t3 = (t1 * t1) - (t2 * t2);

            spectrum_2.realp[i] = -c;
            spectrum_2.imagp[i] = -d;
            spectrum_3.realp[i] = t3.real();
            spectrum_3.imagp[i] = t3.imag();
        }

        // Deconvolve - convert to time domain - copy out to buffer

        deconvolve(fft_setup, spectrum_1, spectrum_3, spectrum_4, filter_specifier, range_specifier, 0, filter_in.get(), filter_length, fft_size, SPECTRUM_REAL, deconvolve_mode, deconvolve_phase, deconvolve_delay, sample_rate);
        spectrum_to_time(fft_setup, out_buf, spectrum_1, fft_size, SPECTRUM_REAL);
        error = buffer_write((t_object *)x, target_1, out_buf, fft_size, x->write_chan - 1, x->resize, sample_rate, 1.0);
        if (error)
            overall_error = true;
        
        // Deconvolve - convert to time domain - copy out to buffer

        deconvolve(fft_setup, spectrum_2, spectrum_3, spectrum_4, filter_specifier, range_specifier, 0, filter_in.get(), filter_length, fft_size, SPECTRUM_REAL, deconvolve_mode, deconvolve_phase, deconvolve_delay, sample_rate);
        spectrum_to_time(fft_setup, out_buf, spectrum_2, fft_size, SPECTRUM_REAL);
        error = buffer_write((t_object *)x, target_2, out_buf, fft_size, x->write_chan - 1, x->resize, sample_rate, 1.0);
        if (error)
            overall_error = true;
    }

    if (!overall_error)
        outlet_bang(x->process_done);
}
