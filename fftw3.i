%module fftw3
%{
#include <fftw3.h>
#include "fftw.hpp"
%}

%include "stdint.i"
%include "std_vector.i"
%include "std_math.i"

typedef double fftw_complex[2]; 
typedef struct fftw_plan_s *fftw_plan; 
typedef struct fftw_iodim_do_not_use_me fftw_iodim; 
typedef struct fftw_iodim64_do_not_use_me fftw_iodim64; 
typedef enum fftw_r2r_kind_do_not_use_me fftw_r2r_kind; 
typedef fftw_write_char_func_do_not_use_me fftw_write_char_func; 
typedef fftw_read_char_func_do_not_use_me fftw_read_char_func; 

void fftw_execute(const fftw_plan p); 
fftw_plan fftw_plan_dft(int rank, const int *n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags); 
fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
fftw_plan fftw_plan_guru_split_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags); 
fftw_plan fftw_plan_guru64_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
fftw_plan fftw_plan_guru64_split_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags); 
void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out); 
void fftw_execute_split_dft(const fftw_plan p, double *ri, double *ii, double *ro, double *io); 
fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
fftw_plan fftw_plan_dft_r2c(int rank, const int *n, double *in, fftw_complex *out, unsigned flags); fftw_plan fftw_plan_dft_r2c_1d(int n,double *in,fftw_complex *out,unsigned flags); 
fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out, unsigned flags); 
fftw_plan fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double *in, fftw_complex *out, unsigned flags); 
fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, unsigned flags); 
fftw_plan fftw_plan_dft_c2r(int rank, const int *n, fftw_complex *in, double *out, unsigned flags); 
fftw_plan fftw_plan_dft_c2r_1d(int n,fftw_complex *in,double *out,unsigned flags); 
fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex *in, double *out, unsigned flags); 
fftw_plan fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex *in, double *out, unsigned flags); 
fftw_plan fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, fftw_complex *out, unsigned flags); 
fftw_plan fftw_plan_guru_dft_c2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, double *out, unsigned flags); 
fftw_plan fftw_plan_guru_split_dft_r2c(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *ro, double *io, unsigned flags); 
fftw_plan fftw_plan_guru_split_dft_c2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *out, unsigned flags); 
fftw_plan fftw_plan_guru64_dft_r2c(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, fftw_complex *out, unsigned flags); 
fftw_plan fftw_plan_guru64_dft_c2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, double *out, unsigned flags); 
fftw_plan fftw_plan_guru64_split_dft_r2c(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *ro, double *io, unsigned flags); 
fftw_plan fftw_plan_guru64_split_dft_c2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *out, unsigned flags); 
void fftw_execute_dft_r2c(const fftw_plan p, double *in, fftw_complex *out); void fftw_execute_dft_c2r(const fftw_plan p, fftw_complex *in, double *out); 
void fftw_execute_split_dft_r2c(const fftw_plan p, double *in, double *ro, double *io); 
void fftw_execute_split_dft_c2r(const fftw_plan p, double *ri, double *ii, double *out);
fftw_plan fftw_plan_many_r2r(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, const fftw_r2r_kind *kind, unsigned flags); 
fftw_plan fftw_plan_r2r(int rank, const int *n, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags); 
fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out, fftw_r2r_kind kind, unsigned flags); 
fftw_plan fftw_plan_r2r_2d(int n0, int n1, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags); 
fftw_plan fftw_plan_r2r_3d(int n0, int n1, int n2, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags); 
fftw_plan fftw_plan_guru_r2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags); 
fftw_plan fftw_plan_guru64_r2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags); 
void fftw_execute_r2r(const fftw_plan p, double *in, double *out); void fftw_destroy_plan(fftw_plan p); 
void fftw_forget_wisdom(void); 
void fftw_cleanup(void); 
void fftw_set_timelimit(double t); 
void fftw_plan_with_nthreads(int nthreads); 
int fftw_init_threads(void); 
void fftw_cleanup_threads(void); 
void fftw_make_planner_thread_safe(void); 
int fftw_export_wisdom_to_filename(const char *filename); 
void fftw_export_wisdom_to_file(FILE *output_file); char * fftw_export_wisdom_to_string(void); 
void fftw_export_wisdom(fftw_write_char_func write_char, void *data); int fftw_import_system_wisdom(void); 
int fftw_import_wisdom_from_filename(const char *filename); int fftw_import_wisdom_from_file(FILE *input_file); 
int fftw_import_wisdom_from_string(const char *input_string); int fftw_import_wisdom(fftw_read_char_func read_char, void *data); 
void fftw_fprint_plan(const fftw_plan p, FILE *output_file); void fftw_print_plan(const fftw_plan p); 
char * fftw_sprint_plan(const fftw_plan p); void * fftw_malloc(size_t n); double * fftw_alloc_real(size_t n); 
fftw_complex * fftw_alloc_complex(size_t n); void fftw_free(void *p); 
void fftw_flops(const fftw_plan p, double *add, double *mul, double *fmas); 
double fftw_estimate_cost(const fftw_plan p); 
double fftw_cost(const fftw_plan p); int fftw_alignment_of(double *p); 
const char fftw_version[]; 
const char fftw_cc[]; 
const char fftw_codelet_optim[];

typedef float fftwf_complex[2]; 
typedef struct fftwf_plan_s *fftwf_plan; 
typedef struct fftw_iodim_do_not_use_me fftwf_iodim; 
typedef struct fftw_iodim64_do_not_use_me fftwf_iodim64; 
typedef enum fftw_r2r_kind_do_not_use_me fftwf_r2r_kind; 
typedef fftw_write_char_func_do_not_use_me fftwf_write_char_func; 
typedef fftw_read_char_func_do_not_use_me fftwf_read_char_func; 

void fftwf_execute(const fftwf_plan p); 
fftwf_plan fftwf_plan_dft(int rank, const int *n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
fftwf_plan fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags); 
fftwf_plan fftwf_plan_guru_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
fftwf_plan fftwf_plan_guru_split_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags); 
fftwf_plan fftwf_plan_guru64_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
fftwf_plan fftwf_plan_guru64_split_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags); 
void fftwf_execute_dft(const fftwf_plan p, fftwf_complex *in, fftwf_complex *out); 
void fftwf_execute_split_dft(const fftwf_plan p, float *ri, float *ii, float *ro, float *io); 
fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
fftwf_plan fftwf_plan_dft_r2c(int rank, const int *n, float *in, fftwf_complex *out, unsigned flags); 
fftwf_plan fftwf_plan_dft_r2c_1d(int n,float *in,fftwf_complex *out,unsigned flags); 
fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, float *in, fftwf_complex *out, unsigned flags); 
fftwf_plan fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float *in, fftwf_complex *out, unsigned flags); 
fftwf_plan fftwf_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, unsigned flags); 
fftwf_plan fftwf_plan_dft_c2r(int rank, const int *n, fftwf_complex *in, float *out, unsigned flags); 
fftwf_plan fftwf_plan_dft_c2r_1d(int n,fftwf_complex *in,float *out,unsigned flags); 
fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex *in, float *out, unsigned flags); 
fftwf_plan fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex *in, float *out, unsigned flags); 
fftwf_plan fftwf_plan_guru_dft_r2c(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, fftwf_complex *out, unsigned flags); 
fftwf_plan fftwf_plan_guru_dft_c2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, float *out, unsigned flags); 
fftwf_plan fftwf_plan_guru_split_dft_r2c(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *ro, float *io, unsigned flags); 
fftwf_plan fftwf_plan_guru_split_dft_c2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *out, unsigned flags); 
fftwf_plan fftwf_plan_guru64_dft_r2c(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, fftwf_complex *out, unsigned flags); 
fftwf_plan fftwf_plan_guru64_dft_c2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru64_split_dft_r2c(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *ro, float *io, unsigned flags); 
fftwf_plan fftwf_plan_guru64_split_dft_c2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *out, unsigned flags); 
void fftwf_execute_dft_r2c(const fftwf_plan p, float *in, fftwf_complex *out); 
void fftwf_execute_dft_c2r(const fftwf_plan p, fftwf_complex *in, float *out); 
void fftwf_execute_split_dft_r2c(const fftwf_plan p, float *in, float *ro, float *io); 
void fftwf_execute_split_dft_c2r(const fftwf_plan p, float *ri, float *ii, float *out); 
fftwf_plan fftwf_plan_many_r2r(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, const fftwf_r2r_kind *kind, unsigned flags); 
fftwf_plan fftwf_plan_r2r(int rank, const int *n, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags); 
fftwf_plan fftwf_plan_r2r_1d(int n, float *in, float *out, fftwf_r2r_kind kind, unsigned flags); 
fftwf_plan fftwf_plan_r2r_2d(int n0, int n1, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, unsigned flags); 
fftwf_plan fftwf_plan_r2r_3d(int n0, int n1, int n2, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, fftwf_r2r_kind kind2, unsigned flags); 
fftwf_plan fftwf_plan_guru_r2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags); 
fftwf_plan fftwf_plan_guru64_r2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags); 
void fftwf_execute_r2r(const fftwf_plan p, float *in, float *out); void fftwf_destroy_plan(fftwf_plan p); 
void fftwf_forget_wisdom(void); void fftwf_cleanup(void); void fftwf_set_timelimit(double t); 
void fftwf_plan_with_nthreads(int nthreads); int fftwf_init_threads(void); 
void fftwf_cleanup_threads(void); void fftwf_make_planner_thread_safe(void); 
int fftwf_export_wisdom_to_filename(const char *filename); void fftwf_export_wisdom_to_file(FILE *output_file); 
char * fftwf_export_wisdom_to_string(void); 
void fftwf_export_wisdom(fftwf_write_char_func write_char, void *data); 
int fftwf_import_system_wisdom(void); 
int fftwf_import_wisdom_from_filename(const char *filename); 
int fftwf_import_wisdom_from_file(FILE *input_file); 
int fftwf_import_wisdom_from_string(const char *input_string); 
int fftwf_import_wisdom(fftwf_read_char_func read_char, void *data); 
void fftwf_fprint_plan(const fftwf_plan p, FILE *output_file); 
void fftwf_print_plan(const fftwf_plan p); 
char * fftwf_sprint_plan(const fftwf_plan p); 
void * fftwf_malloc(size_t n); 
float * fftwf_alloc_real(size_t n); 
fftwf_complex * fftwf_alloc_complex(size_t n); void fftwf_free(void *p); 
void fftwf_flops(const fftwf_plan p, double *add, double *mul, double *fmas); 
double fftwf_estimate_cost(const fftwf_plan p); 
double fftwf_cost(const fftwf_plan p); 
int fftwf_alignment_of(float *p); 
const char fftwf_version[]; 
const char fftwf_cc[]; 
const char fftwf_codelet_optim[];

typedef long double fftwl_complex[2]; 
typedef struct fftwl_plan_s *fftwl_plan; 
typedef struct fftw_iodim_do_not_use_me fftwl_iodim; 
typedef struct fftw_iodim64_do_not_use_me fftwl_iodim64; 
typedef enum fftw_r2r_kind_do_not_use_me fftwl_r2r_kind; 
typedef fftw_write_char_func_do_not_use_me fftwl_write_char_func; 
typedef fftw_read_char_func_do_not_use_me fftwl_read_char_func; 
void fftwl_execute(const fftwl_plan p); 
fftwl_plan fftwl_plan_dft(int rank, const int *n, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
fftwl_plan fftwl_plan_dft_1d(int n, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
fftwl_plan fftwl_plan_dft_2d(int n0, int n1, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
fftwl_plan fftwl_plan_dft_3d(int n0, int n1, int n2, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
fftwl_plan fftwl_plan_many_dft(int rank, const int *n, int howmany, fftwl_complex *in, const int *inembed, int istride, int idist, fftwl_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags); 
fftwl_plan fftwl_plan_guru_dft(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
fftwl_plan fftwl_plan_guru_split_dft(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *ri, long double *ii, long double *ro, long double *io, unsigned flags); 
fftwl_plan fftwl_plan_guru64_dft(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
fftwl_plan fftwl_plan_guru64_split_dft(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *ri, long double *ii, long double *ro, long double *io, unsigned flags); 
void fftwl_execute_dft(const fftwl_plan p, fftwl_complex *in, fftwl_complex *out); 
void fftwl_execute_split_dft(const fftwl_plan p, long double *ri, long double *ii, long double *ro, long double *io); 
fftwl_plan fftwl_plan_many_dft_r2c(int rank, const int *n, int howmany, long double *in, const int *inembed, int istride, int idist, fftwl_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
fftwl_plan fftwl_plan_dft_r2c(int rank, const int *n, long double *in, fftwl_complex *out, unsigned flags);
fftwl_plan fftwl_plan_dft_r2c_1d(int n,long double *in,fftwl_complex *out,unsigned flags); 
fftwl_plan fftwl_plan_dft_r2c_2d(int n0, int n1, long double *in, fftwl_complex *out, unsigned flags); 
fftwl_plan fftwl_plan_dft_r2c_3d(int n0, int n1, int n2, long double *in, fftwl_complex *out, unsigned flags); 
 fftwl_plan fftwl_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwl_complex *in, const int *inembed, int istride, int idist, long double *out, const int *onembed, int ostride, int odist, unsigned flags);
 fftwl_plan fftwl_plan_dft_c2r(int rank, const int *n, fftwl_complex *in, long double *out, unsigned flags);
 fftwl_plan fftwl_plan_dft_c2r_1d(int n,fftwl_complex *in,long double *out,unsigned flags);
 fftwl_plan fftwl_plan_dft_c2r_2d(int n0, int n1, fftwl_complex *in, long double *out, unsigned flags);
 fftwl_plan fftwl_plan_dft_c2r_3d(int n0, int n1, int n2, fftwl_complex *in, long double *out, unsigned flags);
 fftwl_plan fftwl_plan_guru_dft_r2c(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *in, fftwl_complex *out, unsigned flags);
 fftwl_plan fftwl_plan_guru_dft_c2r(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, fftwl_complex *in, long double *out, unsigned flags);
 fftwl_plan fftwl_plan_guru_split_dft_r2c(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *in, long double *ro, long double *io, unsigned flags);
 fftwl_plan fftwl_plan_guru_split_dft_c2r(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *ri, long double *ii, long double *out, unsigned flags);
 fftwl_plan fftwl_plan_guru64_dft_r2c(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *in, fftwl_complex *out, unsigned flags);
 fftwl_plan fftwl_plan_guru64_dft_c2r(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, fftwl_complex *in, long double *out, unsigned flags);
 fftwl_plan fftwl_plan_guru64_split_dft_r2c(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *in, long double *ro, long double *io, unsigned flags);
 fftwl_plan fftwl_plan_guru64_split_dft_c2r(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *ri, long double *ii, long double *out, unsigned flags);
 void fftwl_execute_dft_r2c(const fftwl_plan p, long double *in, fftwl_complex *out);
 void fftwl_execute_dft_c2r(const fftwl_plan p, fftwl_complex *in, long double *out);
 void fftwl_execute_split_dft_r2c(const fftwl_plan p, long double *in, long double *ro, long double *io);
 void fftwl_execute_split_dft_c2r(const fftwl_plan p, long double *ri, long double *ii, long double *out);
 fftwl_plan fftwl_plan_many_r2r(int rank, const int *n, int howmany, long double *in, const int *inembed, int istride, int idist, long double *out, const int *onembed, int ostride, int odist, const fftwl_r2r_kind *kind, unsigned flags);
 fftwl_plan fftwl_plan_r2r(int rank, const int *n, long double *in, long double *out, const fftwl_r2r_kind *kind, unsigned flags);
 fftwl_plan fftwl_plan_r2r_1d(int n, long double *in, long double *out, fftwl_r2r_kind kind, unsigned flags);
 fftwl_plan fftwl_plan_r2r_2d(int n0, int n1, long double *in, long double *out, fftwl_r2r_kind kind0, fftwl_r2r_kind kind1, unsigned flags);
 fftwl_plan fftwl_plan_r2r_3d(int n0, int n1, int n2, long double *in, long double *out, fftwl_r2r_kind kind0, fftwl_r2r_kind kind1, fftwl_r2r_kind kind2, unsigned flags);
 fftwl_plan fftwl_plan_guru_r2r(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *in, long double *out, const fftwl_r2r_kind *kind, unsigned flags);
 fftwl_plan fftwl_plan_guru64_r2r(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *in, long double *out, const fftwl_r2r_kind *kind, unsigned flags);
 void fftwl_execute_r2r(const fftwl_plan p, long double *in, long double *out);
 void fftwl_destroy_plan(fftwl_plan p);
 void fftwl_forget_wisdom(void);
 void fftwl_cleanup(void);
 void fftwl_set_timelimit(double t);
 void fftwl_plan_with_nthreads(int nthreads);
 int fftwl_init_threads(void);
 void fftwl_cleanup_threads(void);
 void fftwl_make_planner_thread_safe(void);
 int fftwl_export_wisdom_to_filename(const char *filename);
 void fftwl_export_wisdom_to_file(FILE *output_file);
 char * fftwl_export_wisdom_to_string(void);
 void fftwl_export_wisdom(fftwl_write_char_func write_char, void *data);
 int fftwl_import_system_wisdom(void);
 int fftwl_import_wisdom_from_filename(const char *filename);
 int fftwl_import_wisdom_from_file(FILE *input_file);
 int fftwl_import_wisdom_from_string(const char *input_string);
 int fftwl_import_wisdom(fftwl_read_char_func read_char, void *data);
 void fftwl_fprint_plan(const fftwl_plan p, FILE *output_file);
 void fftwl_print_plan(const fftwl_plan p);
 char * fftwl_sprint_plan(const fftwl_plan p);
 void * fftwl_malloc(size_t n);
 long double * fftwl_alloc_real(size_t n);
 fftwl_complex * fftwl_alloc_complex(size_t n);
 void fftwl_free(void *p);
 void fftwl_flops(const fftwl_plan p, double *add, double *mul, double *fmas);
 double fftwl_estimate_cost(const fftwl_plan p);
 double fftwl_cost(const fftwl_plan p);
 int fftwl_alignment_of(long double *p);
 const char fftwl_version[];
 const char fftwl_cc[];
 const char fftwl_codelet_optim[];


 typedef __float128 fftwq_complex[2];
 typedef struct fftwq_plan_s *fftwq_plan;
 typedef struct fftw_iodim_do_not_use_me fftwq_iodim;
 typedef struct fftw_iodim64_do_not_use_me fftwq_iodim64;
 typedef enum fftw_r2r_kind_do_not_use_me fftwq_r2r_kind;
 typedef fftw_write_char_func_do_not_use_me fftwq_write_char_func;
 typedef fftw_read_char_func_do_not_use_me fftwq_read_char_func;
 void fftwq_execute(const fftwq_plan p);
 fftwq_plan fftwq_plan_dft(int rank, const int *n, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags);
 fftwq_plan fftwq_plan_dft_1d(int n, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags);
 fftwq_plan fftwq_plan_dft_2d(int n0, int n1, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags);
 fftwq_plan fftwq_plan_dft_3d(int n0, int n1, int n2, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags);
 fftwq_plan fftwq_plan_many_dft(int rank, const int *n, int howmany, fftwq_complex *in, const int *inembed, int istride, int idist, fftwq_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags);
 fftwq_plan fftwq_plan_guru_dft(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags);
 fftwq_plan fftwq_plan_guru_split_dft(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *ri, __float128 *ii, __float128 *ro, __float128 *io, unsigned flags);
 fftwq_plan fftwq_plan_guru64_dft(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags);
 fftwq_plan fftwq_plan_guru64_split_dft(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *ri, __float128 *ii, __float128 *ro, __float128 *io, unsigned flags);
 void fftwq_execute_dft(const fftwq_plan p, fftwq_complex *in, fftwq_complex *out);
 void fftwq_execute_split_dft(const fftwq_plan p, __float128 *ri, __float128 *ii, __float128 *ro, __float128 *io);
 fftwq_plan fftwq_plan_many_dft_r2c(int rank, const int *n, int howmany, __float128 *in, const int *inembed, int istride, int idist, fftwq_complex *out, const int *onembed, int ostride, int odist, unsigned flags);
 fftwq_plan fftwq_plan_dft_r2c(int rank, const int *n, __float128 *in, fftwq_complex *out, unsigned flags);
 fftwq_plan fftwq_plan_dft_r2c_1d(int n,__float128 *in,fftwq_complex *out,unsigned flags);
 fftwq_plan fftwq_plan_dft_r2c_2d(int n0, int n1, __float128 *in, fftwq_complex *out, unsigned flags);
 fftwq_plan fftwq_plan_dft_r2c_3d(int n0, int n1, int n2, __float128 *in, fftwq_complex *out, unsigned flags);
 fftwq_plan fftwq_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwq_complex *in, const int *inembed, int istride, int idist, __float128 *out, const int *onembed, int ostride, int odist, unsigned flags);
 fftwq_plan fftwq_plan_dft_c2r(int rank, const int *n, fftwq_complex *in, __float128 *out, unsigned flags);
 fftwq_plan fftwq_plan_dft_c2r_1d(int n,fftwq_complex *in,__float128 *out,unsigned flags);
 fftwq_plan fftwq_plan_dft_c2r_2d(int n0, int n1, fftwq_complex *in, __float128 *out, unsigned flags);
 fftwq_plan fftwq_plan_dft_c2r_3d(int n0, int n1, int n2, fftwq_complex *in, __float128 *out, unsigned flags);
 fftwq_plan fftwq_plan_guru_dft_r2c(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *in, fftwq_complex *out, unsigned flags);
 fftwq_plan fftwq_plan_guru_dft_c2r(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, fftwq_complex *in, __float128 *out, unsigned flags);
 fftwq_plan fftwq_plan_guru_split_dft_r2c(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *in, __float128 *ro, __float128 *io, unsigned flags);
 fftwq_plan fftwq_plan_guru_split_dft_c2r(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *ri, __float128 *ii, __float128 *out, unsigned flags);
 fftwq_plan fftwq_plan_guru64_dft_r2c(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *in, fftwq_complex *out, unsigned flags);
 fftwq_plan fftwq_plan_guru64_dft_c2r(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, fftwq_complex *in, __float128 *out, unsigned flags);
 fftwq_plan fftwq_plan_guru64_split_dft_r2c(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *in, __float128 *ro, __float128 *io, unsigned flags);
 fftwq_plan fftwq_plan_guru64_split_dft_c2r(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *ri, __float128 *ii, __float128 *out, unsigned flags);
 void fftwq_execute_dft_r2c(const fftwq_plan p, __float128 *in, fftwq_complex *out);
 void fftwq_execute_dft_c2r(const fftwq_plan p, fftwq_complex *in, __float128 *out);
 void fftwq_execute_split_dft_r2c(const fftwq_plan p, __float128 *in, __float128 *ro, __float128 *io);
 void fftwq_execute_split_dft_c2r(const fftwq_plan p, __float128 *ri, __float128 *ii, __float128 *out);
 fftwq_plan fftwq_plan_many_r2r(int rank, const int *n, int howmany, __float128 *in, const int *inembed, int istride, int idist, __float128 *out, const int *onembed, int ostride, int odist, const fftwq_r2r_kind *kind, unsigned flags);
 fftwq_plan fftwq_plan_r2r(int rank, const int *n, __float128 *in, __float128 *out, const fftwq_r2r_kind *kind, unsigned flags);
 fftwq_plan fftwq_plan_r2r_1d(int n, __float128 *in, __float128 *out, fftwq_r2r_kind kind, unsigned flags);
 fftwq_plan fftwq_plan_r2r_2d(int n0, int n1, __float128 *in, __float128 *out, fftwq_r2r_kind kind0, fftwq_r2r_kind kind1, unsigned flags);
 fftwq_plan fftwq_plan_r2r_3d(int n0, int n1, int n2, __float128 *in, __float128 *out, fftwq_r2r_kind kind0, fftwq_r2r_kind kind1, fftwq_r2r_kind kind2, unsigned flags);
 fftwq_plan fftwq_plan_guru_r2r(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *in, __float128 *out, const fftwq_r2r_kind *kind, unsigned flags);
 fftwq_plan fftwq_plan_guru64_r2r(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *in, __float128 *out, const fftwq_r2r_kind *kind, unsigned flags);
 void fftwq_execute_r2r(const fftwq_plan p, __float128 *in, __float128 *out);
 void fftwq_destroy_plan(fftwq_plan p);
 void fftwq_forget_wisdom(void);
 void fftwq_cleanup(void);
 void fftwq_set_timelimit(double t);
 void fftwq_plan_with_nthreads(int nthreads);
 int fftwq_init_threads(void);
 void fftwq_cleanup_threads(void);
 void fftwq_make_planner_thread_safe(void);
 int fftwq_export_wisdom_to_filename(const char *filename);
 void fftwq_export_wisdom_to_file(FILE *output_file);
 char * fftwq_export_wisdom_to_string(void);
 void fftwq_export_wisdom(fftwq_write_char_func write_char, void *data);
 int fftwq_import_system_wisdom(void);
 int fftwq_import_wisdom_from_filename(const char *filename);
 int fftwq_import_wisdom_from_file(FILE *input_file);
 int fftwq_import_wisdom_from_string(const char *input_string);
 int fftwq_import_wisdom(fftwq_read_char_func read_char, void *data);
 void fftwq_fprint_plan(const fftwq_plan p, FILE *output_file);
 void fftwq_print_plan(const fftwq_plan p);
 char * fftwq_sprint_plan(const fftwq_plan p);
 void * fftwq_malloc(size_t n);
 __float128 * fftwq_alloc_real(size_t n);
 fftwq_complex * fftwq_alloc_complex(size_t n);
 void fftwq_free(void *p);
 void fftwq_flops(const fftwq_plan p, double *add, double *mul, double *fmas);
 double fftwq_estimate_cost(const fftwq_plan p);
 double fftwq_cost(const fftwq_plan p);
 int fftwq_alignment_of(__float128 *p);
 const char fftwq_version[];
 const char fftwq_cc[];
 const char fftwq_codelet_optim[];

%include "fftw.hpp"

%extend FloatVector {
        float __getitem__(size_t i) { return $self->vector[i]; }
        void __setitem(size_t i, float v) { $self->vector[i] = v;}
    }    
%extend ComplexFloatVector {
        std::complex<float> __getitem__(size_t i) { return std::complex<float>($self->vector[i][0],$self->vector[i][1]); }
        void __setitem(size_t i, std::complex<float> & v) { $self->vector[i][0] = v.real();$self->vector[i][1] = v.imag();}
    }    
 %extend DoubleVector {
        double __getitem__(size_t i) { return $self->vector[i]; }
        void __setitem(size_t i, double v) { $self->vector[i] = v;}
    }
%extend ComplexDoubleVector {
        std::complex<double> __getitem__(size_t i) { return std::complex<double>($self->vector[i][0],$self->vector[i][1]); }
        void __setitem(size_t i, std::complex<double> & v) { $self->vector[i][0] = v.real();$self->vector[i][1] = v.imag();}
    }    
/*    
%extend Window<float> {
    FloatVector __mul__(FloatVector & v) {
        FloatVector r(v.size());
        for(size_t i = 0; i < v.ssize(); i++) r.vector[i] = $self->vector[i] * v.vector[i];
        return r;
    }
} 
*/   
%template(WindowFloat) Window<float>;
%template(RectangleWindowFloat) Rectangle<float>;
%template(HammingWindowFloat) Hamming<float>;
%template(HanningWindowFloat) Hanning<float>;
%template(BlackmanWindowFloat) Blackman<float>;
%template(BlackmanHarrisWindowFloat) BlackmanHarris<float>;
%template(GaussianWindowFloat) Gaussian<float>;
%template(WelchWindowFloat) Welch<float>;
%template(ParzenWindowFloat) Parzen<float>;
%template(TukeyWindowFloat) Tukey<float>;
