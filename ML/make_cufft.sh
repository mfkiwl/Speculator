clear
rm cufft_wrap.cu
swig -lua -c++ -I/usr/local/cuda/include cufft.i 
mv cufft_wrap.cxx cufft_wrap.cu
rm cufft_wrap.cxx 
nvcc --ptxas-options=-v --compiler-options '-fPIC -fmax-errors=1' -o cufft.so --shared cufft_wrap.cu -lstdc++ -lluajit-5.1 -L/usr/local/cuda/lib64 -lcublas -lcudart_static -lcufft
