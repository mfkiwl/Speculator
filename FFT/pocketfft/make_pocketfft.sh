swig -lua -c++ pocketfft.i
gcc -O2 -fPIC -march=native -mavx2 -shared -o pocketfft.so pocketfft_wrap.cxx pocketfft.c -lstdc++ -lm -lluajit
