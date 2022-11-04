swig -lua -c++ pffft.i
gcc -O2 -fPIC -march=native -mavx2 -shared -o pffft_blustein.so pffft_wrap.cxx pffft.c -lstdc++ -lm -lluajit
