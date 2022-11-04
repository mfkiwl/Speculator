swig -lua -c++ pffft.i
gcc -O2 -fPIC -march=native -mavx2 -shared -o pffft.so pffft_wrap.cxx build/libpffft.a -lstdc++ -lm -lluajit
