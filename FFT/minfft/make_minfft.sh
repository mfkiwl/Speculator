swig -lua -c++ minfft.i
gcc -std=c++17 -I. -O2 -fPIC -march=native -mavx2 -shared -o minfft.so minfft_wrap.cxx minfft.c -lstdc++ -lm -lluajit
