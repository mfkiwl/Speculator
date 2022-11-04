swig -lua -c++ resample.i
gcc -O2 -fPIC -march=native -mavx2 -shared -o resample.so resample_wrap.cxx resample.cpp -lstdc++ -lm -lluajit
