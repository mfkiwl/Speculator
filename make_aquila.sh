swig -lua -c++ -IAquila/aquila -Isrc  aquila.i
gcc -IAquila/aquila -O2 -fPIC -march=native -mavx2 -shared -o aquila.so aquila_wrap.cxx Aquila/aquila/lib/ooura/fft4g.c lib/libAquila.a -lstdc++ -lm -lluajit
