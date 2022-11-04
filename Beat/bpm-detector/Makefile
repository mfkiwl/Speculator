default: sequential 

sequential:
	mkdir -p bin/
	gcc   bpm_detector.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -o bin/bpm-detector  -lm 
 
clean:
	rm -r bin
	rm error* output*