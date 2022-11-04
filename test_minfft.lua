require('minfft')
require('sndfile')
s = sndfile.SndFileReaderFloat("baby_elephant.wav")
v = sndfile.float_vector(s:size())
s:read(v:size(),v:data())
m = minfft.float_vector(1024)
c = minfft.float_vector(1024*4)
aux = minfft.minfft_mkaux_realdft_1d(1024)
for i=1,v:size()/1024 do        
    for j=1,1024 do 
        m[j] = v[i*1024+j]
    end
    minfft.minfft_realdft(m:data(),c:data(),aux)
    for j=1,c:size() do
        c[j] = c[j] / 1024       
    end
    print()
end
minfft.minfft_free_aux(aux)