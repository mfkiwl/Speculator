require('mkl_array')
a = mkl_array.float_array(10)
for i=1,10 do a[i] = i*i end
a:print()
for i=1,10 do io.write(a[i],",") end
print()
for i=1,10 do a[i] = i end
a:print()
