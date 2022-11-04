require('se')
a = se.FloatMatrix(3,3)
for i=0,2 do
for j=0,2 do
a[i][j]=i*3+j
end
end
a:print()
se.sigmoid_float(a)
a:print()
se.sigmoid_deriv_float(a)
a:print()
