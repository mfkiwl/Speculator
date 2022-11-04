clc;
clear all;
%getting the large input x as a row vector from user
x=input("Enter the large sequence to be processed: ");
%default impulse response of the system to be assumed
h=input("Enter the impulse response of the system:");
%default length of the output convoluted blocks to be considered
N=input("Give a value for length of the block after circular convolution: ");
M=length(h);%length of impulse response sequence
L=N-M+1;%length of each block
%Padding process to impulse response
h=[h zeros(1,L-1)];%padded with L-1 zeros
%to normalise the array compared to the size of each block in order to
%perform ease computation
x1=[];
if(mod(length(x),L)~=0)
    x1=[x zeros(1,(L-mod(length(x),L)))];
else
    x1=x;
end    
%using looping structure the long sequnece is split into smaller array of size and padded with M-1 zeros
a={};%a cell is used to store the padded input sequence arrays with length L+M-1
temp=[];
for i=1:length(x1)/L
        temp=x1(1:L);
        temp=[temp,zeros(1,M-1)];
        a{end+1}=temp;
        x1(1:L)=[];
end     
[r,c]=size(a);
for i=1:c
    a{i};
end
%cell consisting of cicular convoluted outputs of indvidual blocks in x[n]
 y={};
 for i=1:c
     y{end+1}=cconv(a{i},h,N);
 end
 [r1,c1]=size(y);
 for i=1:c1
     y{i}
 end
temp1=[];
intemp=[];
temp3=[];
addtemp=[];
for i=1:c1-1
    if(i==1)
        temp1=y{i};
        intemp=temp1(M+1:length(temp));
        temp1=temp1(1:M);
        j=i+1;
        temp3=y{j};
        temp3=temp3(1:M-1);
        addtemp=intemp+temp3;
        temp1=[temp1 addtemp];
        y{i}=temp1;
    else
        temp1=y{i};
        intemp=temp1(M+1:length(temp));
        temp1=temp1(M);
        j=i+1;
        temp3=y{j};
        temp3=temp3(1:M-1);
        addtemp=intemp+temp3;
        temp1=[temp1 addtemp];
        y{i}=temp1;
    end   
end
temp1=y{c1};
temp1=temp1(M:length(temp1));
y{c1}=temp1;
sample=cell2mat(y);
output="Output sequence";
disp(output)
disp(sample)
