% ͨ����������sin��x) cos(x)
% xȡֵΪPI/2 PI/4 PI/8 PI/16 ... PI/(2^k)
% �˳����������ɱ�����ʹ�ò����ɵ����

clc;
clear

k = 0:20;
x = pi./(2.^k);
y1 = sin(x);
y2 = cos(x);
plot(x,y1,x,y2),grid on

% ע�����ڱ��е�ֵ�ɲ������Բ�ֵ�ķ�����ȡ������߾���
fid = fopen('sin_tb.h', 'w');
fprintf(fid, 'const float sin_tb[] = {  // ����(PI PI/2 PI/4 PI/8 PI/16 ... PI/(2^k))\n');

fprintf(fid, '%.6f, %.6f, %.6f, %.6f, %.6f, %.6f , %.6f, %.6f, %.6f, %.6f,\n', y1);
fprintf(fid, '\n};\n');
fclose(fid);

% ע�����ڱ��е�ֵ�ɲ������Բ�ֵ�ķ�����ȡ������߾���
fid = fopen('cos_tb.h', 'w');
fprintf(fid, 'const float cos_tb[] = {  // ����(PI PI/2 PI/4 PI/8 PI/16 ... PI/(2^k))\n');

fprintf(fid, '%.6f, %.6f, %.6f, %.6f, %.6f, %.6f , %.6f, %.6f, %.6f, %.6f,\n', y2);
fprintf(fid, '\n};\n');
fclose(fid);
