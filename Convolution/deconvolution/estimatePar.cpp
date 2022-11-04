#include "estimatepar.h"

/*****************************************************************
*函数名：directionDiffAngle
*功能：  方向微分法估计运动模糊角度,微元半径为2，3X3微分算子
*****************************************************************/
double  directionDiffAngle(Mat &input)
{
	//3X3算子
	Mat mask = Mat_<double>(3, 3);
	double angle = 0, step = 1;
	double n = 0;

	double s0 = 255;
	double sx = 0, sy = 0, temp;
	int i, j;
	int row = input.rows;
	int col = input.cols;

	for (angle = -90; angle <= 90; angle = angle + step)
	{
		sx = 0;
		sectionMask(angle, mask);

		Scalar ss = sum(mask);

		for (j = 2; j < row - 2; j = j + 4)
		{
			sy = 0;
			uchar* previous1 = input.ptr<uchar>(j - 1);
			uchar* previous2 = input.ptr<uchar>(j - 2);
			uchar* current = input.ptr<uchar>(j);
			uchar* next1 = input.ptr<uchar>(j + 1);
			uchar* next2 = input.ptr<uchar>(j + 2);
			for (i = 0; i < col - 2; i = i + 2)
			{
				if (angle >= -90.0&&angle < 0.0)
				{
					temp = (mask.at<double>(0)*previous2[i] + mask.at<double>(1)*previous2[i + 1] + mask.at<double>(2)*previous2[i + 2] +
						mask.at<double>(3)*previous1[i] + mask.at<double>(4)*previous1[i + 1] + mask.at<double>(5)*previous1[i + 2] +
						mask.at<double>(6)*current[i] + mask.at<double>(7)*current[i + 1] + mask.at<double>(8)*current[i + 2]);
					if (temp>255)
						temp = 255;
					temp = temp / (col - 2);
				}
				else
				{
					temp = (mask.at<double>(0)*current[i] + mask.at<double>(1)*current[i + 1] + mask.at<double>(2)*current[i + 2] +
						mask.at<double>(3)*next1[i] + mask.at<double>(4)*next1[i + 1] + mask.at<double>(5)*next1[i + 2] +
						mask.at<double>(6)*next2[i] + mask.at<double>(7)*next2[i + 1] + mask.at<double>(8)*next2[i + 2]);
					if (temp > 255)
						temp = 255;
					temp = temp / (col - 2);
				}
				sy = sy + temp;
			}
			sx += sy / (row - 4);
		}

		if (sx < s0)
		{
			s0 = sx;
			n = angle;
		}
	}
	angle = -angle;
	if (angle < 0)
		angle = 180 + angle;

	return angle;
}


/*****************************************************************
*函数名：directionDiffAngle2
*功能：  方向微分法估计运动模糊角度,微元半径为2，2X2微分算子
*****************************************************************/
double  directionDiffAngle2(Mat &input)
{
	Mat Temp = Mat_<double>(input);
	double min = 5000000,a=0,n;
	double degree = 0;

	double angle = 0;
	double step = 1;
	double s0 = 255;
	double ii, jj, rii, rjj, x, y;

	int i, j;
	int row = input.rows;
	int col = input.cols;

	for (angle = -90; angle <= 90; angle = angle + step)
	{
		angle = angle*PI / 180.0;
		
		for (j = 3; j < row - 3; j++)
		{
			double* current = Temp.ptr<double>(j);
			for (i = 3; i < col - 3; i++)
			{
				ii = i + 2 * sin(angle);
				jj = j + 2 * cos(angle);
				rii = floor(ii);
				rjj = floor(jj);
				x = ii - rii;
				y = jj - rjj;
				current[i] = (1 - x - y + x*y)*input.at<uchar>(rjj, rii) + (y - x*y)*input.at<uchar>(rjj + 1, rii) + (x - x*y)*input.at<uchar>(rjj, rii + 1) + x*y*input.at<uchar>(rjj + 1,rii + 1);
				current[i] = abs(current[i] - input.at<uchar>(j, i)); //求@角度下的微分图像灰度差绝对值
				a += current[i];
			}
		}

		if (a < min)
		{
			min = a;
			n = angle;
		}
	}
	angle = -angle;
	if (angle < 0)
		angle = 180 + angle;

	return angle;
}



/*****************************************************************
*函数名：directionDiffAngle1
*功能：  方向微分法估计运动模糊角度,微元半径为1，2X2微分算子
*****************************************************************/
double  directionDiffAngle1(Mat &input)
{
	Mat mask = Mat_<double>(2, 2);
	double angle = 0, step = 1;
	int m = 0, n = 0;

	double s0 = 255;
	double sx = 0, sy = 0, temp;
	int i, j;
	int row = input.rows;
	int col = input.cols;


	for (angle = -90; angle <= 90; angle = angle + step)
	{
		sx = 0;
		sectionMask1(angle, mask);
		for (j = 1; j < row - 1; j = j + 2)
		{
			sy = 0;
			uchar* previous = input.ptr<uchar>(j - 1);
			uchar* current = input.ptr<uchar>(j);
			uchar* next = input.ptr<uchar>(j + 1);
			for (i = 0; i < col - 1; i = i + 1)
			{
				if (angle >= -90.0&&angle < 0.0)
				{
					temp = (mask.at<double>(0)*previous[i] + mask.at<double>(1)*previous[i + 1] +
						mask.at<double>(2)*current[i] + mask.at<double>(3)*current[i + 1]);
					if (temp>255)
						temp = 255;
					temp = temp / (col - 1);
				}
				else
				{
					temp = (mask.at<double>(0)*current[i] + mask.at<double>(1)*current[i + 1] +
						mask.at<double>(2)*next[i] + mask.at<double>(3)*next[i + 1]);
					if (temp > 255)
						temp = 255;
					temp = temp / (col - 1);
				}
				sy = sy + temp;
			}
			sx += sy / (row - 2);
		}

		if (sx < s0)
		{
			s0 = sx;
			n = m;
		}
		m++;
	}
	angle = -90 + n*step;
	if (angle > 0)
		angle = 180 - angle;
	angle = -angle;

	return angle;
}
/*****************************************************************
*函数名：sectionMask1
*功能：  判定待插值像素区域，计算双线性插值后方向微分掩膜
*****************************************************************/
void sectionMask1(double angle, Mat &mask)
{
	if (angle >= -90.0&&angle < 0)
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = 1 - cos(angle) + sin(angle) - sin(angle)*cos(angle);
		mask.at<double>(1) = cos(angle) + sin(angle)*cos(angle);
		mask.at<double>(2) = -sin(angle) - sin(angle)*cos(angle) - 1;
		mask.at<double>(3) = -sin(angle)*cos(angle);
	}
	else if (angle >= 0 && angle < 90.0)
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = -cos(angle) + sin(angle) - sin(angle)*cos(angle);
		mask.at<double>(1) = cos(angle) + sin(angle)*cos(angle);
		mask.at<double>(2) = -sin(angle) - sin(angle)*cos(angle);
		mask.at<double>(3) = -sin(angle)*cos(angle);
	}
}

/*****************************************************************
*函数名：sectionMask
*功能：  判定待插值像素区域，计算双线性插值后方向微分掩膜
*****************************************************************/
void sectionMask(double angle, Mat &mask)
{
	if (angle >= -90.0&&angle < -60.0)
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = -1 - 2 * sin(angle) + 2 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(1) = -2 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(2) = 0.0;
		mask.at<double>(3) = 2 + 2 * sin(angle) - 4 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(4) = 4 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(5) = 0.0;
		mask.at<double>(6) = -1.0;
		mask.at<double>(7) = 0.0;
		mask.at<double>(8) = 0.0;
	}
	else if (angle >= -60.0&&angle < -30.0)
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = 0.0;
		mask.at<double>(1) = -2 - 4 * sin(angle) + 2 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(2) = 1 + 2 * sin(angle) - 2 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(3) = 0.0;
		mask.at<double>(4) = 4 + 4 * sin(angle) - 4 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(5) = -2 - 2 * sin(angle) + 4 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(6) = -1.0;
		mask.at<double>(7) = 0.0;
		mask.at<double>(8) = 0.0;
	}
	else if (angle >= -30.0&&angle < 0.0)
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = 0.0;
		mask.at<double>(1) = 0.0;
		mask.at<double>(2) = 0.0;
		mask.at<double>(3) = 0.0;
		mask.at<double>(4) = -4 * sin(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(5) = 2 * sin(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(6) = -1.0;
		mask.at<double>(7) = 2 + 4 * sin(angle) - 2 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(8) = -1 - 2 * sin(angle) + 2 * cos(angle) + 4 * sin(angle)*cos(angle);
	}
	else if (angle >= 0.0&&angle < 30.0)
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = -1.0;
		mask.at<double>(1) = 2 - 4 * sin(angle) - 2 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(2) = -1 + 2 * sin(angle) + 2 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(3) = 0.0;
		mask.at<double>(4) = 4 * sin(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(5) = -2 * sin(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(6) = 0.0;
		mask.at<double>(7) = 0.0;
		mask.at<double>(8) = 0.0;
	}
	else if (angle >= 30.0&&angle < 60.0)
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = -1.0,
			mask.at<double>(1) = 0.0;
		mask.at<double>(2) = 0.0;
		mask.at<double>(3) = 0.0;
		mask.at<double>(4) = 4 - 4 * sin(angle) - 4 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(5) = -2 + 2 * sin(angle) + 4 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(6) = 0.0;
		mask.at<double>(7) = -2 + 4 * sin(angle) + 2 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(8) = 1 - 2 * sin(angle) - 2 * cos(angle) + 4 * sin(angle)*cos(angle);
	}
	else
	{
		angle = angle*PI / 180.0;
		mask.at<double>(0) = -1.0;
		mask.at<double>(1) = 0.0;
		mask.at<double>(2) = 0.0;
		mask.at<double>(3) = 2 - 2 * sin(angle) - 4 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(4) = 4 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(5) = 0.0;
		mask.at<double>(6) = -1 + 2 * sin(angle) + 2 * cos(angle) - 4 * sin(angle)*cos(angle);
		mask.at<double>(7) = -2 * cos(angle) + 4 * sin(angle)*cos(angle);
		mask.at<double>(8) = 0.0;
	}
}


void cepstral(Mat &input, Mat &output)
{
	Mat lg, lgcep, temp;

	fft2(input, temp);
	calMag(temp, temp);
	log((1 + temp), lg);
	dft(lg, lgcep, DFT_INVERSE + DFT_SCALE);
	fftshift(lgcep, lgcep);
	calMag(lgcep, temp);
	output = temp;
}


/***********************************************************
函数名称：get_angle
功能：求模糊角度（方向）
参数：*data：width*height维的Y值
返回值：求得的模糊角度
/***********************************************************/
float get_angle(double *motion_matrix, unsigned char*data,
	const unsigned short width, const unsigned short height)
{
	//unsigned char angle;
	float min, sumx, sumy, angle = 0, alpha;
	min = 255.0;
	for (alpha = -90.0; alpha < 90.0; alpha++)//对每个角度求微分图像的和并找出能量最小的方向
	{
		int i, j;
		sumx = 0.0;

		matrix_orient(alpha, motion_matrix);

		for (i = 2; i < height - 2; i = i + 4)
		{
			sumy = 0.0;
			for (j = 0; j < width - 2; j = j + 4)
				sumy += (float)motion_orient(data, i, j, motion_matrix, alpha, width) / (width - 2);
			sumx += sumy / (height - 4);
		}

		if (sumx <= min + 0.0025 && fabs(angle - alpha) >= 5.0)
		{
			min = sumx;
			angle = alpha;
		}
	}
	//  motion_matrix = matrix_orient(angle);
	//printf("模糊角度是：%d",angle);
	return angle;

}




///***********************************************************
//函数名称：get_length
//功能：求模糊尺度
//参数：*Y_buf：width*height维的Y值
//返回值：求得的模糊尺度
///***********************************************************/
//unsigned short get_length(float *Y_buf,
//	unsigned short width, unsigned short height)
//{
//	float *S_buf, *S, min;
//	int i, j, k;
//	S = (float*)malloc(width*sizeof(float));
//	S_buf = (float*)malloc(width*height*sizeof(float));
//
//	memset(S, 0, width * 4);
//	memset(S_buf, 0, width*height * 4);
//	/*for(i=0;i<height;i+=10)
//	for(j=0;j<width;j+=10)
//	{
//	printf("%3.0f,",*(Y_buf+i*width+j));
//	}*/
//	(void)diff_coef(Y_buf, width, height); //第一步除以2 bytank
//
//
//	/*printf("差分后的Y_buf\n");
//	for(i=0;i<height;i+=10)
//	for(j=0;j<width;j+=10)
//	{
//	printf("%3.0f,",*(Y_buf+i*width+j));
//	}*/
//	for (i = 0; i<height; i++)//第二步：自相关
//	{
//		for (j = 0; j<width; j++)
//		{
//			for (k = 0; k<width - j; k++)//注意-j的动作
//			{
//				*(S_buf + i*width + j) += *(Y_buf + i*width + k)*(*(Y_buf + i*width + k + j)) / 128;//problem whereabouts
//			}
//		}
//	}
//
//
//	/* printf("S_buf:\n");
//	for(i=0;i<height;i+=10)
//	for(j=0;j<width;j+=10)
//	{
//	printf("%3.0f,",*(S_buf+i*width+j));
//	}*/
//	for (j = 0; j<width; j++)//求每一列的和
//	{
//		for (i = 0; i<height; i++)
//		{
//			S[j] += *(S_buf + i*width + j) / width;
//		}
//
//
//	}
//
//	min = 255.0;
//	for (j = 0; j<width; j++)//看哪一列的和最小
//	{
//		if (S[j]<min)
//		{
//			min = S[j];
//			k = j;
//		}
//	}
//
//
//	free(S);
//	S = NULL;
//
//	return k;
//}


/* 确定运动模糊角度的两个函数 */
void matrix_orient(float angle, double *motion_matrix)
{


	if (angle >= -90.0&&angle < -60.0)
	{
		motion_matrix[0] = -1 - 2 * sin(angle*PI / 180.0) + 2 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[1] = -2 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[2] = 0.0;
		motion_matrix[3] = 2 + 2 * sin(angle*PI / 180.0) - 4 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[4] = 4 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[5] = 0.0;
		motion_matrix[6] = -1.0;
		motion_matrix[7] = 0.0;
		motion_matrix[8] = 0.0;
	}
	else if (angle >= -60.0&&angle < -30.0)
	{
		motion_matrix[0] = 0.0;
		motion_matrix[1] = -2 - 4 * sin(angle*PI / 180.0) + 2 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[2] = 1 + 2 * sin(angle*PI / 180.0) - 2 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[3] = 0.0;
		motion_matrix[4] = 4 + 4 * sin(angle*PI / 180.0) - 4 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[5] = -2 - 2 * sin(angle*PI / 180.0) + 4 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[6] = -1.0;
		motion_matrix[7] = 0.0;
		motion_matrix[8] = 0.0;
	}
	else if (angle >= -30.0&&angle < 0.0)
	{
		motion_matrix[0] = 0.0;
		motion_matrix[1] = 0.0;
		motion_matrix[2] = 0.0;
		motion_matrix[3] = 0.0;
		motion_matrix[4] = -4 * sin(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[5] = 2 * sin(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[6] = -1.0;
		motion_matrix[7] = 2 + 4 * sin(angle*PI / 180.0) - 2 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[8] = -1 - 2 * sin(angle*PI / 180.0) + 2 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
	}
	else if (angle >= 0.0&&angle < 30.0)
	{
		motion_matrix[0] = -1.0;
		motion_matrix[1] = 2 - 4 * sin(angle*PI / 180.0) - 2 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[2] = -1 + 2 * sin(angle*PI / 180.0) + 2 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[3] = 0.0;
		motion_matrix[4] = 4 * sin(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[5] = -2 * sin(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[6] = 0.0;
		motion_matrix[7] = 0.0;
		motion_matrix[8] = 0.0;
	}
	else if (angle >= 30.0&&angle < 60.0)
	{
		motion_matrix[0] = -1.0,
			motion_matrix[1] = 0.0;
		motion_matrix[2] = 0.0;
		motion_matrix[3] = 0.0;
		motion_matrix[4] = 4 - 4 * sin(angle*PI / 180.0) - 4 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[5] = -2 + 2 * sin(angle*PI / 180.0) + 4 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[6] = 0.0;
		motion_matrix[7] = -2 + 4 * sin(angle*PI / 180.0) + 2 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[8] = 1 - 2 * sin(angle*PI / 180.0) - 2 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
	}
	else
	{
		motion_matrix[0] = -1.0;
		motion_matrix[1] = 0.0;
		motion_matrix[2] = 0.0;
		motion_matrix[3] = 2 - 2 * sin(angle*PI / 180.0) - 4 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[4] = 4 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[5] = 0.0;
		motion_matrix[6] = -1 + 2 * sin(angle*PI / 180.0) + 2 * cos(angle*PI / 180.0) - 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[7] = -2 * cos(angle*PI / 180.0) + 4 * sin(angle*PI / 180.0)*cos(angle*PI / 180.0);
		motion_matrix[8] = 0.0;
	}


	//return motion_matrix;
}

unsigned char motion_orient(unsigned char *Y, unsigned int i, unsigned int j,
	double *motion_matrix, float angle, unsigned int width)
{
	unsigned char k1, k2;
	unsigned char y;
	double fm;

	fm = 0.0;
	for (k1 = 0; k1 < 3; k1++)
	for (k2 = 0; k2 < 3; k2++)
	{
		if (angle >= -90.0&&angle<0.0)
			//fm = fm + Y[i-2+k1][j+k2]*motion_matrix[k1*3+k2];
			fm = fm + *(Y + (i - 2 + k1)*width + j + k2)*motion_matrix[k1 * 3 + k2];
		else
			//fm = fm + Y[i  +k1][j+k2]*motion_matrix[k1*3+k2];
			fm = fm + *(Y + (i + k1)*width + j + k2)*motion_matrix[k1 * 3 + k2];
	}

	fm = fabs(fm);

	if (fm > 255)
		y = 255;
	else y = (unsigned char)(fm + 0.5);

	return y;
}

double cepstrumAngle(Mat &input)
{
	Mat temp,edge;
	Mat planes[] = { Mat_<float>(input), Mat::zeros(input.size(), CV_32F) };

	fft2(input, temp);

	split(temp, planes);
	magnitude(planes[0],planes[1],planes[0]);
	planes[0] += Scalar::all(1);
	log(planes[0], planes[0]);
	fftshift(planes[0],planes[0]);
	normalize(planes[0], planes[0], 1, 0, NORM_MINMAX);
	imshow("频谱", planes[0]);
	//
	dft(planes[0], temp, DFT_INVERSE + DFT_SCALE);
	split(temp, planes);
	magnitude(planes[0], planes[1], planes[0]);
	normalize(planes[0], planes[0], 1, 0, NORM_MINMAX);
	fftshift(planes[0], planes[0]);

	Mat can = Mat_<uchar>(planes[0]);
	Canny(can, edge, 0.1, 0.5);

	imshow("倒谱", planes[0]);
	imshow("edge", edge);
	
	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180,5);

	cout << lines.size() << endl;
	cout << lines.at(1)[1]/PI*180 << endl;
	cout << lines.at(2)[1] / PI * 180 << endl;
	cout << lines.at(3)[1] / PI * 180 << endl;
	cout << lines.at(4)[1] / PI * 180 << endl;



	

	return 0;
}

double cepstrumLen(Mat &input)
{

	return 0;
}