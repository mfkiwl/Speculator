#include "assessment.h"

/*****************************************************************
*��������calSpatialFreq
*���ܣ�  ����ͼ��Ŀռ�Ƶ��
*****************************************************************/
double  calSpatialFreq(Mat &input)
{
	double RF = 0, CF = 0, SF = 0;
	int i, j;
	int row = input.rows;
	int col = input.cols;
	int num = (row/-1)*(col-1);
	double s = 0;

	for (j = 0; j < row-1 ; j++)
	{
		uchar* current = input.ptr<uchar>(j);
		for (i = 0; i < col-1; i++)
		{
			s += (current[i + 1] - current[i])*(current[i + 1] - current[i]);
		}
	}
	RF = sqrt(s / num);

	for (j = 0; j < row-1 ; j++)
	{
		uchar* current = input.ptr<uchar>(j);
		uchar* next = input.ptr<uchar>(j + 1);
		for (i = 0; i < col-1; i++)
		{
			s += (next[i] - current[i])*(next[i] - current[i]);
		}
	}
	CF = sqrt(s / num);

	SF = sqrt(RF*RF + CF*CF);
	return SF;
}

/*****************************************************************
*��������calGMG
*���ܣ�  ����ͼ��Ҷ�ƽ���ݶ�ֵ����ֵԽ���ʾͼ��Խ����
*****************************************************************/
double  calGMG(Mat &input)
{
	Mat I = Mat_<double>(input);
	int i, j;
	int row = input.rows;
	int col = input.cols;
	int num = (row -1)*(col - 1);
	double s = 0;
	double resolution = 0;

	for (j = 0; j < row-1; j++)
	{
		uchar* current = input.ptr<uchar>(j);
		uchar* next = input.ptr<uchar>(j + 1);
		for (i = 0; i < col-1; i++)
		{
			s += sqrt(((current[i + 1] - current[i])*(current[i + 1] - current[i]) + (next[i] - current[i])*(next[i] - current[i])) / 2);
		}
	}
	resolution = s / num;
	
	return resolution;
}


/********************************************************************************
*����������	SpaceFreq ���㲢����һ��ͼ��Ŀռ�Ƶ��
*����������	IplImage *img ��ͨ��8λͼ��
*��������ֵ��double
*********************************************************************************/
double SpaceFreq(IplImage *img)
{
	double RF = 0;
	double CF = 0;
	double SF = 0;

	int i, j;//ѭ������
	int height = img->height;
	int width = img->width;
	int step = img->widthStep / sizeof(uchar);
	uchar *data = (uchar*)img->imageData;
	double num = width*height;

	//��Ƶ����
	for (i = 0; i<height; i++)
	{
		for (j = 0; j<width; j++)
		{
			RF += (data[i*step + j + 1] - data[i*step + j])*(data[i*step + j + 1] - data[i*step + j]);
		}
	}
	RF = sqrt(1.0*RF / num);

	//��Ƶ����
	for (i = 0; i<height; i++)
	{
		for (j = 0; j<width; j++)
		{
			CF += (data[(i + 1)*step + j] - data[i*step + j])*(data[(i + 1)*step + j] - data[i*step + j]);
		}
	}
	CF = sqrt(1.0*CF / num);

	//�ռ�Ƶ��
	SF = sqrt(RF*RF + CF*CF);
	return SF;
}
/********************************************************************************
*����������	DefRto ���㲢����һ��ͼ���������
*����������	IplImage *img ��ͨ��8λͼ��
*��������ֵ��double
*********************************************************************************/
double DefRto(IplImage *img)
{
	double temp = 0;
	double DR = 0;
	int i, j;//ѭ������
	int height = img->height;
	int width = img->width;
	int step = img->widthStep / sizeof(uchar);
	uchar *data = (uchar*)img->imageData;
	double num = width*height;

	for (i = 0; i<height; i++)
	{
		for (j = 0; j<width; j++)
		{
			temp += sqrt((pow((double)(data[(i + 1)*step + j] - data[i*step + j]), 2) + pow((double)(data[i*step + j + 1] - data[i*step + j]), 2)) / 2);
		}
	}
	DR = temp / num;
	return DR;
}


/*************************************************************************
*
* @�������ƣ�
*	calMSE()
*
* @�������:
*   Mat &image1           - ����ͼ��1
*	Mat &image2           - ����ͼ��2
*
* @����ֵ:
*   float                 - ���ؼ������MSE
*
* @�����
*	��
*
* @˵��:
*   �ú���������������ͼ��ľ������MSE
*
************************************************************************/
double calMSE(Mat &image1, Mat &image2)
{
	double MSE = 0, s = 0;
	int row1 = image1.rows;
	int col1 = image1.cols;
	int row2 = image2.rows;
	int col2 = image2.cols;
	int i = 0, j = 0;
	if (row1!=row2||col1!=col2)
		cerr<< "Image size not match!"<< endl;
	for (j = 0; j < row1; j++)
	{
		uchar* pvalue1 = image1.ptr<uchar>(j);
		uchar* pvalue2 = image2.ptr<uchar>(j);
		for (i = 0; i < col1; i++)
		{
			s+=(pvalue1[i] - pvalue2[i])*(pvalue1[i] - pvalue2[i]);
		}
	}
	MSE = s / (row1*col1);
	return MSE;
}

/*************************************************************************
*
* @�������ƣ�
*	calPSNR()
*
* @�������:
*   Mat &image1           - ����ͼ��1
*	Mat &image2           - ����ͼ��2
*
* @����ֵ:
*   float                 - ���ؼ������PSNR
*
* @�����
*	��
*
* @˵��:
*   �ú���������������ͼ��ķ�ֵ�����PSNR��PSNR=10*log10((2^n-1)^2/MSE)
*
************************************************************************/
double calPSNR(Mat &image1, Mat &image2)
{
	double PSNR = 0, MSE = 0;

	MSE = calMSE(image1, image2);
	PSNR = 10 * log10(255 * 255 / MSE) ;

	return PSNR;
}

/*************************************************************************
*
* @�������ƣ�
*	calContrast()
*
* @�������:
*   Mat &image            - ����ͼ��
*
* @����ֵ:
*   float                 - ���ؼ�����ĶԱȶ�
*
* @�����
*	��
*
* @˵��:
*   �ú����������㵥��ͼ��ĶԱȶ�
*
************************************************************************/
double calContrast(Mat &image)
{
	double con = 0;
	double imin, imax;

	minMaxLoc(image, &imin, &imax);
	con = (imax - imin) / (imax + imin);


	return con;
}


/*************************************************************************
*
* @�������ƣ�
*	calWeight()
*
* @�������:
*   Mat &image            - ����ͼ��
*
* @����ֵ:
*   float                 - ���ؼ�����ĶԱȶ�
*
* @�����
*	��
*
* @˵��:
*   �ú���ͨ������ֲ���������ø������ص��Ȩֵ�����Ȩֵͼ�����2X2��
*
************************************************************************/
void calWeight(Mat &input ,Mat &output)
{
	Mat temp = Mat_<double>(input);
	Mat mean ;

	int row = input.rows;
	int col = input.cols;

	for (int j = 0; j < row-2; j += 2)
	{
		for (int i = 0; i < col-2; i += 2)
		{
			Mat re(temp, Rect(i, j, 2, 2));
			meanStdDev(re,mean,re);
		}
	}
	temp = temp.mul(temp);
	Mat lc1(temp, Rect(col - 2, 0, 1, row));
	Mat lc2(temp, Rect(col - 1, 0, 1, row));

	Mat lr1(temp, Rect(0, row - 2, col, 1));
	Mat lr2(temp, Rect(0, row - 1, col, 1));

	lc2.copyTo(lc1);
	lr2.copyTo(lr1);

	output = temp.clone();

}

double calGradPer(Mat &input)
{
	int row = input.rows;
	int col = input.cols;
	int i, j;
	double s = 0;

	Mat tempr = Mat_<uchar>(row, col - 1), tempc = Mat_<uchar>(row-1, col);

	for (j = 0; j < row ; j++)
	{
		uchar* current = input.ptr<uchar>(j);
		uchar* tcurrent = tempr.ptr<uchar>(j);

		for (i = 0; i < col - 1; i++)
		{
			tcurrent[i] = abs(current[i + 1] - current[i]);
		}
	}

	for (j = 0; j < row - 1; j++)
	{
		uchar* current = input.ptr<uchar>(j);
		uchar* next = input.ptr<uchar>(j + 1);
		uchar* tcurrent = tempc.ptr<uchar>(j);

		for (i = 0; i < col ; i++)
		{
			tcurrent[i] = abs(next[i] - current[i]);
		}
	}

	Mat hist;
	int histSie = 255;
	float ranges[] = { 0,255 };
	const float* histRange = { ranges };
	calcHist(&tempr,1,0,Mat(),hist,1,&histSie,&histRange);
	cout << hist << endl;
	cout << hist.size() << endl;
	//cvCountNonZero


	return 0;
}
