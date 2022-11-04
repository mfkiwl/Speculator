#include "basicoperation.h"
#include "estimatepar.h"
#include "restore.h"
#include "assessment.h"

double *inputImageMatrix, *outputImageMatrix;
fftw_complex *inputImageFFT, *outputImageFFT;
fftw_plan realForwardPlan, realBackwardPlan;

int main()
{
	const string SRC_NAME = "lena64.bmp";
	Mat srcImage = imread(SRC_NAME, CV_LOAD_IMAGE_GRAYSCALE);

	if ( !srcImage.data) 
	{
		printf("No image data ！\n");
		return -1;
	}

	double t = (double)cvGetTickCount();
	imageInit(srcImage, srcImage);
	t = ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000);
	cout << "t=" << t << "ms" << endl;
	cout << "srcImage=" << srcImage << endl;
	cout << "srcImage.data=" << &srcImage.data << endl;
	inputImageMatrix = (double*)(srcImage.data);
	fftw_execute(realForwardPlan);
	cout << "tinputImageMatrix=" << &inputImageMatrix << endl;

	Mat medImage, dstImage1, dstImage2, dstImage3, psf;


	//imageInit(srcImage, srcImage);
	namedWindow("原图", 1);
	imshow("原图",srcImage);

	//fft2(srcImage, srcImage);
	//fftshift(srcImage, srcImage);
	//fftShow(srcImage);
	//Mat A = (Mat_<float>(3, 3) << 3, 2, 1, 4.5, 5, 2, 4, 2, 3); Mat B,C;
	//float d=A.at<float>(1,1);
	//float e = A.at<float>(1,2);
	//calWeight(srcImage, medImage);


	//genaratePsf(psf,25, 10);
	//filter2D(srcImage, medImage, -1, psf, Point(-1, -1), 0, BORDER_REPLICATE);
	//deconvRL3(medImage, psf,dstImage1, 80);

	//imshow("模糊图", medImage);
	//imshow("处理图", dstImage1);

	//freeResource();
	waitKey(0);
	destroyAllWindows();
}