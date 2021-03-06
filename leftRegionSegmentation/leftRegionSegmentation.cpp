#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <string>
#include <iostream>
#include <omp.h>

#include <cv.h>
#include <highgui.h>
#include <imgproc/imgproc.hpp>
#include <gpu/gpu.hpp>
#include <photo/photo.hpp>

using namespace cv;
using namespace gpu;
using namespace std;

int image_window = 0;
int spatial_window = 10;
int color_window = 26;
int cluster_size = 2700;
int binary_threshold = 90;

int contrast_threshold = 50;
int brightness_threshold = 15;

Mat mSFilteringImgHost, mSSegRegionsImgHost, imgIntermedia, mSSegImgHost, outimgProc, outProcPts, 
bin_mSFilteringImgHost, bin_mSSegImgHost, bin_mSSegRegionsImgHost, gris_mSSegRegionsImgHost, gris_mSFilteringImgHost, gris_mSSegImgHost;

Mat img;

vector<string> fileNames;

gpu::GpuMat pimgGpu, interGPU, outImgProcGPU, destPoints,  imgGpu, mSFilteringImgGPU;

//alpha = contract ; beta = brightness
void adjustBrightnessContrast( Mat& m, float alpha, int beta)
{
	 /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for( int y = 0; y < m.rows; y++ )
	{ 
		for( int x = 0; x < m.cols; x++ )
		{ 
				m.at<uchar>(y,x) =
				saturate_cast<uchar>( alpha*( m.at<uchar>(y,x) ) + beta );
		}
	}
}

void resizeCol(Mat& m, size_t sz, const Scalar& s)
{
    Mat tm(m.rows, m.cols + sz, m.type());
    tm.setTo(s);
    m.copyTo(tm(Rect(Point(0, 0), m.size())));
    m = tm;
}


void ProccTimePrint( unsigned long Atime , string msg)   
{   
	unsigned long Btime=0;   
	float sec, fps;   
	Btime = getTickCount();   
	sec = (Btime - Atime)/getTickFrequency();   
	fps = 1/sec;   
	printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(),  sec, fps );   
} 

void createNames(vector<string> & input)
{
	for(int i = 1; i<= 48; i++)
		input.push_back(to_string(i)+".png");
}

static void colorTesting(int, void*)
{	
	gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, spatial_window,color_window, cluster_size);
	
	imshow("regions", mSSegRegionsImgHost);
	cvtColor( mSSegRegionsImgHost, mSSegRegionsImgHost, COLOR_RGB2GRAY );
	threshold( mSSegRegionsImgHost, mSSegRegionsImgHost, binary_threshold, 255,  CV_THRESH_BINARY); 
	imshow("bin regions", mSSegRegionsImgHost);
}

static void spatialTesting(int, void*)
{
	colorTesting(0,0);
}

static void clusterTesting(int, void*)
{
	spatialTesting(0,0);
}

static void imageSwitching(int, void*)
{
	img = imread("../data/"+fileNames.at(image_window));
	imshow("Original image", img);

	fastNlMeansDenoising(img,img, 20);

	cvtColor( img, gris_mSFilteringImgHost, COLOR_RGB2GRAY );

	Mat leftRegion = img(Range::all(), Range(0,65));

	resizeCol(leftRegion, img.cols - 65, Scalar(150,150,150));
	adjustBrightnessContrast(leftRegion, contrast_threshold*0.1, brightness_threshold);
	imshow("Segmented region", leftRegion);

	pimgGpu.upload(leftRegion);

	//gpu meanshift only support 8uc4 type.
	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);

	clusterTesting(0,0);
}

static void binaryAdjustment(int, void*)
{
		imageSwitching(0,0);
}

static void brightnessAdjustment(int, void*)
{
	binaryAdjustment(0,0);
}

static void contrastAdjustment(int, void*)
{
	brightnessAdjustment(0,0);
}

int main(int argc, char** argv)
{
	unsigned long AAtime=0, AAtimeCpu = 0;
	int element_shape = MORPH_ELLIPSE;	
	Mat element = getStructuringElement(element_shape, Size(2*2+1, 2*2+1), Point(2, 2) );

	createNames(fileNames);

	AAtimeCpu = getTickCount();

	namedWindow("Regions",1);
	createTrackbar("Spatial", "Regions",&spatial_window,20,spatialTesting);
	createTrackbar("Color", "Regions",&color_window,50,colorTesting);
	createTrackbar("Cluster", "Regions",&cluster_size,5000,clusterTesting);
	createTrackbar("Image", "Regions",&image_window,47,imageSwitching);
	createTrackbar("Binary Threshold", "Regions",&binary_threshold,150,imageSwitching);

	createTrackbar("Contrast", "Regions",&contrast_threshold,50,imageSwitching);
	createTrackbar("Brightness", "Regions",&brightness_threshold,50,imageSwitching);

	contrastAdjustment(0,0);
	
	ProccTimePrint(AAtimeCpu , "cpu");
	waitKey();

	return 0;
}



