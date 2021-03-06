//#include <time.h>   
//#include <opencv2\opencv.hpp>   
//#include <opencv2\gpu\gpu.hpp>   
//#include <string>   
//#include <stdio.h>   

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

int image_window = 14;
//int spatial_window = 10;
int spatial_window = 0;
int color_window = 0;
//int color_window = 26;
int cluster_size = 2700;
int binary_threshold = 20;
int morphSize = 5; //5 for 14-19; 7 for 33-36
int contrast_threshold = 0;
int brightness_threshold = 0;
int blobFilter = 0;

Mat mSFilteringImgHost, mSSegRegionsImgHost, imgIntermedia, mSSegImgHost, outimgProc, outProcPts, 
bin_mSFilteringImgHost, bin_mSSegImgHost, bin_mSSegRegionsImgHost, gris_mSSegRegionsImgHost, gris_mSFilteringImgHost, gris_mSSegImgHost, ms_fil_left;

Mat img, transductor_img;

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

void removeAllSmallerBlobs(Mat& m, float minArea)
{
	float maxBlobArea = 0.0;
	int 	maxBlobIndex = -1;
	
	Mat m_in(m);

	vector<vector<Point> > contours, endContours;
  vector<Vec4i> hierarchy;

	findContours( m, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	cout << "Found " << contours.size() << " blobs " << endl;

	for( int i = 0; i < contours.size(); i++ )
	{ 
			float area = contourArea(contours[i]);
			if(area >= minArea)
			{
				if(area > maxBlobArea)
				{
					maxBlobArea = area;
					maxBlobIndex = i;
				}
			}
			cout << "blob " << i << " area " << area << endl;
	}
	
	if(maxBlobIndex >= 0)
	{
		cout << "Index " << maxBlobIndex << endl;

		endContours.push_back(contours[maxBlobIndex]);

		vector<vector<Point> > contours_poly( endContours.size() );

		for( int i = 0; i < endContours.size(); i++ )
			approxPolyDP( Mat(endContours[i]), contours_poly[i], 0, true );

		m = Mat::zeros( m.size(), CV_8UC3 );

		for( int i = 0; i< endContours.size(); i++ )
		{
			Scalar color = Scalar( 255,255,255 );
			drawContours( m, contours_poly, i, color, CV_FILLED, 8 );
		}
	}
}

//Open abre lo negro
//Close abre lo blanco

static void colorTesting(int, void*)
{	
	int element_shape = MORPH_ELLIPSE;
	Mat element = getStructuringElement(element_shape, Size(2*morphSize+1, 2*morphSize+1), Point(morphSize,  morphSize) );
	
	gpu::meanShiftFiltering(imgGpu, imgGpu, spatial_window,color_window);

	gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, spatial_window,color_window, cluster_size);

	imgGpu.download(ms_fil_left);

	cvtColor( ms_fil_left, ms_fil_left, COLOR_RGB2GRAY );
	threshold( ms_fil_left, ms_fil_left, binary_threshold, 255,  CV_THRESH_BINARY);

	imshow("ms filtering regions", ms_fil_left);
	//adjustBrightnessContrast(mSSegRegionsImgHost, contrast_threshold*0.1, brightness_threshold);

	imshow("regions", mSSegRegionsImgHost);
	cvtColor( mSSegRegionsImgHost, mSSegRegionsImgHost, COLOR_RGB2GRAY );
	threshold( mSSegRegionsImgHost, mSSegRegionsImgHost, binary_threshold, 255,  CV_THRESH_BINARY);
	morphologyEx(mSSegRegionsImgHost, mSSegRegionsImgHost, CV_MOP_OPEN, element);
	morphologyEx(mSSegRegionsImgHost, mSSegRegionsImgHost, CV_MOP_CLOSE, element);
	imshow("bin regions", mSSegRegionsImgHost);

	Mat tissue = ms_fil_left - mSSegRegionsImgHost;
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);

	removeAllSmallerBlobs(tissue, blobFilter);

	pimgGpu.upload(img);
	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);
	gpu::meanShiftFiltering(imgGpu, imgGpu, spatial_window,color_window);

	imgGpu.download(transductor_img);
	cvtColor( transductor_img, transductor_img, COLOR_RGB2GRAY );
	threshold( transductor_img, transductor_img, binary_threshold, 255,  CV_THRESH_BINARY);

	Mat transductor =  mSSegRegionsImgHost - transductor_img ;

	element_shape = MORPH_RECT;
	element = getStructuringElement(element_shape, Size(2*3+1, 2*3+1), Point(3,  3) );
	morphologyEx(transductor, transductor, CV_MOP_CLOSE, element);
	morphologyEx(transductor, transductor, CV_MOP_OPEN, element);
	
	imshow("Tissue", tissue);
	imshow("Transductor", transductor);
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
	imshow("Regions", img);

	//fastNlMeansDenoising(img,img, 5);
	
	//blur(img, img, Size(5,5), Point(0,0));
	//cvtColor( img, gris_mSFilteringImgHost, COLOR_RGB2GRAY );
//	threshold( gris_mSFilteringImgHost, bin_mSFilteringImgHost, binary_threshold, 255,  CV_THRESH_BINARY); 

//	imshow("bin img", bin_mSFilteringImgHost);

	Mat leftRegion = img(Range::all(), Range(0,70));
	//resize(leftRegion, leftRegion, img.size());
	resizeCol(leftRegion, img.cols - 70, Scalar(150,150,150));
	//adjustBrightnessContrast(leftRegion, contrast_threshold*0.1, brightness_threshold);
	//imshow("Segmented region", leftRegion);

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
//	int element_shape = MORPH_ELLIPSE;	
//	Mat element = getStructuringElement(element_shape, Size(2*2+1, 2*2+1), Point(2, 2) );

	createNames(fileNames);

	AAtimeCpu = getTickCount();

	//string defaultFile("3.png");

	//defaultFile = fileNames[i];
	//image load
//	if(argc > 1) 
//	{
//		img = imread( argv[1], 1 );
//		defaultFile = argv[1];

//		int n = 0;
//		while(n = defaultFile.find("/") != string::npos)
//			defaultFile = defaultFile.substr(n+1);
//	}
//	else
//		img = imread("../data/"+defaultFile);

//	unsigned int index = 0;
//	for(int j = 0;j < bin_mSFilteringImgHost.rows;j++)
//	{
//	    for(int i = 0;i < bin_mSFilteringImgHost.cols;i++)
//		{
//		   unsigned char b = bin_mSFilteringImgHost.at<unsigned char>(i,j);
//			if(b == 255)
//			{
//				cout << "Found " << i << " " << j << endl;
//				index = i;
//				break;
//			}
//	    }
//		if(index != 0)
//			break;
//	}

	namedWindow("Regions",1);
	createTrackbar("Spatial", "Regions",&spatial_window,20,spatialTesting);
	createTrackbar("Color", "Regions",&color_window,50,colorTesting);
	createTrackbar("Cluster", "Regions",&cluster_size,5000,clusterTesting);
	createTrackbar("Image", "Regions",&image_window,47,imageSwitching);
	createTrackbar("Binary Threshold", "Regions",&binary_threshold,30,imageSwitching);
	createTrackbar("Morph size Threshold", "Regions",&morphSize,10,imageSwitching);
	createTrackbar("Contrast", "Regions",&contrast_threshold,50,imageSwitching);
	createTrackbar("Brightness", "Regions",&brightness_threshold,50,imageSwitching);
	createTrackbar("Blobs", "Regions",&blobFilter,100,imageSwitching);

	contrastAdjustment(0,0);
	//gpu version meanshift

	//AAtime = getTickCount();

	
	//gpu::blur(imgGpu, interGPU, Size(5,5), Point(-1,-1));

	// BASE VALUES; these work for 15, 16, 24, and similar
	// gpu::meanShiftFiltering(imgGpu, mSFilteringImgGPU, 40, 30);

	// gpu::meanShiftSegmentation(imgGpu, mSSegImgHost, 100,20, 300);
	////Segment the gel, air, and water
	// gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, 5,30, 1800);

	//	TermCriteria criteria(TermCriteria::MAX_ITER, 5, 1);
	//	// To get Transductor
	// 	gpu::meanShiftFiltering(imgGpu, mSFilteringImgGPU, 1, 30);

	//	// To get Tissue
	// 	gpu::meanShiftSegmentation(imgGpu, mSSegImgHost, 1,20, 300);

	//Segment the gel, air, and water
	//	gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, 100,100, 300);

	// mSFilteringImgGPU.download(mSFilteringImgHost);
	//// ProccTimePrint(AAtime , "gpu");

	// cvtColor( mSFilteringImgHost, gris_mSFilteringImgHost, COLOR_RGB2GRAY );
	// threshold( gris_mSFilteringImgHost, bin_mSFilteringImgHost, 20, 255,  CV_THRESH_BINARY ); 

	// cvtColor( mSSegImgHost, gris_mSSegImgHost, COLOR_RGB2GRAY );
	// threshold( gris_mSSegImgHost, bin_mSSegImgHost,  20, 255,  CV_THRESH_BINARY ); 

	// cvtColor( mSSegRegionsImgHost, gris_mSSegRegionsImgHost, COLOR_RGB2GRAY );
	// threshold( gris_mSSegRegionsImgHost, bin_mSSegRegionsImgHost, 20, 255,  CV_THRESH_BINARY );

	// Mat transductor_img =  bin_mSSegRegionsImgHost - bin_mSFilteringImgHost;
	// Mat tissue_img = bin_mSSegImgHost - bin_mSSegRegionsImgHost;

	//	
	//	morphologyEx(tissue_img, tissue_img, CV_MOP_OPEN, element);
	//	morphologyEx(tissue_img, tissue_img, CV_MOP_CLOSE, element);

	//	element_shape = MORPH_RECT;
	//	morphologyEx(transductor_img , transductor_img , CV_MOP_OPEN, element);
	//	morphologyEx(transductor_img , transductor_img , CV_MOP_CLOSE, element);

	//show image

	// imshow("origin", img);
	// imshow("intermedia", imgIntermedia);
	// imshow("MeanShift Filter cpu", outImg);
	// imshow("bin_mSFilteringImgHost", bin_mSFilteringImgHost);
	// imshow("bin_mSSegRegionsImgHost", bin_mSSegRegionsImgHost);
	// imshow("gris_mSSegRegionsImgHost", gris_mSSegRegionsImgHost);
	// imshow(" bin_mSSegImgHost",  bin_mSSegImgHost);

	// imshow("Resta mSFiltering - SegRegions", diff_img);
	// imshow("Resta SSegImgHost - SegRegions", diff_img2);

	//	imwrite("../data/"+defaultFile+"_imgFilter.png", img);
	//	imwrite("../data/"+defaultFile+"_MS.png", gris_mSFilteringImgHost);
	//	imwrite("../data/"+defaultFile+"_MSSegmented.png", gris_mSSegImgHost);
	//	imwrite("../data/"+defaultFile+"_regions.png", gris_mSSegRegionsImgHost);

	//	imwrite("../data/"+defaultFile+"_MS_bin.png", bin_mSFilteringImgHost);
	//	imwrite("../data/"+defaultFile+"_MSSegmented_bin.png", bin_mSSegImgHost);
	//	imwrite("../data/"+defaultFile+"_regions_bin.png", bin_mSSegRegionsImgHost);

	//	imwrite( "../data/"+defaultFile+"_tissue.png", tissue_img );
	//	imwrite("../data/"+defaultFile+"_transductor.png", transductor_img);
	// imshow("MeanShift Filter gpu", outimg2);
	// imshow("MeanShift Segmentation gpu", outImgSegmentation);
	//imshow("MeanShift Proc gpu", outimgProc);
	//imshow("MeanShift Proc Pts gpu", outProcPts);
	//cout << "Pts = " << outProcPts << endl;

	cout << "FINISHED" << endl;
	ProccTimePrint(AAtimeCpu , "cpu");
	waitKey();

	return 0;
}



