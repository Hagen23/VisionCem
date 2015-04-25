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

int spatial_window = 1;
int color_window = 1;
int cluster_size = 1;

Mat mSFilteringImgHost, mSSegRegionsImgHost, imgIntermedia, mSSegImgHost, outimgProc, outProcPts, 
			bin_mSFilteringImgHost, bin_mSSegImgHost, bin_mSSegRegionsImgHost, gris_mSSegRegionsImgHost, gris_mSFilteringImgHost, gris_mSSegImgHost;

gpu::GpuMat pimgGpu, interGPU, outImgProcGPU, destPoints,  imgGpu, mSFilteringImgGPU;

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
	for(int i = 1; i< 16; i++)
		input.push_back(to_string(i)+".png");
}

static void colorTesting(int, void*)
{	
	gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, spatial_window,color_window, cluster_size);
	imshow("regions", mSSegRegionsImgHost);
}

static void spatialTesting(int, void*)
{
	colorTesting(0,0);
}

static void clusterTesting(int, void*)
{
	spatialTesting(0,0);
}


int main(int argc, char** argv)
{
 unsigned long AAtime=0, AAtimeCpu = 0;
	int element_shape = MORPH_ELLIPSE;	
	Mat element = getStructuringElement(element_shape, Size(2*2+1, 2*2+1), Point(2, 2) );
	vector<string> fileNames;
	createNames(fileNames);
	
 AAtimeCpu = getTickCount();

 Mat img;
	
	string defaultFile("1.png");

	//defaultFile = fileNames[i];
 //image load
 if(argc > 1) 
	{
	 	img = imread( argv[1], 1 );
		defaultFile = argv[1];
		
		int n = 0;
		while(n = defaultFile.find("/") != string::npos)
			defaultFile = defaultFile.substr(n+1);
	}
 else
 	img = imread("../data/"+defaultFile);
 

fastNlMeansDenoising(img,img, 10);
 

	namedWindow("Regions",1);
	createTrackbar("Spatial", "Regions",&spatial_window,255,spatialTesting);
	createTrackbar("Color", "Regions",&color_window,255,colorTesting);
	createTrackbar("cluster", "Regions",&cluster_size,10000,clusterTesting);

 //gpu version meanshift
 
 //AAtime = getTickCount();

 pimgGpu.upload(img);

 //gpu meanshift only support 8uc4 type.
 gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);
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



