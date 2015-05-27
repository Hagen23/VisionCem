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
#include <vector>

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

string defaultFile("24.png");

 Mat mSFilteringImgHost, mSSegRegionsImgHost, imgIntermedia, mSSegImgHost, outimgProc, outProcPts, 
			bin_mSFilteringImgHost, bin_mSSegImgHost, bin_mSSegRegionsImgHost, gris_mSSegRegionsImgHost, gris_mSFilteringImgHost, gris_mSSegImgHost, leftRegion;

 gpu::GpuMat pimgGpu, interGPU, outImgProcGPU, destPoints,  imgGpu, mSFilteringImgGPU;

vector<vector<string>> getCsvContent(string filename)
{
	ifstream file ( filename ); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	string value;
	vector<vector<string> > cvsContents;
	while ( file.good() )
	{
		vector<string> line;
		while(getline ( file, value, ',' ))
			line.push_back(value);
		cvsContents.push_back(line);
	}
	return cvsContents;
}

//alpha = contrast ; beta = brightness
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

void obtainLeftRegion(Mat inImg, Mat &outImg)
{
	Mat greyImg, leftRegionSeg, regionHost, regionHostColor;
	gpu::GpuMat imgGpuSeg, imgGpuColor; 

	fastNlMeansDenoising(inImg,inImg, 5);

	cvtColor( inImg, greyImg, COLOR_RGB2GRAY );

	leftRegionSeg = inImg(Range::all(), Range(0,70));
	resizeCol(leftRegionSeg, inImg.cols - 70, Scalar(150,150,150));
	adjustBrightnessContrast(leftRegionSeg, contrast_threshold*0.1, brightness_threshold);

	imgGpuSeg.upload(leftRegionSeg);
	gpu::cvtColor(imgGpuSeg, imgGpuColor, CV_BGR2BGRA);

	gpu::meanShiftSegmentation(imgGpuColor, regionHost, spatial_window,color_window, cluster_size);
	
	cvtColor( regionHost, regionHostColor, COLOR_RGB2GRAY );
	imwrite( "../data/"+defaultFile+"_png_regions.png", regionHost );
	threshold( regionHostColor, outImg, binary_threshold, 255,  CV_THRESH_BINARY); 
}

void createNames(vector<string> & input)
{
	for(int i = 1; i< 49; i++)
		input.push_back(to_string(i)+".png");
}

int main(int argc, char** argv)
{
 unsigned long AAtime=0, AAtimeCpu = 0;
	int element_shape = MORPH_ELLIPSE;	
	Mat element = getStructuringElement(element_shape, Size(2*2+1, 2*2+1), Point(2, 2) );
	vector<string> fileNames;
	createNames(fileNames);
	
 AAtimeCpu = getTickCount();

	cout << "STARTED" << endl;

	int i = 0;
	vector<vector<string> > cvsContents = getCsvContent("../data/allData/surveyFilesToProcess.csv");
	for(auto v : cvsContents)
	{
		string directory = v[0];
		vector<string> filenames;
		for( int i = 1; i< 5; i++)
			filenames.push_back(v[i]);
	}
	//#pragma omp parallel for
	for(i = 0; i < fileNames.size(); i++)
{
 Mat img;
	
	defaultFile = fileNames[i];

	cout << "Processing " << defaultFile << endl;
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
 
fastNlMeansDenoising(img,img, 20);


 //cpu version meanshift
 //blur(img,img, Size(5,5), Point(-1,-1));
//blur(img,imgIntermedia, Size(2,2), Point(-1,-1));
 //medianBlur(img,imgIntermedia, 5);
 //GaussianBlur(img,imgIntermedia, Size(33,33), 0.9, 0.9);
// pyrMeanShiftFiltering(imgIntermedia, outImg, 70, 30, 2);
// cvtColor( outImg, gris, COLOR_RGB2GRAY );
// threshold( gris, binarizada, 20, 255,  CV_THRESH_BINARY );
// ProccTimePrint(AAtime , "cpu");


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

	TermCriteria criteria(TermCriteria::MAX_ITER, 5, 1);

	// To get Transductor
 	gpu::meanShiftFiltering(imgGpu, mSFilteringImgGPU, 1, 30);

	// To get Tissue
 	gpu::meanShiftSegmentation(imgGpu, mSSegImgHost, 1,20, 300);

	//Segment the gel, air, and water
//	gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, 100,100, 
	obtainLeftRegion(img, leftRegion);

 mSFilteringImgGPU.download(mSFilteringImgHost);
// ProccTimePrint(AAtime , "gpu");

 cvtColor( mSFilteringImgHost, gris_mSFilteringImgHost, COLOR_RGB2GRAY );
 threshold( gris_mSFilteringImgHost, bin_mSFilteringImgHost, 10, 255,  CV_THRESH_BINARY ); 

 cvtColor( mSSegImgHost, gris_mSSegImgHost, COLOR_RGB2GRAY );
 threshold( gris_mSSegImgHost, bin_mSSegImgHost,  20, 255,  CV_THRESH_BINARY ); 

//	cout << "Filtered image to binary 3\n" ;
// cvtColor( mSSegRegionsImgHost, gris_mSSegRegionsImgHost, COLOR_RGB2GRAY );
//	cout << "Filtered image to binary 3.5\n" ;
// threshold( gris_mSSegRegionsImgHost, bin_mSSegRegionsImgHost, 20, 255,  CV_THRESH_BINARY );

 Mat transductor_img =  leftRegion - bin_mSFilteringImgHost;
// Mat transductor_img =  bin_mSSegRegionsImgHost - bin_mSFilteringImgHost;
 Mat tissue_img = bin_mSSegImgHost - leftRegion;

	morphologyEx(tissue_img, tissue_img, CV_MOP_OPEN, element);
	morphologyEx(tissue_img, tissue_img, CV_MOP_CLOSE, element);

	element_shape = MORPH_RECT;
	morphologyEx(transductor_img , transductor_img , CV_MOP_OPEN, element);
	morphologyEx(transductor_img , transductor_img , CV_MOP_CLOSE, element);
 
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
	imwrite("../data/"+defaultFile+"_imgFilter.png", img);
//	imwrite("../data/"+defaultFile+"_MS.png", gris_mSFilteringImgHost);
//	imwrite("../data/"+defaultFile+"_MSSegmented.png", gris_mSSegImgHost);
//	imwrite("../data/"+defaultFile+"_regions.png", gris_mSSegRegionsImgHost);
	
//	imwrite("../data/"+defaultFile+"_MS_bin.png", bin_mSFilteringImgHost);
//	imwrite("../data/"+defaultFile+"_MSSegmented_bin.png", bin_mSSegImgHost);
	imwrite("../data/"+defaultFile+"_regions_bin.png", leftRegion);

	imwrite( "../data/"+defaultFile+"_tissue.png", tissue_img );
	imwrite("../data/"+defaultFile+"_transductor.png", transductor_img);
// imshow("MeanShift Filter gpu", outimg2);
// imshow("MeanShift Segmentation gpu", outImgSegmentation);
 //imshow("MeanShift Proc gpu", outimgProc);
 //imshow("MeanShift Proc Pts gpu", outProcPts);
 //cout << "Pts = " << outProcPts << endl;

}
cout << "FINISHED" << endl;
ProccTimePrint(AAtimeCpu , "cpu");
	return 0;
}



