#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
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
int spatial_window = 20;
int color_window = 0;
int cluster_size = 2700;
int binary_threshold = 20;

int contrast_threshold = 50;
int brightness_threshold = 15;

string defaultFile("24.png");

 Mat 					img, transductor_img, leftRegion, ms_fil_left, mSSegRegionsImgHost;

 gpu::GpuMat	pimgGpu, interGPU, outImgProcGPU, destPoints,  imgGpu, mSFilteringImgGPU;

vector<vector<string>> getCsvContent(string filename)
{
	ifstream file ( filename ,  ios::in); 
	string value, value_aux;
	vector<vector<string>> cvsContents;

	while (!file.eof() )
	{
		vector<string> line;
		size_t position0 = 0, position1 = 0;
		getline(file, value);
		while(position1 !=std::string::npos)
		{
			position1 = value.find(",", position1+1);
			value_aux = value.substr(position0, position1-position0);
			if(!value_aux.empty())
				line.push_back(value_aux);
			position0 = position1+1;
		}
		if (!line.empty())
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

//void obtainLeftRegion(Mat inImg, Mat &outImg, string directory, string filename)
//{
//	Mat greyImg, leftRegionSeg, regionHost, regionHostColor;
//	gpu::GpuMat imgGpuSeg, imgGpuColor; 

//	fastNlMeansDenoising(inImg,inImg, 5);
//	//blur(inImg, inImg, Size(5,5), Point(-1,-1));
//	cvtColor( inImg, greyImg, COLOR_RGB2GRAY );

//	leftRegionSeg = inImg(Range::all(), Range(0,70));
//	resizeCol(leftRegionSeg, inImg.cols - 70, Scalar(150,150,150));
//	//adjustBrightnessContrast(leftRegionSeg, contrast_threshold*0.1, brightness_threshold);

//	imgGpuSeg.upload(leftRegionSeg);
//	gpu::cvtColor(imgGpuSeg, imgGpuColor, CV_BGR2BGRA);

//	gpu::meanShiftSegmentation(imgGpuColor, regionHost, spatial_window,color_window, cluster_size);
//	
//	cvtColor( regionHost, regionHostColor, COLOR_RGB2GRAY );
//	imwrite( "../data/allData/processedMS/"+directory + "-" +filename+"_png_regions.png", regionHost );
//	threshold( regionHostColor, outImg, binary_threshold, 255,  CV_THRESH_BINARY); 
//}

void createNames(vector<string> & input)
{
	for(int i = 1; i< 49; i++)
		input.push_back(to_string(i)+".png");
}

//void processImage(Mat img, string directory, string filename)
//{
//	int element_shape =  MORPH_ELLIPSE;
//	int morphSize = 2;
//	Mat element = getStructuringElement(element_shape, Size(2*morphSize+1, 2*morphSize+1), Point(morphSize, morphSize) );
//	Mat test;
//	gpu::GpuMat testGPU;

//	//Segment the gel, air, and water
//	obtainLeftRegion(img, leftRegion, directory, filename);

//	//fastNlMeansDenoising(img,img, 5);
//	pimgGpu.upload(img);

//	gpu::FastNonLocalMeansDenoising obj;
//	obj.labMethod(pimgGpu, pimgGpu, 10,10);
// 	//gpu meanshift only support 8uc4 type.
// 	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);

//	//gpu::blur(testGPU, imgGpu, Size(5,5), Point(-1,-1));
//	gpu::meanShiftFiltering(imgGpu, mSFilteringImgGPU, 0, 0);

//	// To get Tissue
// 	gpu::meanShiftSegmentation(imgGpu, mSSegImgHost, 0,0, 300);

//	mSFilteringImgGPU.download(mSFilteringImgHost);
//	imgGpu.download(test);

//	cvtColor( mSFilteringImgHost, gris_mSFilteringImgHost, COLOR_RGB2GRAY );
//	threshold( gris_mSFilteringImgHost, bin_mSFilteringImgHost, 10, 255,  CV_THRESH_BINARY ); 

//	cvtColor( mSSegImgHost, gris_mSSegImgHost, COLOR_RGB2GRAY );
//	threshold( gris_mSSegImgHost, bin_mSSegImgHost,  20, 255,  CV_THRESH_BINARY ); 

//	Mat transductor_img =  leftRegion - bin_mSFilteringImgHost;
//	Mat tissue_img = bin_mSSegImgHost - leftRegion;

//	morphologyEx(tissue_img, tissue_img, CV_MOP_OPEN, element);
//	morphologyEx(tissue_img, tissue_img, CV_MOP_CLOSE, element);
//	morphologyEx(tissue_img, tissue_img, CV_MOP_CLOSE, element);

//	//removeAllSmallerBlobs(tissue_img, 200);
//	
//	//cout << "PASO BLOBS" << endl;
//  element_shape = MORPH_RECT;
//	morphologyEx(transductor_img , transductor_img , CV_MOP_OPEN, element);
//	morphologyEx(transductor_img , transductor_img , CV_MOP_CLOSE, element);

//	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_imgFilter.png", img);
//	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_regions_bin.png", leftRegion);
//	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_tissue.png", tissue_img );
//	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_transductor.png", transductor_img);
//}

void removeAllSmallerBlobs(Mat& m, float minArea)
{
	float maxBlobArea = 0.0;
	int 	maxBlobIndex = -1;
	
	Mat m_in(m);

	vector<vector<Point> > contours, endContours;
  vector<Vec4i> hierarchy;

	findContours( m, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
//	cout << "Found " << contours.size() << " blobs " << endl;

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
//			cout << "blob " << i << " area " << area << endl;
	}
	
	if(maxBlobIndex >= 0)
	{
//		cout << "Index " << maxBlobIndex << endl;

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

void processImage(Mat img, string directory, string filename)
{
	int element_shape = MORPH_ELLIPSE;
	int morphSize_axial = 5, morphSize_sagittal = 7;
	Mat element = getStructuringElement(element_shape, Size(2*morphSize_axial+1, 2*morphSize_axial+1), Point(morphSize_axial,  morphSize_axial) );

	//cvtColor( img, gris_mSFilteringImgHost, COLOR_RGB2GRAY );
	Mat leftRegion = img(Range::all(), Range(0,70));
	resizeCol(leftRegion, img.cols - 70, Scalar(150,150,150));
	pimgGpu.upload(leftRegion);
	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);

	// For gel segmentation
	gpu::meanShiftFiltering(imgGpu, imgGpu, spatial_window,color_window);

	// For tissue segmentation
	gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, spatial_window,color_window, cluster_size);

	imgGpu.download(ms_fil_left);

	cvtColor( ms_fil_left, ms_fil_left, COLOR_RGB2GRAY );
	threshold( ms_fil_left, ms_fil_left, binary_threshold, 255,  CV_THRESH_BINARY);

	cvtColor( mSSegRegionsImgHost, mSSegRegionsImgHost, COLOR_RGB2GRAY );
	threshold( mSSegRegionsImgHost, mSSegRegionsImgHost, binary_threshold, 255,  CV_THRESH_BINARY);

	morphologyEx(mSSegRegionsImgHost, mSSegRegionsImgHost, CV_MOP_OPEN, element);
	morphologyEx(mSSegRegionsImgHost, mSSegRegionsImgHost, CV_MOP_CLOSE, element);

	Mat tissue = ms_fil_left - mSSegRegionsImgHost;
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);

	removeAllSmallerBlobs(tissue, 0);

	pimgGpu.upload(img);
	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);
	gpu::meanShiftFiltering(imgGpu, imgGpu, spatial_window,color_window);

	Mat transductor_img;

	imgGpu.download(transductor_img);
	cvtColor( transductor_img, transductor_img, COLOR_RGB2GRAY );
	threshold( transductor_img, transductor_img, binary_threshold, 255,  CV_THRESH_BINARY);

	Mat transductor =  mSSegRegionsImgHost - transductor_img ;

	element_shape = MORPH_RECT;
	element = getStructuringElement(element_shape, Size(2*morphSize_sagittal+1, 2*morphSize_sagittal+1), Point(morphSize_sagittal,  morphSize_sagittal) );
	morphologyEx(transductor, transductor, CV_MOP_CLOSE, element);
	morphologyEx(transductor, transductor, CV_MOP_OPEN, element);

	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_original.png", img);
	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_filtering.png", ms_fil_left);
	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_regions_bin.png", mSSegRegionsImgHost);
	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_tissue.png", tissue );
	imwrite("../data/allData/processedMS/"+directory + "-" +filename+"_transductor.png", transductor);
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
	vector<vector<string>> cvsContents = getCsvContent("../data/allData/surveyFilesToProcess.csv");
	vector<string> filenames;

	for(int i = 0; i < cvsContents.size(); i++)
	{		
		filenames.clear();
		string directory = cvsContents[i][0];

		for(int j = 1; j< cvsContents[i].size(); j++)
		{
			Mat img;
			cout << "../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png" << endl;
			img = imread("../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png"); 
			processImage(img, directory, cvsContents[i][j]);
		}
	}
	
	cout << "FINISHED" << endl;
	ProccTimePrint(AAtimeCpu , "cpu");
	return 0;
}



