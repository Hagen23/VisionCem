#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <photo/photo.hpp>

#include "utilityFunctions.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

Mat src, src_gray, transductor;

int thresh = 18;
int image_window = 36;
int max_thresh = 25;
int tissue_thresh = 650;
int transductor_thresh = 1000;
int blur_v = 5;
int mode_g = 0;
vector<string> fileNames;
string source_window;
unsigned long AAtime=0, AAtimeCpu = 0;

RNG rng(12345);

/// Function header
void thresh_callback(int, void* );
void image_callback(int, void* );
void singleImage(void);
void allImages(void);
void processImage(Mat img, string directory, string filename);
void processImage(Mat img, string directory, string filename, int mode);

/** @function main */
int main( int argc, char** argv )
{
	if(argc > 1)
	{
		if(argv[1] == std::string("1"))
		{
			mode_g = 1;
			allImages();
			return 0;
		}
	}

	mode_g = 0;
	singleImage();
		
  	return(0);
}

void allImages(void)
{
	createNames(fileNames);
	
	AAtimeCpu = getTickCount();

	cout << "STARTED" << endl;
	
	vector<vector<string>> cvsContents = getCsvContent("../data/allData/surveyFilesToProcess.csv");
	vector<string> filenames;

	for(int i = 0; i < cvsContents.size(); i++)
	{		
		filenames.clear();
		string directory = cvsContents[i][0];

		for(int j = 1; j< cvsContents[i].size(); j++)
		{
			Mat img;
			//cout << "../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png" << endl;
			img = imread("../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png"); 
			processImage(img, directory, cvsContents[i][j], mode_g);
		}
	}
	
	cout << "FINISHED" << endl;
	ProccTimePrint(AAtimeCpu , "cpu");
	
}

void processImage(Mat src, string directory, string filename, int mode)
{
	cvtColor( src, src_gray, CV_BGR2GRAY );
	blur( src_gray, src_gray, Size(blur_v,blur_v) );

	src_gray.copyTo(transductor);

	src_gray = src_gray(Range::all(), Range(0,70));

	resizeCol(src_gray, src.cols - 70, Scalar(0,0,0));

	Mat threshold_output, threshold_output_transductor;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int element_shape = MORPH_RECT;
	int morphSize = 5;
	Mat element = getStructuringElement(element_shape, Size(2*morphSize+1, 2*morphSize+1), Point(morphSize, morphSize) );

	if(image_window > 23)
		threshold( src_gray, threshold_output, (thresh - 10) > 0?(thresh-10):0, 255, THRESH_BINARY );
	else
		threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

	if(image_window > 23)
		threshold( transductor, threshold_output_transductor, (thresh - 10) > 0?(thresh-10):0, 255, THRESH_BINARY_INV );
	else
		threshold(transductor, threshold_output_transductor, thresh, 255, THRESH_BINARY_INV);

	if(image_window > 23)
		obtainRegionInMat(threshold_output_transductor, 98,threshold_output_transductor.cols, Scalar(0,0,0));
	else
		obtainRegionInMat(threshold_output_transductor, 90,threshold_output_transductor.cols, Scalar(0,0,0));

	morphologyEx(threshold_output_transductor, threshold_output_transductor, CV_MOP_CLOSE, element);
	morphologyEx(threshold_output_transductor, threshold_output_transductor, CV_MOP_CLOSE, element);

	threshold_output_transductor = removeSmallBlobs(threshold_output_transductor, transductor_thresh);

	matrixData rectData = maxRectInMat(threshold_output);
	rectData.printData();
	rectangle(src, Point(rectData.col, rectData.row),Point(rectData.col-rectData.width,rectData.row-rectData.height), Scalar(255,0,0), 2);
	
	Mat aux(threshold_output);
	obtainRegionInMat(aux, rectData.col - rectData.width-5, rectData.col+5, rectData.row - rectData.height-10, rectData.row+10, Scalar(0,0,0));

	Mat tissue = threshold_output(Range::all(), Range(0,rectData.col - rectData.width));
	resizeCol(tissue, threshold_output.cols - (rectData.col - rectData.width), Scalar(0,0,0));

	Mat gel = threshold_output - tissue;
	Mat transductor_final = threshold_output + threshold_output_transductor;

	gel = removeSmallBlobs(gel, tissue_thresh);

	element_shape = MORPH_ELLIPSE;
	morphSize = 2;
	element = getStructuringElement(element_shape, Size(2*morphSize+1, 2*morphSize+1), Point(morphSize, morphSize) );

	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);
	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);

	tissue = removeSmallBlobs(tissue, tissue_thresh);
	
	if(mode == 0)
	{
	//	imshow("Threshold", threshold_output);
		imshow( source_window, src );
		imshow("Transductor", threshold_output_transductor);
		imshow("Tissue", tissue);
		imshow("Gel", gel);
		imshow("AUX", aux);
	}
	if(mode == 1)
	{
		imwrite("../data/allData/processed/"+directory + "-" +filename+"_1_original.png", src);
		imwrite("../data/allData/processed/"+directory + "-" +filename+"_2_tissue.png", tissue );
		imwrite("../data/allData/processed/"+directory + "-" +filename+"_3_gel.png", gel);
		imwrite("../data/allData/processed/"+directory + "-" +filename+"_4_transductor.png", threshold_output_transductor);
	}
}

void singleImage(void)
{
	createNames(fileNames);

	source_window = "Source";
	namedWindow( source_window, CV_WINDOW_AUTOSIZE );

	createTrackbar( "Threshold", "Source", &thresh, max_thresh, thresh_callback );
	createTrackbar("Image", "Source",&image_window,47,image_callback);
	createTrackbar("Blobs Tissue", "Source",&tissue_thresh,1000,thresh_callback);
	createTrackbar("Blobs Transductor", "Source",&transductor_thresh,1000,thresh_callback);
	image_callback(0,0);

	waitKey(0);
}

void image_callback(int, void*)
{
	string src_file = "../data/TestData/"+fileNames.at(image_window);
	src = imread(src_file);

	cvtColor( src, src_gray, CV_BGR2GRAY );
	blur( src_gray, src_gray, Size(blur_v,blur_v) );

	src_gray.copyTo(transductor);

	src_gray = src_gray(Range::all(), Range(0,70));

	resizeCol(src_gray, src.cols - 70, Scalar(0,0,0));

	imshow( source_window, src );

	thresh_callback( 0, 0 );
}
/** @function thresh_callback */
void thresh_callback(int, void* )
{
	processImage(src, "", "", mode_g);
}
