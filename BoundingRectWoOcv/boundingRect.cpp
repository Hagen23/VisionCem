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

int thresh = 10;
int image_window = 16;
int max_thresh = 255;
int small_thresh = 400;
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

/** @function main */
int main( int argc, char** argv )
{
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
			cout << "../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png" << endl;
			img = imread("../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png"); 
			processImage(img, directory, cvsContents[i][j]);
		}
	}
	
	cout << "FINISHED" << endl;
	ProccTimePrint(AAtimeCpu , "cpu");
	
}

void processImage(Mat src, string directory, string filename)
{
	Mat threshold_output, src_gray;
	int element_shape = MORPH_ELLIPSE;
	Mat element = getStructuringElement(element_shape, Size(2*2+1, 2*2+1), Point(3, 3) );
	
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(5,5) );

	src_gray = src_gray(Range::all(), Range(0,70));

	resizeCol(src_gray, src.cols - 70, Scalar(0,0,0));
	
	threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
	threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
		
	matrixData rectData = maxRectInMat(threshold_output);
	
	Mat tissue = threshold_output(Range::all(), Range(0,rectData.col - rectData.width));
	resizeCol(tissue, threshold_output.cols - (rectData.col - rectData.width), Scalar(0,0,0));
	
  Mat gel = threshold_output - tissue;
    
	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);
	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	
	tissue = removeSmallBlobs(tissue, small_thresh);
	
	imwrite("../data/allData/processed/"+directory + "-" +filename+"_original.png", src);
	imwrite("../data/allData/processed/"+directory + "-" +filename+"_tissue.png", tissue );
		imwrite("../data/allData/processed/"+directory + "-" +filename+"_gel.png", gel);
	imwrite("../data/allData/processed/"+directory + "-" +filename+"_transductor.png", transductor);
}

void singleImage(void)
{
	createNames(fileNames);

  source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
	createTrackbar("Image", "Source",&image_window,47,image_callback);
	createTrackbar("Blobs", "Source",&small_thresh,400,thresh_callback);
	image_callback(0,0);

  waitKey(0);
}

void image_callback(int, void*)
{
	src = imread("../data/"+fileNames.at(image_window));
	
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(5,5) );

	src_gray.copyTo(transductor);
	
	src_gray = src_gray(Range::all(), Range(0,70));

	resizeCol(src_gray, src.cols - 70, Scalar(0,0,0));
	
  imshow( source_window, src );

  thresh_callback( 0, 0 );
}
/** @function thresh_callback */
void thresh_callback(int, void* )
{
  Mat threshold_output, threshold_output_transductor;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
 	int element_shape = MORPH_RECT;
 	int morphSize = 5;
	Mat element = getStructuringElement(element_shape, Size(2*morphSize+1, 2*morphSize+1), Point(morphSize, morphSize) );
	
  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  threshold(transductor, threshold_output_transductor, thresh, 255, THRESH_BINARY);
  
  obtainRegionInMat(threshold_output_transductor, 90,threshold_output_transductor.cols, Scalar(255,255,255));
  
  imshow("Transductor", threshold_output_transductor);
  
 	morphologyEx(threshold_output_transductor, threshold_output_transductor, CV_MOP_OPEN, element);
	morphologyEx(threshold_output_transductor, threshold_output_transductor, CV_MOP_OPEN, element);

//	imshow("Threshold", threshold_output);
		
	matrixData rectData = maxRectInMat(threshold_output);
	
	Mat tissue = threshold_output(Range::all(), Range(0,rectData.col - rectData.width));
	resizeCol(tissue, threshold_output.cols - (rectData.col - rectData.width), Scalar(0,0,0));
	
  Mat gel = threshold_output - tissue;
  Mat transductor_final = threshold_output + threshold_output_transductor;
  
  imshow("Gel", gel);
  
  element_shape = MORPH_ELLIPSE;
  morphSize = 2;
	element = getStructuringElement(element_shape, Size(2*morphSize+1, 2*morphSize+1), Point(morphSize, morphSize) );
	
	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);
	morphologyEx(tissue, tissue, CV_MOP_OPEN, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	morphologyEx(tissue, tissue, CV_MOP_CLOSE, element);
	
	tissue = removeSmallBlobs(tissue, small_thresh);
	imshow("Tissue", tissue);
	
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	
	Mat drawing(src_gray); // = Mat::zeros( threshold_output.size(), CV_8UC3 );
	cvtColor( drawing, drawing, CV_GRAY2BGR );

	line( drawing, Point(rectData.col, rectData.row), Point(rectData.col -  rectData.width, rectData.row), color, 1, 8 );
	line( drawing, Point(rectData.col -  rectData.width, rectData.row), Point(rectData.col -  rectData.width, rectData.row - rectData.height), color, 1, 8 );
	line( drawing, Point(rectData.col -  rectData.width, rectData.row - rectData.height), Point(rectData.col, rectData.row - rectData.height), color, 1, 8 );
	line( drawing, Point(rectData.col, rectData.row - rectData.height), Point(rectData.col, rectData.row), color, 1, 8 );
				
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}
