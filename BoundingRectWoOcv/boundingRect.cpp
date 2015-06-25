#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utilityFunctions.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;

int thresh = 35;
int image_window = 14;
int max_thresh = 255;
vector<string> fileNames;
char* source_window;

RNG rng(12345);

void createNames(vector<string> & input)
{
	for(int i = 1; i<= 48; i++)
		input.push_back(to_string(i)+".png");
}

void resizeCol(Mat& m, size_t sz, const Scalar& s)
{
    Mat tm(m.rows, m.cols + sz, m.type());
    tm.setTo(s);
    m.copyTo(tm(Rect(Point(0, 0), m.size())));
    m = tm;
}
/// Function header
void thresh_callback(int, void* );
void image_callback(int, void* );

/** @function main */
int main( int argc, char** argv )
{
	createNames(fileNames);
  /// Load source image and convert it to gray
  //src = imread( argv[1], 1 );

  source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
	createTrackbar("Image", "Source",&image_window,47,image_callback);
	image_callback(0,0);

  waitKey(0);
  return(0);
}

void image_callback(int, void*)
{
	src = imread("../data/"+fileNames.at(image_window));
	/// Convert image to gray and blur it
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

	src_gray = src_gray(Range::all(), Range(0,70));

	resizeCol(src_gray, src.cols - 70, Scalar(0,0,0));
	
  imshow( source_window, src );

  thresh_callback( 0, 0 );
}
/** @function thresh_callback */
void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
	
  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
	imshow("Threshold", threshold_output);
	matrixData rectData = maxRectInMat(threshold_output);
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

	line( drawing, Point(rectData.col, rectData.row), Point(rectData.col -  rectData.width, rectData.row), color, 1, 8 );
	line( drawing, Point(rectData.col -  rectData.width, rectData.row), Point(rectData.col -  rectData.width, rectData.row - rectData.height), color, 1, 8 );
	line( drawing, Point(rectData.col -  rectData.width, rectData.row - rectData.height), Point(rectData.col, rectData.row - rectData.height), color, 1, 8 );
	line( drawing, Point(rectData.col, rectData.row - rectData.height), Point(rectData.col, rectData.row), color, 1, 8 );
				
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
  /// Find contours
//  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

//  /// Find the rotated rectangles and ellipses for each contour
//  vector<RotatedRect> minRect( contours.size() );
//  vector<RotatedRect> minEllipse( contours.size() );

//  for( int i = 0; i < contours.size(); i++ )
//     { minRect[i] = minAreaRect( Mat(contours[i]) );
//       if( contours[i].size() > 5 )
//         { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
//     }

//  /// Draw contours + rotated rects + ellipses
//  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
//  for( int i = 0; i< contours.size(); i++ )
//     {
//       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//       // contour
//       drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//       // ellipse
//      // ellipse( drawing, minEllipse[i], color, 2, 8 );
//       // rotated rectangle
//       Point2f rect_points[4]; minRect[i].points( rect_points );
//       for( int j = 0; j < 4; j++ )
//          line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
//     }

//  /// Show in a window
//  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
//  imshow( "Contours", drawing );
}
