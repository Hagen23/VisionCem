#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 0, limit = 50, contourSizeLimit = 50;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );
void limit_callback(int, void* );

/** @function main */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  src = imread( argv[1], 1 );

  /// Convert image to gray and blur it
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// Create Window
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  createTrackbar( " Limit:", "Source", &limit, max_thresh, limit_callback );
  thresh_callback( 0, 0 );

  waitKey(0);
  return(0);
}

void limit_callback(int, void*)
{
	contourSizeLimit = limit;
	thresh_callback( 0, 0 );
}

/** @function thresh_callback */
void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours, endContours;
  vector<Vec4i> hierarchy;

  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
	imshow( "Threshold result ", threshold_output );
  /// Find contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	cout << endl << endl;
	
  for( int i = 0; i < contours.size(); i++ )
     { 
				cout << i << " " << contours[i].size() << endl;
				double area = contourArea(contours[i]);
				if(area > contourSizeLimit)
				{
					endContours.push_back(contours[i]);
					cout << "Added " << i << endl;
				}
     }
	cout << "Final size " << endContours.size() << endl;
  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( endContours.size() );
  vector<Rect> boundRect( endContours.size() );
  vector<Point2f>center( endContours.size() );
  vector<float>radius( endContours.size() );



  for( int i = 0; i < endContours.size(); i++ )
     { approxPolyDP( Mat(endContours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
     }


  /// Draw polygonal contour + bonding rects + circles
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< endContours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

       drawContours( drawing, contours_poly, i, color, CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
//       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//       circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
     }

  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}

//using namespace cv;
//using namespace std;

//Mat src; Mat src_gray;
//int thresh = 100;
//int max_thresh = 255;
//RNG rng(12345);

///// Function header
//void thresh_callback(int, void* );

///** @function main */
//int main( int argc, char** argv )
//{
//  /// Load source image and convert it to gray
//  src = imread( argv[1], 1 );

//  /// Convert image to gray and blur it
//  cvtColor( src, src_gray, CV_BGR2GRAY );
//  blur( src_gray, src_gray, Size(3,3) );

//  /// Create Window
//  char* source_window = "Source";
//  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
//  imshow( source_window, src );

//  createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
//  thresh_callback( 0, 0 );

//  waitKey(0);
//  return(0);
//}

///** @function thresh_callback */
//void thresh_callback(int, void* )
//{
//  Mat canny_output;
//  vector<vector<Point> > contours;
//  vector<Vec4i> hierarchy;

//  /// Detect edges using canny
//  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
//  /// Find contours
//  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

//  /// Draw contours
//  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
//  for( int i = 0; i< contours.size(); i++ )
//     {
//       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
//     }

//  /// Show in a window
//  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
//  imshow( "Contours", drawing );
//}

//using namespace cv;
//using namespace std;

//int main( int argc, char** argv )
//{
//		Mat src;
//		if(argc > 1)
//		src = imread(argv[1]);
//		else
//		src = imread("../data/15.png");

//		cvtColor(src, src, CV_BGR2GRAY);
//		threshold(src, src, 127, 255, CV_THRESH_OTSU);

//    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

//    src = src > 1;
//    namedWindow( "Source", 1 );
//    imshow( "Source", src );

//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;

//    findContours( src, contours, hierarchy,
//        CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

//    // iterate through all the top-level contours,
//    // draw each connected component with its own random color
//    int idx = 0;
//    for( ; idx >= 0; idx = hierarchy[idx][0] )
//    {
//        Scalar color( rand()&255, rand()&255, rand()&255 );
//        drawContours( dst, contours, idx, color, 1, 8, hierarchy );
//    }

//    namedWindow( "Components", 1 );
//    imshow( "Components", dst );
//    waitKey(0);
//}
