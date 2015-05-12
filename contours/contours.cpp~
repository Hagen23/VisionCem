//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <math.h>
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>

//#include <cv.h>
//#include <highgui.h>
//#include <imgproc/imgproc.hpp>
//#include <gpu/gpu.hpp>
//#include <photo/photo.hpp>

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

Mat src; Mat src_gray;
int thresh = 45, limit = 500, contourSizeLimit = 500;
int max_thresh = 255;
RNG rng(12345);

int spatial_window = 10;
int color_window = 0;
int image_window = 0;
int contrast_threshold = 14;
int brightness_threshold = 29;

vector<string> fileNames;

/// Function header
void thresh_callback(int, void* );
void limit_callback(int, void* );

void adjustBrightnessContrast( Mat& m, float alpha, int beta)
{
	uchar aux;
	 /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for( int y = 0; y < m.rows; y++ )
	{ 
		for( int x = 0; x < m.cols; x++ )
		{ 
			for(int i = 0; i< 3; i++)
			{
				aux = m.at<Vec3b>(y,x)[i];
				m.at<Vec3b>(y,x)[i] =
				saturate_cast<uchar>( alpha*aux + beta );
			}
		}
	}
}

void morph(Mat& m)
{
	int element_shape = MORPH_ELLIPSE;
	Mat element = getStructuringElement(element_shape, Size(2*2+1, 2*2+1), Point(3, 3) );
	morphologyEx(m, m, CV_MOP_OPEN, element);
	morphologyEx(m, m, CV_MOP_CLOSE, element);

//	element_shape = MORPH_RECT;
//	morphologyEx(m , m , CV_MOP_OPEN, element);
//	morphologyEx(m , m , CV_MOP_CLOSE, element);
}

static void imageSwitching(int, void*)
{
	src = imread("../data/"+fileNames.at(image_window));

  imshow( "Source", src );
	//imshow("Original image", src);

//	blur( src_gray, src_gray, Size(3,3) );
	fastNlMeansDenoising(src,src,10);
	adjustBrightnessContrast(src, contrast_threshold*0.1, brightness_threshold);
	pyrMeanShiftFiltering(src, src, spatial_window, color_window);
  cvtColor( src, src_gray, CV_BGR2GRAY );
	imshow("Denoised image", src_gray);
	
//	morph(src_gray);
  /// Convert image to gray and blur it
  //blur( src_gray, src_gray, Size(3,3) );

	thresh_callback(0,0);
}

void createNames(vector<string> & input)
{
	for(int i = 1; i<= 48; i++)
		input.push_back(to_string(i)+".png");
}

/** @function main */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  //src = imread( argv[1], 1 );

	createNames(fileNames);

	src = imread("../data/"+fileNames.at(image_window));

  /// Create Window
  namedWindow( "Source", 1);

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  createTrackbar( " Limit:", "Source", &limit, contourSizeLimit, limit_callback );
	createTrackbar("Image", "Source",&image_window ,47,imageSwitching);
	createTrackbar("Contrast", "Source",&contrast_threshold,50,imageSwitching);
	createTrackbar("Brightness", "Source",&brightness_threshold,50,imageSwitching);
	createTrackbar("Spatial", "Source",&spatial_window,20,imageSwitching);
	createTrackbar("Color", "Source",&color_window,50,imageSwitching);
  imageSwitching( 0, 0 );

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

//	morph( threshold_output);
	imshow( "Threshold result ", threshold_output );
  /// Find contours

  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);//, Point(0, 0) );
//  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

//	cout << endl << endl;
	
  for( int i = 0; i < contours.size(); i++ )
     { 
//				cout << i << " " << contours[i].size() << endl;
				double area = contourArea(contours[i]);
				if(area > contourSizeLimit)
				{
					endContours.push_back(contours[i]);
//					cout << "Added " << i << endl;
				}
     }
//	cout << "Final size " << endContours.size() << endl;
  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( endContours.size() );
  vector<Rect> boundRect( endContours.size() );
  vector<Point2f>center( endContours.size() );
  vector<float>radius( endContours.size() );

  for( int i = 0; i < endContours.size(); i++ )
     { approxPolyDP( Mat(endContours[i]), contours_poly[i], 0, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
     }


  /// Draw polygonal contour + bonding rects + circles
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< endContours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

       drawContours( drawing, contours_poly, i, color, CV_FILLED, 8 );
//       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//       circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
     }

	cvtColor( drawing, drawing, CV_BGR2GRAY );
	threshold( drawing, drawing, thresh, 255, THRESH_BINARY );
  /// Show in a window
	morph(drawing);
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
