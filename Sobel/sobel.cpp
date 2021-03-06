#include <string>
#include <iostream>

#include <cv.h>
#include <highgui.h>
#include <imgproc/imgproc.hpp>
#include <photo/photo.hpp>

using namespace cv;
using namespace std;

int scale = 2;
int delta = 0;
int image = 0;
int thresh = 250;
int spatial_window = 10; 
int color_window = 36; 
int alpha = 42;
int beta = 66; 
int contourSizeLimit = 10;
int canny_thresh = 100;

Mat src, src_gray;
Mat grad;

string window_name = "Sobel Demo - Simple Edge Detector";

static void image_callback(int, void*);
static void sobel_callback(int, void*);

void resizeCol(Mat& m, size_t columns, size_t rows, const Scalar& s, Point startingPoint = Point(0,0))
{
    Mat tm(m.rows + rows, m.cols + columns, m.type());
    tm.setTo(s);
    m.copyTo(tm(Rect(startingPoint, m.size())));
    m = tm;
}

Mat obtainLeftRegion(Mat img)
{
	Mat img_local, leftRegion;
	//blur(img, img_local, Size(5,5), Point(-1,-1));
	//fastNlMeansDenoising(img,img_local, 20);

	leftRegion = img(Range::all(), Range(0,70));
	resizeCol(leftRegion, img.cols - 70, 0, Scalar(150,150,150));

	return leftRegion;
}

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

static void image_callback(int, void*)
{
	src = imread("../data/" + to_string(image + 1)+".png");
	Mat leftRegion = obtainLeftRegion(src);
	fastNlMeansDenoising(leftRegion,src, 50);
	//adjustBrightnessContrast(src, alpha*0.1, beta);
	pyrMeanShiftFiltering(src, src, spatial_window, color_window);
	cvtColor( src, src_gray, CV_RGB2GRAY );
	imshow("Original", src_gray);
	sobel_callback(0,0);
}

static void sobel_callback(int, void*)
{
	int ddepth = CV_16S;
	/// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

	vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 1.5, abs_grad_y, 1.5, 0, grad );

	//Canny( src_gray, grad, canny_thresh, canny_thresh*2, 3 );
	threshold( grad, grad, thresh, 255,  CV_THRESH_BINARY);
	imshow( window_name, grad );

	findContours( grad, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Mat drawing = Mat::zeros( grad.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       	Scalar color = Scalar(255,255,255);
				if(contourArea(contours[i]) >= contourSizeLimit)//contours[i].size() >= contourSizeLimit
	       	drawContours( drawing, contours, i, color, CV_FILLED, 8 );
     }

  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}

/** @function main */
int main( int argc, char** argv )
{
  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	
	createTrackbar( "Image:", window_name, &image, 47, image_callback );
	createTrackbar( "Scale:", window_name, &scale, 20, sobel_callback );
	createTrackbar( "Depth:", window_name, &delta, 20, sobel_callback );
	createTrackbar( "Thresh:", window_name, &thresh, 255, sobel_callback );
	createTrackbar( "Spatial:", window_name, &spatial_window, 20, image_callback );
	createTrackbar( "Color:", window_name, &color_window, 255, image_callback);
	createTrackbar( "Contrast:", window_name, &alpha, 255, image_callback);
	createTrackbar( "Brightness:", window_name, &beta, 255, image_callback);
	createTrackbar( "Contour limit:", window_name, &contourSizeLimit, 500, image_callback);
	createTrackbar( "Canny Threshold:", window_name, &canny_thresh, 100, image_callback);

	image_callback(0,0);

  waitKey(0);

  return 0;
  }
