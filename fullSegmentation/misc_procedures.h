#ifndef misc_procedures
#define misc_procedures

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <imgproc/imgproc.hpp>
#include <gpu/gpu.hpp>
#include <photo/photo.hpp>

#include <algorithm>   
#include <vector>       

using namespace cv;
using namespace std;
using namespace gpu;

RNG 	rng(12345);

bool descending (int i,int j) { return (i>j); }

void adjustBrightnessContrastV3( Mat& m, float alpha, int beta)
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

void adjustBrightnessContrast( Mat& m, float alpha, int beta)
{
	uchar aux;
	 /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for( int y = 0; y < m.rows; y++ )
	{ 
		for( int x = 0; x < m.cols; x++ )
		{ 
				aux = m.at<uchar>(y,x);
				m.at<uchar>(y,x) =
				saturate_cast<uchar>( alpha*aux + beta );
		}
	}
}

void resizeCol(Mat& m, size_t columns, size_t rows, const Scalar& s, Point startingPoint = Point(0,0))
{
    Mat tm(m.rows + rows, m.cols + columns, m.type());
    tm.setTo(s);
    m.copyTo(tm(Rect(startingPoint, m.size())));
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

void morph(Mat& m)
{
	int element_shape = MORPH_ELLIPSE;
	Mat element = getStructuringElement(element_shape, Size(2*2+1, 2*2+1), Point(3, 3) );
	//morphologyEx(m, m, CV_MOP_OPEN, element);
	morphologyEx(m, m, CV_MOP_CLOSE, element);
	morphologyEx(m, m, CV_MOP_CLOSE, element);
	morphologyEx(m, m, CV_MOP_CLOSE, element);
	//morphologyEx(m, m, CV_MOP_OPEN, element);
//	element_shape = MORPH_RECT;
//	morphologyEx(m , m , CV_MOP_OPEN, element);
//	morphologyEx(m , m , CV_MOP_CLOSE, element);
}

void removeAllSmallerBlobs(Mat& m, float minArea)
{
	float maxBlobArea = 0.0, secondMaxArea = 0.0;
	int 	maxBlobIndex = 0, secondIndex = 0;
	
	Mat m_in(m);

	vector<vector<Point> > contours, endContours;
  vector<Vec4i> hierarchy;

	findContours( m, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	cout << "Found " << contours.size() << " blobs " << endl;

	for( int i = 0; i < contours.size(); i++ )
	{ 
			float area = contourArea(contours[i]);
			if(area > minArea)
			{
				if(area > maxBlobArea)
				{
					maxBlobArea = area;
					maxBlobIndex = i;
				}
				else
					if(area > secondMaxArea)
					{	
						secondMaxArea = area;
						secondIndex = i;	
					}
			}
			cout << "blob " << i << " area " << area << endl;
	}
	
	cout << "secondIndex " << secondIndex << " secondMaxArea " << secondMaxArea << endl;
	endContours.push_back(contours[secondIndex]);

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

Mat removeSmallBlobs(Mat m, float maxBlobArea)
{
	Mat drawing;

	vector<vector<Point> > contours, endContours;
  vector<Vec4i> hierarchy;

	findContours( m, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	for( int i = 0; i < contours.size(); i++ )
	{ 
			double area = contourArea(contours[i]);
			if(area > maxBlobArea)
				endContours.push_back(contours[i]);
	}

  vector<vector<Point> > contours_poly( endContours.size() );

  for( int i = 0; i < endContours.size(); i++ )
		approxPolyDP( Mat(endContours[i]), contours_poly[i], 0, true );

  drawing = Mat::zeros( m.size(), CV_8UC3 );

  for( int i = 0; i< endContours.size(); i++ )
	{
		Scalar color = Scalar( 255,255,255 );
		drawContours( drawing, contours_poly, i, color, CV_FILLED, 8 );
	}

	return drawing;
}


#endif
