#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <utilityFunctions.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 5;
int max_thresh = 50;
int 			image_window = 35; 		/** Variable that controls the image selection trackbar */
vector<string> 	fileNames;			/** Vector that contains the filenames to process */
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);

void image_callback(int, void*)
{
	// Fixed path to the test images. Can be changed as needed. 
	string src_file = "../../VisionCem-master/VisionCem-master/data/TestData/" + fileNames.at(image_window);

	// This line was added just to have a different path to the files. Can be deleted as needed. 
	//    src_file = "../../../data/TestData/"+fileNames.at(image_window);
	src = imread(src_file, 1);

	if (src.empty())
	{
		cout << "Couldn't open image " << src_file << ". Usage: watershed <image_name>\n";
		return;
	}

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	imshow("Source", src);
	thresh_callback(thresh, 0);
}

/** @function main */
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray

	createNames(fileNames);

	/// Create Window
	namedWindow("Source", CV_WINDOW_NORMAL);

	createTrackbar(" Threshold:", "Source", &thresh, max_thresh, image_callback);
	createTrackbar(" Image:", "Source", &image_window, 47, image_callback);
	image_callback(0, 0);

	waitKey(0);
	return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}


	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::ones(threshold_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		
		if (boundRect[i].width * boundRect[i].height > 80)
		{
			cout << i << " " << boundRect[i].width * boundRect[i].height << endl;
			//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
		}

	}

	drawing = drawing * 0.7 + src * 0.3;
	cout << endl;
	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}