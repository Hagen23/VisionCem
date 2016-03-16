/*
Transductor segmentation based on thresholding, a median filter, and contour extraction.
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ThreadPool.h>
#include <utilityFunctions.h>
#include <cstdio>
#include <iostream>

#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

vector<string> 		fileNames;			/** Vector that contains the filenames to process */
string          	src_file;				/** String that contains the source file location */
int 				image_window = 35; 		/** Variable that controls the image selection trackbar */
int 				region_window = 3; 		/** Variable that controls the image selection trackbar */
int 				threshold_window = 5; 		/** Variable that controls the image selection trackbar */

/**
* Loads the filenames to process, and creates the basic trackbars for the application.
*/
void setUp(void);

/**
* Loads a single image to be processed. The image selection is done through the image trackbar.
* The images for this test are in a fixed directory that is hardcoded. If a change to the location of the images is made, the path should be changed appropriately.
* @param int An integer that is passed from the trackbar.
* @param void A function to be called when the trackbar is moved.
*/
void image_callback(int, void*);

/**
* This method is used only if mode == 0. It sets up the window and trackbars, and processess a single image.
*/
void singleImage(void);

/**
* Takes an image and perfoms the mask generation and watershed algorithm.
* @param directory The directory where the images to process are located. Only used if mode == 1 to save the output images.
* @param filename The file to process. Only used if mode == 1 to save the output images.
* @param mode The mode to use. 1 to save the output results, 0 to show the results.
*/
void processImage(Mat originalImage, Mat &processedImage);

void image_callback(int, void*)
{
	// Fixed path to the test images. Can be changed as needed. 
	src_file = "../../VisionCem-master/VisionCem-master/data/TestData/" + fileNames.at(image_window);

	// This line was added just to have a different path to the files. Can be deleted as needed. 
	//    src_file = "../../../data/TestData/"+fileNames.at(image_window);
	Mat img0 = imread(src_file, 1), img;

	if (img0.empty())
	{
		cout << "Couldn't open image " << src_file << ". Usage: watershed <image_name>\n";
		return;
	}

	processImage(img0, img);

	imshow("image", img0);
}

void processImage(Mat src, Mat &dst)
{
	Mat srcGray, markerMask, rightRegion, waterShedImage;
	int area_to_remove = 5;

	cvtColor(src, srcGray, COLOR_BGR2GRAY);

	rightRegion = srcGray(Range::all(), Range(75, srcGray.cols));
	waterShedImage = src(Range::all(), Range(75, src.cols));
	srcGray = srcGray(Range::all(), Range(75, srcGray.cols));

	for (int row = 0; row < rightRegion.rows; row++)
	for (int col = 0; col < rightRegion.cols; col++)
	{
		if (row <= area_to_remove || row >= rightRegion.rows - area_to_remove ||
			col <= area_to_remove || col >= rightRegion.cols - area_to_remove)
			rightRegion.at<uchar>(row, col) = 255;
	}

	threshold(rightRegion, rightRegion, threshold_window, 255, THRESH_BINARY_INV);
	
	if (region_window % 2 != 0)
		medianBlur(rightRegion, rightRegion, region_window);
	else
		medianBlur(rightRegion, rightRegion, region_window -1);
	
	//imshow("result", dst);

	int i, j, compCount = 0, idx;
	vector<vector<Point>> contours, endContours;
	vector<Vec4i> hierarchy;

	imshow("rr", rightRegion);

	//dst.convertTo(dst, CV_32S);
	dst = Mat::zeros(rightRegion.rows, rightRegion.cols, CV_32S);
	
	// Checks to see if any contours are found in the mask. 
	findContours(rightRegion, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 30)
			endContours.push_back(contours[i]);
	}

	vector<vector<Point> > contours_poly(endContours.size());

	for (int i = 0; i < endContours.size(); i++)
		approxPolyDP(Mat(endContours[i]), contours_poly[i], 0, true);

	//for (idx = 0; idx >= 0; idx = hierarchy[idx][0])
	for( int i = 0; i< endContours.size(); i++ )
	{
		Scalar color(255, 0,0);
		//drawContours(dst, contours, idx, color, 1, 8, hierarchy);
		drawContours(dst, contours_poly, i, color, 1, 8);
	}

	//watershed(waterShedImage, dst);

	//Mat wshed(dst.size(), CV_8UC3);

	//vector<Vec3b> colorTab;
	//for (i = 0; i < endContours.size(); i++)
	//{
	//	int b = theRNG().uniform(0, 255);
	//	int g = theRNG().uniform(0, 255);
	//	int r = theRNG().uniform(0, 255);

	//	colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	//}
	//// Paint the watershed image.
	//for (i = 0; i < dst.rows; i++)
	//for (j = 0; j < dst.cols; j++)
	//{
	//	int index = dst.at<int>(i, j);
	//	if (index == -1)
	//		wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
	//	else if (index <= 0 || index > endContours.size())
	//		wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
	//	else
	//		wshed.at<Vec3b>(i, j) = colorTab[index - 1];
	//}

	////wshed = wshed*0.5 + srcGray*0.5;

	//imshow("ws", wshed);
	
	dst.convertTo(dst, CV_8UC3);
	imshow("dst", dst);
	//cvtColor(img, imgGray, COLOR_BGR2GRAY);

}

void setUp(void)
{
	createNames(fileNames);
	namedWindow("rr", WINDOW_NORMAL);
	namedWindow("dst", WINDOW_NORMAL);
	namedWindow("image", WINDOW_NORMAL);
	//namedWindow("ws", WINDOW_NORMAL);

	createTrackbar("Image", "image", &image_window, 47, image_callback);
	createTrackbar("Region window", "image", &region_window, 50, image_callback);
	createTrackbar("Thresh window", "image", &threshold_window, 50, image_callback);
}


void singleImage(void)
{
	setUp();

	image_callback(0, 0);

	waitKey(0);
}

int main(int argc, char** argv)
{
	singleImage();
	return 0;
}