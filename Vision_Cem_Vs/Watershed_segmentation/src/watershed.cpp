/**
*	Author:	Octavio Navarro Hinojosa
*	Date:	June 2015
*	
*	Watershed segmentation using dynamically generated masks. 
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

vector<string> 	fileNames;			/** Vector that contains the filenames to process */  
//Mat 				markerMask, 			/** Mat that stores the generated mask */			
//				img, img0,
//				imgGray;		
Point 			prevPt(-1, -1);
int 			image_window = 35, 		/** Variable that controls the image selection trackbar */
			thresh_window = 15,		/** Variable that controls the binary threshold selection trackbar */
			trans_window = 5,
			blur_window = 3;
string          	src_file;				/** String that contains the source file location */
int             	mode = 0;				/** Mode indicates whether to process one image, with mode value = 0, (provided as a 									paramanter), or all the test images, with mode value = 1 */
unsigned long   	AAtime=0, AAtimeCpu = 0; /** Variables that allow the measuring of the execution time */

/**
* Loads the filenames to process, and creates the basic trackbars for the application.
*/
void setUp(void);

/**
* Applies the watershed algorithm to the input image
* @param directory The directory where the images to process are located. Only used if mode == 1 to save the output images. 
* @param filename The file to process. Only used if mode == 1 to save the output images. 
* @param mode The mode to use. 1 to save the output results, 0 to show the results.
*/
void watershed_callback(string directory, string filename, int mode, Mat img0, Mat img, Mat imgGray, Mat markerMask);

/**
* Loads a single image to be processed. The image selection is done through the image trackbar.
* The images for this test are in a fixed directory that is hardcoded. If a change to the location of the images is made, the path should be changed appropriately. 
* @param int An integer that is passed from the trackbar.
* @param void A function to be called when the trackbar is moved. 
*/
void image_callback(int, void*);

/**
* Separes the different regions of an image and generates the mask to be used in the watershed algorithm.
*/
void region_separation(Mat &markerMask, Mat imgGray);

/**
* This method is used only if mode == 0. It sets up the window and trackbars, and processess a single image. 
*/
void singleImage(void);

/**
* This method is used only if mode == 1. It loads and processess all the images contained in a cvs file. 
*/
void allImages(void);

/**
* Takes an image and perfoms the mask generation and watershed algorithm.
* @param directory The directory where the images to process are located. Only used if mode == 1 to save the output images. 
* @param filename The file to process. Only used if mode == 1 to save the output images. 
* @param mode The mode to use. 1 to save the output results, 0 to show the results.
*/
void processImage(string directory, string filename, int mode, Mat originalImage, Mat &processedImage);

void obtainTransductor(Mat src, Mat &dst);

void obtainTransductorHole(Mat src, Mat &dst);

void obtainTransductorHole(Mat src, Mat &dst)
{
	Mat rightRegion;
	matrixData rightRect;

	src.copyTo(rightRegion);

	obtainRegionInMat(rightRegion, 120, src.cols, 0, src.rows, Scalar(255, 255, 255));
	threshold(rightRegion, rightRegion, thresh_window, 255, THRESH_BINARY_INV);
	rightRegion = FillHoles(rightRegion);

	rightRect = maxRectInMat(rightRegion);

	src.copyTo(rightRegion);

	obtainRegionInMat(rightRegion, (rightRect.col - rightRect.width), rightRect.col, (rightRect.row - rightRect.height), rightRect.row, Scalar(255, 255, 255));
	threshold(rightRegion, rightRegion, thresh_window, 255, THRESH_BINARY);

	for (int i = rightRect.col - rightRect.width; i < rightRect.col; i++)
	for (int j = rightRect.row - rightRect.height; j < rightRect.row; j++)
	{
		uchar index = rightRegion.at<uchar>(Point(i,j));

		if (index == 0)
			rightRegion.at<uchar>(Point(i,j)) = 7;
		else if (index == 255)
			rightRegion.at<uchar>(Point(i, j)) = 0;
	}

	rightRegion = removeSmallBlobs(rightRegion, 120, Scalar::all(7));
	obtainRegionInMat(rightRegion, (rightRect.col - rightRect.width), rightRect.col,
		(rightRect.row - rightRect.height), rightRect.row, Scalar::all(0));
	
	imshow("Th", rightRegion*255);
	rightRegion.copyTo(dst);
}

void obtainTransductor(Mat src, Mat &dst)
{
	Mat temp;
	int area_to_remove = 5;

	//cvtColor(src, srcGray, COLOR_BGR2GRAY);

	obtainRegionInMat(src, 80, src.cols, Scalar(255, 255, 255));
	temp = src;
	dst = Mat::zeros(dst.size(), dst.type());
		//srcGray(Range::all(), Range(75, srcGray.cols));

	for (int row = 0; row < temp.rows; row++)
	for (int col = 0; col < temp.cols; col++)
	{
		if (row <= area_to_remove || row >= src.rows - area_to_remove ||
			col <= area_to_remove || col >= src.cols - area_to_remove)
			src.at<uchar>(row, col) = 255;
	}

	threshold(src, src, trans_window, 255, THRESH_BINARY_INV);

	blur_window = blur_window % 2 == 0 ? blur_window + 1 : blur_window;
	medianBlur(src, src, blur_window);

	int i, j, compCount = 0, idx;
	vector<vector<Point>> contours, endContours;
	vector<Vec4i> hierarchy;
	
	// Checks to see if any contours are found in the mask. 
	findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		double area = contours[i].size();
			//contourArea(contours[i]);
		cout << area << endl;
		if (area > 50)
			endContours.push_back(contours[i]);
	}
	cout << endl;

	vector<vector<Point> > contours_poly(endContours.size());

	for (int i = 0; i < endContours.size(); i++)
		approxPolyDP(Mat(endContours[i]), contours_poly[i], 0, true);

	//for (idx = 0; idx >= 0; idx = hierarchy[idx][0])
	for (int i = 0; i< endContours.size(); i++)
	{
		//drawContours(dst, contours, idx, color, 1, 8, hierarchy);
		drawContours(dst, contours_poly, i, Scalar::all(6), 1, 8);
	}

	//FillHoles(dst);
	imshow("transductor", dst * 255);
}

void watershed_callback(string directory, string filename, int mode, Mat img0, Mat img, Mat imgGray, Mat markerMask)
{
	cvtColor(imgGray, imgGray, COLOR_GRAY2BGR);
	int i, j, compCount = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Checks to see if any contours are found in the mask. 
	//findContours(markerMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	//if( contours.empty() )
	//	return;
	//	
	//Mat markers(markerMask.size(), CV_32S);
	//markers = Scalar::all(0);
	//
	//vector<Scalar> colorTab_temp;
	//for (i = 0; i < hierarchy.size(); i++)
	//{
	//	int b = theRNG().uniform(0, 255);
	//	int g = theRNG().uniform(0, 255);
	//	int r = theRNG().uniform(0, 255);

	//	colorTab_temp.push_back(Scalar((uchar)b, (uchar)g, (uchar)r));
	//}

	//// Generate the marker mask to be used by the watershed algorithm. It is based on the rectangular masks obtained previously. 
	//int idx = 0;
	//for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
	//	drawContours(markers, contours, idx, colorTab_temp[idx], -1, 8, hierarchy, INT_MAX);

	//if( compCount == 0 )
	//	return;

	vector<Vec3b> colorTab;
	for( i = 0; i < 50; i++ )
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);

		colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	//
	//markers.convertTo(markers, CV_32S);

	double t = (double)getTickCount();
	watershed( img0, markerMask);
	t = (double)getTickCount() - t;

	Mat wshed(img0.size(), CV_8UC3);
	//Mat wshed(markers.size(), CV_8UC3);

	// Paint the watershed image.
	for (i = 0; i < markerMask.rows; i++)
	for (j = 0; j < markerMask.cols; j++)
		{
			int index = markerMask.at<int>(i, j);
			if( index == -1 )
				wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
			else if( index <= 0)
				wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
			else
				wshed.at<Vec3b>(i,j) = colorTab[index - 1];
		}

	// Add transparency in order to see the original image and the watershed areas. 
	wshed = wshed*0.5 + imgGray*0.5;
	
	if(mode == 0)
        imshow( "watershed transform", wshed );
        
    else if(mode == 1)
    {
    		imwrite("../../data/allData/processedWatershed/"+directory + "-" +filename+"_1_original.png", imgGray);
		imwrite("../../data/allData/processedWatershed/"+directory + "-" +filename+"_2_regions.png", wshed);
//		imwrite("../../../data/allData/processedWatershed/"+directory + "-" +filename+"_1_original.png", imgGray);
//		imwrite("../../../data/allData/processedWatershed/"+directory + "-" +filename+"_2_regions.png", wshed);
    }
}

void region_separation(Mat &markerMask, Mat imgGray)
{
	Mat leftRegion, middleRegion, rightRegion, regionBin, transductorMat, transductorHoleMat;
	matrixData leftRect, middleRect, rightRect;

	// Separates the region based on information given. The region is segmented at column 75.
	middleRegion = imgGray(Range::all(), Range(0,75));
	resizeCol(middleRegion, imgGray.cols - 75, Scalar(0,0,0));

	threshold( middleRegion, regionBin, thresh_window, 255, THRESH_BINARY );

	// Obtain the max rectangle in the binary image. 
	middleRect = maxRectInMat(regionBin);

	// Paint the obtained rectangle in the marker mask. 
	rectangle(markerMask, Point(middleRect.col-10, middleRect.row-10), Point(middleRect.col-middleRect.width+10,middleRect.row-middleRect.height+10), Scalar::all(4), 1);

	leftRegion = imgGray(Range::all(), Range(0,middleRect.col - middleRect.width));
	resizeCol(leftRegion, imgGray.cols - (middleRect.col - middleRect.width), Scalar(0,0,0));

	threshold( leftRegion, regionBin, thresh_window, 255, THRESH_BINARY );
    regionBin = removeSmallBlobs(regionBin, 400);
    regionBin = FillHoles(regionBin);

	leftRect = maxRectInMat(regionBin);

	if(leftRect.col < 90)
        rectangle(markerMask, Point(leftRect.col-3, leftRect.row-3), Point(leftRect.col-leftRect.width+3,leftRect.row-leftRect.height+3), Scalar::all(5), 1);

	obtainTransductorHole(imgGray, transductorHoleMat);
	transductorHoleMat.convertTo(transductorHoleMat, CV_32S);
	markerMask += transductorHoleMat;
    
	imgGray.copyTo(rightRegion);

    //obtainRegionInMat(rightRegion, 120, imgGray.cols, 0, imgGray.rows, Scalar(255,255,255));
    //threshold( rightRegion, regionBin, thresh_window, 255, THRESH_BINARY_INV );
    //regionBin = FillHoles(regionBin);

    //rightRect= maxRectInMat(regionBin);

    //rectangle(markerMask, Point(rightRect.col, rightRect.row), Point(rightRect.col-rightRect.width,rightRect.row-rightRect.height), Scalar(255,0,0), 1);

	transductorMat = Mat(rightRegion.size(), rightRegion.type());
	obtainTransductor(rightRegion, transductorMat);

	transductorMat.convertTo(transductorMat, CV_32S);
	markerMask += transductorMat;

	// These  points were added to mask the background of the image. 
    rectangle(markerMask, Point(10, 10), Point(11,11), Scalar::all(1), 1); 	// The upper left corner, to obtain the left background
	rectangle(markerMask, Point(90, 125), Point(90, 125), Scalar::all(2), 1); 	// The middle of the image, to obtain the right background
	rectangle(markerMask, Point(90, 3), Point(237, 237), Scalar::all(3), 1);	// A rectangle surrounding the right background, to eliminate 																	additional noise

    Mat wshed(markerMask.size(), CV_8UC3);
	imgGray.convertTo(imgGray, CV_32S);
    wshed = markerMask*0.5 + imgGray*0.5;

	if (mode == 0)
	{
		Mat markers;
		markerMask.convertTo(markers, CV_8UC3);
		imshow("markers", markers*1000);
	}
	
}

void image_callback(int, void*)
{
	// Fixed path to the test images. Can be changed as needed. 
    src_file = "../../data/TestData/"+fileNames.at(image_window);

    // This line was added just to have a different path to the files. Can be deleted as needed. 
//    src_file = "../../../data/TestData/"+fileNames.at(image_window);
	Mat img0 = imread(src_file, 1), img;

	if( img0.empty() )
	{
		cout << "Couldn't open image " << src_file << ". Usage: watershed <image_name>\n";
		return;
	}

	processImage("", "", mode, img0, img);

    if(mode == 0)
    {
        imshow( "image", img );
    }
}

void processImage(string directory, string filename, int mode, Mat img0, Mat &img)
{
	Mat imgGray, markerMask;
	img0.copyTo(img);
	blur(img, img, Size(3,3));

	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	//cvtColor(img, markerMask, COLOR_BGR2GRAY);

	markerMask = Mat::zeros(img0.size(), CV_32S);

	region_separation(markerMask,imgGray);

	watershed_callback(directory, filename, mode, img0, img, imgGray, markerMask);
}

void setUp(void)
{
	createNames(fileNames);
	namedWindow( "image", WINDOW_NORMAL );
	createTrackbar("Image", "image",&image_window,47,image_callback);
	createTrackbar("Threshold", "image",&thresh_window,50,image_callback);
	createTrackbar("Trans Threshold", "image", &trans_window, 30, image_callback);
	createTrackbar("Blur Threshold", "image", &blur_window, 10, image_callback);
}

void allImages(void)
{
	ThreadPool pool(8);
	createNames(fileNames);

	AAtimeCpu = getTickCount();

	cout << "STARTED" << endl;

	vector<vector<string>> cvsContents = getCsvContent("../../data/allData/surveyFilesToProcess.csv");
//	vector<vector<string>> cvsContents = getCsvContent("../../../data/allData/surveyFilesToProcess.csv");
//	vector<string> filenames;

	for(int i = 0; i < cvsContents.size(); i++)
	{
		pool.Enqueue([=]()
		{
			string directory = cvsContents[i][0];

			for(int j = 1; j< cvsContents[i].size(); j++)
			{
				Mat img0 = imread("../../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png");
				Mat img;
	//			img0 = imread("../../../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png");
				processImage(directory, cvsContents[i][j], mode, img0, img);
			}
		}
		);
	}

	pool.ShutDown();
	cout << "FINISHED" << endl;
	ProccTimePrint(AAtimeCpu , "cpu");

}

void singleImage(void)
{
    setUp();

	image_callback(0,0);

	waitKey(0);
}

int main( int argc, char** argv )
{
    if(argc > 1)
	{
		if(argv[1] == std::string("1"))
		{
			mode = 1;
			allImages();
			return 0;
		}
	}

	mode = 0;
	singleImage();
    return 0;
}

