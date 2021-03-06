/**
*	Author:	Octavio Navarro Hinojosa
*	Date:	June 2015
*	
*	Watershed segmentation using dynamically generated masks. 
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

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
int 			image_window = 36, 		/** Variable that controls the image selection trackbar */
thresh_window = 13,		/** Variable that controls the binary threshold selection trackbar */
trans_window = 10,
blur_window = 3,
water_window = 15;

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
* colors for the regions: 1 - air, 2 - water, 3 - gel, 4 - tissue, 5 - transductor, 6 - transductor hole
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

void obtainTissue(Mat src, Mat &dst);

void obtainTissue(Mat src, matrixData middleRect, Mat &dst)
{
	Mat leftRegion, regionBin, air_marker;
	int area_to_remove = 5;
	air_marker = Mat::zeros(dst.size(), dst.type());

	leftRegion = src(Range::all(), Range(0, middleRect.col - middleRect.width - 5));
	resizeCol(leftRegion, src.cols - (middleRect.col - middleRect.width - 5), Scalar(0, 0, 0));

	threshold(leftRegion, regionBin, thresh_window, 255, THRESH_BINARY);

	blur_window = blur_window % 2 == 0 ? blur_window + 1 : blur_window;
	medianBlur(regionBin, regionBin, blur_window);

	vector<vector<Point>> contours, endContours;
	vector<Vec4i> hierarchy;

	// Checks to see if any contours are found in the mask. 
	findContours(regionBin, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	double max_area = contourArea(contours[0]);
	endContours.push_back(contours[0]);

	for (int i = 1; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > max_area)
		{
			endContours.pop_back();
			endContours.push_back(contours[i]);
			max_area = area;
		}
	}

	vector<vector<Point> > contours_poly(endContours.size());

	for (int i = 0; i < endContours.size(); i++)
		approxPolyDP(Mat(endContours[i]), contours_poly[i], 0, true);

	//for (idx = 0; idx >= 0; idx = hierarchy[idx][0])
	for (int i = 0; i< endContours.size(); i++)
	{
		//drawContours(dst, contours, idx, color, 1, 8, hierarchy);
		drawContours(dst, contours_poly, i, Scalar::all(4), 1, 8);
		drawContours(air_marker, contours_poly, i, Scalar::all(255), -1, 8);
	}

	imshow("Tissue", dst * 255 *0.5f+ src*0.5f);

	int conv_kernel[3][3] = { {1,1,1}, {1,0,1}, {1,1,1} };
	Mat kernel = Mat(Size(3, 3), air_marker.type());

	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		kernel.at<uchar>(i, j) = conv_kernel[i][j];

	do
	{
		filter2D(air_marker, air_marker, -1, kernel, Point(-1, -1));
		contours.clear();
		findContours(air_marker, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	} while (contours.size() > 1);

	Mat air_marker_1;
	air_marker.convertTo(air_marker_1, CV_8UC3);
	imshow("air_marker_1", air_marker_1 * 255 * 0.5 + src*0.5);

	threshold(air_marker, air_marker, 0, 1, THRESH_BINARY_INV);

	obtainRegionInMat(air_marker, 0, middleRect.col - middleRect.width - area_to_remove,
		area_to_remove, air_marker.rows - area_to_remove, Scalar::all(0));

	int dilation_type = MORPH_ELLIPSE;
	int dilation_size = 3;

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));

	erode(air_marker, air_marker, element);
	erode(air_marker, air_marker, element);

	if (mode == 0)
		imshow("air marker", air_marker * 255  + src * 0.5);

	dst += air_marker;
}

void obtainTransductorHole(Mat src, Mat &dst)
{
	Mat rightRegion;
	matrixData rightRect;

	src.copyTo(rightRegion);

	obtainRegionInMat(rightRegion, 80, src.cols, 0, src.rows, Scalar(255, 255, 255));

	threshold(rightRegion, rightRegion, thresh_window, 255, THRESH_BINARY_INV);
	rightRegion = FillHoles(rightRegion);

	rightRect = maxRectInMat(rightRegion);

	src.copyTo(rightRegion);

	obtainRegionInMat(rightRegion, (rightRect.col - rightRect.width), rightRect.col,
		(rightRect.row - rightRect.height), rightRect.row, Scalar(255, 255, 255));
	threshold(rightRegion, rightRegion, thresh_window, 255, THRESH_BINARY_INV);
	
	for (int i = rightRect.col - rightRect.width; i < rightRect.col; i++)
	for (int j = rightRect.row - rightRect.height; j < rightRect.row; j++)
	{
		uchar index = rightRegion.at<uchar>(Point(i,j));

		if (index == 0)
			rightRegion.at<uchar>(Point(i,j)) = 6;
		else if (index == 255)
			rightRegion.at<uchar>(Point(i, j)) = 0;
	}
	
	//rightRegion = removeSmallBlobs(rightRegion, 120, Scalar::all(255));

	blur_window = blur_window % 2 == 0 ? blur_window + 1 : blur_window;
	medianBlur(rightRegion, rightRegion, blur_window);
	
	//obtainRegionInMat(rightRegion, (rightRect.col - rightRect.width), rightRect.col,
	//	(rightRect.row - rightRect.height), rightRect.row, Scalar::all(0));

	if (mode == 0)
		imshow("Transducer Hole", src*0.5f + rightRegion*255*0.5f);
	rightRegion.copyTo(dst);
}

void obtainTransductor(Mat src, Mat &dst, Mat imgGray)
{
	Mat temp, water_marker;
	int area_to_remove = 3;

	int erosion_size = 4;

	//cvtColor(src, srcGray, COLOR_BGR2GRAY);

	obtainRegionInMat(src, 80, src.cols, Scalar(255, 255, 255));
	temp = src;
	dst = Mat::zeros(dst.size(), dst.type());
	water_marker = Mat::zeros(dst.size(), dst.type());

	//srcGray(Range::all(), Range(75, srcGray.cols));

	//for (int row = 0; row < temp.rows; row++)
	//for (int col = 0; col < temp.cols; col++)
	//{
	//	if (row <= area_to_remove || row >= src.rows - area_to_remove ||
	//		col <= area_to_remove || col >= src.cols - area_to_remove)
	//		src.at<uchar>(row, col) = 255;
	//}

	threshold(src, src, trans_window, 255, THRESH_BINARY_INV);
	
	blur_window = blur_window % 2 == 0 ? blur_window + 1 : blur_window;
	medianBlur(src, src, blur_window);

	int i, j, compCount = 0, idx;
	vector<vector<Point>> contours, endContours;
	vector<Vec4i> hierarchy;
	
	// Checks to see if any contours are found in the mask. 
	findContours(src, endContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	//for (int i = 0; i < contours.size(); i++)
	//{
	//	//double area = contours[i].size();
	//		//contourArea(contours[i]);
	//	//cout << area << endl;
	//	//if (area > contour_area)
	//		endContours.push_back(contours[i]);
	//}
	//cout << endl;

	vector<vector<Point> > contours_poly(endContours.size());

	for (int i = 0; i < endContours.size(); i++)
		approxPolyDP(Mat(endContours[i]), contours_poly[i], 0, true);

	//for (idx = 0; idx >= 0; idx = hierarchy[idx][0])
	for (int i = 0; i< endContours.size(); i++)
	{
		//drawContours(dst, contours, idx, color, 1, 8, hierarchy);
		drawContours(dst, contours_poly, i, Scalar::all(5), 1, 8);
		drawContours(water_marker, contours_poly, i, Scalar::all(255), CV_FILLED, 8);
	}

	imshow("Transducer", dst*255*0.5f);

	threshold(water_marker, water_marker, 0, 2, THRESH_BINARY_INV);

	Mat water_marker_1;
	water_marker.convertTo(water_marker_1, CV_8UC3);
	imshow("water_marker_1", water_marker_1 * 255 * 0.5 + src*0.5);
	
	obtainRegionInMat(dst, 80 + area_to_remove, dst.cols - area_to_remove,
		area_to_remove, dst.rows - area_to_remove, Scalar::all(0));

	obtainRegionInMat(water_marker, 80 + area_to_remove, water_marker.cols -area_to_remove,
		area_to_remove, water_marker.rows-area_to_remove, Scalar::all(0));

	int erosion_type = MORPH_RECT;

	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	erode(water_marker, water_marker, element);
	erode(water_marker, water_marker, element);

	dst += water_marker;

	//FillHoles(dst);
	if (mode == 0)
	{
		imshow("Water", water_marker * 255 * 0.5 + imgGray * 0.5);
		imshow("Transducerš", dst * 255 * 0.5f + imgGray* 0.5f);
	}
	
}

void watershed_callback(string directory, string filename, int mode, Mat img0, Mat img, Mat imgGray, Mat markerMask)
{
	Mat floatMarkers;
	cvtColor(imgGray, imgGray, COLOR_GRAY2BGR);
	int i, j, compCount = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

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
	markerMask.convertTo(floatMarkers, CV_32F);

	// Paint the watershed image.
	//* colors for the regions: 1 - air, 2 - water, 3 - gel, 4 - tissue, 5 - transductor, 6 - transductor hole
	try
	{
		for (i = 0; i < markerMask.rows; i++)
		for (j = 0; j < markerMask.cols; j++)
		{
			int index = markerMask.at<int>(i, j);
			if (index == -1)
			{
				floatMarkers.at<float>(i, j) = 0.0f;
				wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else if (index <= 0)
			{
				floatMarkers.at<float>(i, j) = 0.0f;
				wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else
				wshed.at<Vec3b>(i, j) = colorTab[index - 1];

			if (index == 1)
				floatMarkers.at<float>(i, j) = 0.0f;
			if (index == 2)
				floatMarkers.at<float>(i, j) = 1.0f;
			if (index == 3)
				floatMarkers.at<float>(i, j) = 0.5f;
			if (index == 4)
				floatMarkers.at<float>(i, j) = 0.7f;
			if (index == 5)
				floatMarkers.at<float>(i, j) = 0.3f;
			if (index == 6)
				floatMarkers.at<float>(i, j) = 1.0f;
		}
	}
	catch (Exception e)
	{
		int test = 0;
	}
	

	// Add transparency in order to see the original image and the watershed areas. 
	wshed = wshed*0.5 + imgGray*0.5;
	
	if (mode == 0)
	{
		imshow("watershed transform", wshed);

		//wshed.convertTo(floatMarkers, CV_32F);
		
		vector<float> floatM;

		if (floatMarkers.isContinuous())
			floatM.assign((float*)floatMarkers.datastart, (float*)floatMarkers.dataend);
		else
			for (int i = 0; i < floatMarkers.rows; ++i) 
				floatM.insert(floatM.end(), (float*)floatMarkers.ptr<uchar>(i), (float*)floatMarkers.ptr<uchar>(i)+floatMarkers.cols);
			
		toMatlabMat<float>("../../data/allData/processedWatershed/Matlab_Mats/" + to_string(image_window), floatM.data(), 240,240);
	}   
    else if(mode == 1)
    {
		vector<float> floatM;

		if (floatMarkers.isContinuous())
			floatM.assign((float*)floatMarkers.datastart, (float*)floatMarkers.dataend);
		else
		for (int i = 0; i < floatMarkers.rows; ++i)
			floatM.insert(floatM.end(), (float*)floatMarkers.ptr<uchar>(i), (float*)floatMarkers.ptr<uchar>(i)+floatMarkers.cols);

		//toMatlabMat<float>("../../data/allData/processedWatershed/Matlab_Mats/" + directory + "-" + filename , floatM.data(), 240, 240);

    	imwrite("../../data/allData/processedWatershed/"+directory + "-" +filename+"_1_original.png", imgGray);
		imwrite("../../data/allData/processedWatershed/"+directory + "-" +filename+"_2_regions.png", wshed);
//		imwrite("../../../data/allData/processedWatershed/"+directory + "-" +filename+"_1_original.png", imgGray);
//		imwrite("../../../data/allData/processedWatershed/"+directory + "-" +filename+"_2_regions.png", wshed);
    }
	
}

void region_separation(Mat &markerMask, Mat imgGray)
{
	Mat leftRegion, middleRegion, rightRegion, regionBin, transductorMat, transductorHoleMat, tissueMat;
	matrixData leftRect, middleRect, rightRect;
	unsigned long time;

	//time = getTickCount();
	// Separates the region based on information given. The region is segmented at column 75.
	middleRegion = imgGray(Range::all(), Range(0,75));
	resizeCol(middleRegion, imgGray.cols - 75, Scalar(0,0,0));

	threshold( middleRegion, regionBin, thresh_window, 255, THRESH_BINARY );

	// Obtain the max rectangle in the binary image. 
	middleRect = maxRectInMat(regionBin);

	// Paint the obtained rectangle in the marker mask. 
	rectangle(markerMask, Point(middleRect.col-10, middleRect.row-10), Point(middleRect.col-middleRect.width+10,middleRect.row-middleRect.height+10), Scalar::all(3), 1);
	//ProccTimePrint(time, "Gel segmentation");

	Mat gelpad_mat;
	markerMask.convertTo(gelpad_mat, CV_8UC3);
	imshow("gelpad", gelpad_mat * 255 * 0.5 + imgGray*0.5);

	//time = getTickCount();
	tissueMat = Mat::zeros(imgGray.size(), imgGray.type());
	obtainTissue(imgGray, middleRect, tissueMat);
	tissueMat.convertTo(tissueMat, CV_32S);
	markerMask += tissueMat;
	//removeSmallBlobs(regionBin, 400);
	//regionBin = FillHoles(regionBin);
	
	//regionBin.convertTo(regionBin, CV_32S);
	//markerMask += regionBin;

	//imshow("tissue region", regionBin);

	//leftRect = maxRectInMat(regionBin);

	////if(leftRect.col < 90)
 //       rectangle(markerMask, Point(leftRect.col-3, leftRect.row-3), 
		//Point(leftRect.col-leftRect.width+3,leftRect.row-leftRect.height+3), Scalar::all(4), 1);
	//ProccTimePrint(time, "Tissue Segmentation");

	//time = getTickCount();
	obtainTransductorHole(imgGray, transductorHoleMat);
	transductorHoleMat.convertTo(transductorHoleMat, CV_32S);
	markerMask += transductorHoleMat;
	//ProccTimePrint(time, "Transductor hole");
    
	imgGray.copyTo(rightRegion);

    //obtainRegionInMat(rightRegion, 120, imgGray.cols, 0, imgGray.rows, Scalar(255,255,255));
    //threshold( rightRegion, regionBin, thresh_window, 255, THRESH_BINARY_INV );
    //regionBin = FillHoles(regionBin);

    //rightRect= maxRectInMat(regionBin);

    //rectangle(markerMask, Point(rightRect.col, rightRect.row), Point(rightRect.col-rightRect.width,rightRect.row-rightRect.height), Scalar(255,0,0), 1);

	transductorMat = Mat(rightRegion.size(), rightRegion.type());
	//time = getTickCount();
	obtainTransductor(rightRegion, transductorMat, imgGray);
	//ProccTimePrint(time, "Transductor");

	transductorMat.convertTo(transductorMat, CV_32S);
	markerMask += transductorMat;

	// These  points were added to mask the background of the image. 
    //rectangle(markerMask, Point(10, 10), Point(11,11), Scalar::all(1), 1); 	// The upper left corner, to obtain the left background
	//rectangle(markerMask, Point(90, 125), Point(90, 125), Scalar::all(2), 1); 	// The middle of the image, to obtain the right background
	//rectangle(markerMask, Point(90, 3), Point(237, 237), Scalar::all(2), 1);	// A rectangle surrounding the right background, to eliminate 																	additional noise

	if (mode == 0)
	{
		Mat markers;
		markerMask.convertTo(markers, CV_8UC3);
		imshow("markers", markers*1000 * 0.5f + imgGray*0.5);
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

	imshow("Original", img0);
	imshow("Blurred", img);
	
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	//cvtColor(img, markerMask, COLOR_BGR2GRAY);

	markerMask = Mat::zeros(img0.size(), CV_32S);

	//AAtimeCpu = getTickCount();
	region_separation(markerMask,imgGray);
	//ProccTimePrint(AAtimeCpu, "Marker generation");

	//AAtimeCpu = getTickCount();
	watershed_callback(directory, filename, mode, img0, img, imgGray, markerMask);
	//ProccTimePrint(AAtimeCpu, "Watershed segmentation");

	//cout << "-------------------------------------" << endl;
}

void setUp(void)
{
	createNames(fileNames);
	namedWindow( "image", WINDOW_NORMAL );
	createTrackbar("Image", "image",&image_window,47,image_callback);
	createTrackbar("Threshold", "image",&thresh_window,50,image_callback);
	createTrackbar("Trans Threshold", "image", &trans_window, 30, image_callback);
	createTrackbar("Blur Threshold", "image", &blur_window, 10, image_callback);
	createTrackbar("Water Threshold", "image", &water_window, 50, image_callback);
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
				if (img0.empty())
				{
					cout << "Couldn't open image " << directory << "/" << cvsContents[i][j] << endl;
					continue;
				}
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

