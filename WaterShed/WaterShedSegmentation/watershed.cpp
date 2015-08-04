/**
*	Author:	Octavio Navarro Hinojosa
*	Date:	June 2015
*	
*	Watershed segmentation using dynamically generated masks. 
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <utilityFunctions.h>
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

vector<string> 	fileNames;			/** Vector that contains the filenames to process */  
Mat 				markerMask, 			/** Mat that stores the generated mask */			
				img, img0,
				imgGray;		
Point 			prevPt(-1, -1);
int 				image_window = 35, 		/** Variable that controls the image selection trackbar */
				thresh_window = 15;		/** Variable that controls the binary threshold selection trackbar */
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
void watershed_callback(string directory, string filename, int mode);

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
void region_separation(void);

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
void processImage(string directory, string filename, int mode);

void watershed_callback(string directory, string filename, int mode)
{
	cvtColor(imgGray, imgGray, COLOR_GRAY2BGR);
	int i, j, compCount = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Checks to see if any contours are found in the mask. 
	findContours(markerMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if( contours.empty() )
		return;
		
	Mat markers(markerMask.size(), CV_32S);
	markers = Scalar::all(0);
	
	// Generate the marker mask to be used by the watershed algorithm. It is based on the rectangular masks obtained previously. 
	int idx = 0;
	for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
		drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

	if( compCount == 0 )
		return;

	vector<Vec3b> colorTab;
	for( i = 0; i < compCount; i++ )
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);

		colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	double t = (double)getTickCount();
	watershed( img0, markers );
	t = (double)getTickCount() - t;

	Mat wshed(markers.size(), CV_8UC3);

	// Paint the watershed image.
	for( i = 0; i < markers.rows; i++ )
		for( j = 0; j < markers.cols; j++ )
		{
			int index = markers.at<int>(i,j);
			if( index == -1 )
				wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
			else if( index <= 0 || index > compCount )
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

void region_separation(void)
{
	Mat leftRegion, middleRegion, rightRegion, regionBin;
	matrixData leftRect, middleRect, rightRect;

	// Separates the region based on information given. The region is segmented at column 75.
	middleRegion = imgGray(Range::all(), Range(0,75));
	resizeCol(middleRegion, imgGray.cols - 75, Scalar(0,0,0));

	threshold( middleRegion, regionBin, thresh_window, 255, THRESH_BINARY );

	// Obtain the max rectangle in the binary image. 
	middleRect = maxRectInMat(regionBin);

	// Paint the obtained rectangle in the marker mask. 
	rectangle(markerMask, Point(middleRect.col-10, middleRect.row-10), Point(middleRect.col-middleRect.width+10,middleRect.row-middleRect.height+10), Scalar(255,0,0), 1);

	leftRegion = imgGray(Range::all(), Range(0,middleRect.col - middleRect.width));
	resizeCol(leftRegion, imgGray.cols - (middleRect.col - middleRect.width), Scalar(0,0,0));

	threshold( leftRegion, regionBin, thresh_window, 255, THRESH_BINARY );
    regionBin = removeSmallBlobs(regionBin, 400);
    regionBin = FillHoles(regionBin);

	leftRect = maxRectInMat(regionBin);

	if(leftRect.col < 90)
        rectangle(markerMask, Point(leftRect.col-3, leftRect.row-3), Point(leftRect.col-leftRect.width+3,leftRect.row-leftRect.height+3), Scalar(255,0,0), 1);

    imgGray.copyTo(rightRegion);

    obtainRegionInMat(rightRegion, 120, imgGray.cols, 0, imgGray.rows, Scalar(255,255,255));
    threshold( rightRegion, regionBin, thresh_window, 255, THRESH_BINARY_INV );
    regionBin = FillHoles(regionBin);

    rightRect= maxRectInMat(regionBin);

    rectangle(markerMask, Point(rightRect.col, rightRect.row), Point(rightRect.col-rightRect.width,rightRect.row-rightRect.height), Scalar(255,0,0), 1);

	// These  points were added to mask the background of the image. 
    rectangle(markerMask, Point(10, 10), Point(11,11), Scalar(255,0,0), 1); 	// The upper left corner, to obtain the left background
    rectangle(markerMask, Point(90,125), Point(90,125), Scalar(255,0,0), 1); 	// The middle of the image, to obtain the right background
    rectangle(markerMask, Point(90,10), Point(230,230), Scalar(255,0,0), 1);	// A rectangle surrounding the right background, to eliminate 																	additional noise

    Mat wshed(markerMask.size(), CV_8UC3);
    wshed = markerMask*0.5 + imgGray*0.5;

	if(mode == 0)
        imshow( "1", wshed );
}

void image_callback(int, void*)
{
	// Fixed path to the test images. Can be changed as needed. 
    src_file = "../../data/TestData/"+fileNames.at(image_window);
    
    // This line was added just to have a different path to the files. Can be deleted as needed. 
//    src_file = "../../../data/TestData/"+fileNames.at(image_window);
	img0 = imread(src_file, 1);

	if( img0.empty() )
	{
		cout << "Couldn't open image " << src_file << ". Usage: watershed <image_name>\n";
		return;
	}

	processImage("", "", mode);

    if(mode == 0)
    {
        imshow( "image", img );
    }
}

void processImage(string directory, string filename, int mode)
{
    img0.copyTo(img);
	blur(img, img, Size(3,3));

	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	cvtColor(img, markerMask, COLOR_BGR2GRAY);

	markerMask = Scalar::all(0);

	region_separation();

	watershed_callback(directory, filename, mode);
}

void setUp(void)
{
	createNames(fileNames);
	namedWindow( "image", 1 );
	createTrackbar("Image", "image",&image_window,47,image_callback);
	createTrackbar("Threshold", "image",&thresh_window,50,image_callback);
}

void allImages(void)
{
	createNames(fileNames);

	AAtimeCpu = getTickCount();

	cout << "STARTED" << endl;

	vector<vector<string>> cvsContents = getCsvContent("../../data/allData/surveyFilesToProcess.csv");
//	vector<vector<string>> cvsContents = getCsvContent("../../../data/allData/surveyFilesToProcess.csv");
	vector<string> filenames;

	for(int i = 0; i < cvsContents.size(); i++)
	{
		filenames.clear();
		string directory = cvsContents[i][0];

		for(int j = 1; j< cvsContents[i].size(); j++)
		{
			img0 = imread("../../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png");
//			img0 = imread("../../../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png");
			processImage(directory, cvsContents[i][j], mode);
		}
	}

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
