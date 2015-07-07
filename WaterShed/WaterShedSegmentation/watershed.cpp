#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <utilityFunctions.h>
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

vector<string> 	fileNames;
Mat 			markerMask, img, imgGray, img0;
Point 			prevPt(-1, -1);
int 			image_window = 35, thresh_window = 15;
string          src_file;
int             mode = 0;
unsigned long   AAtime=0, AAtimeCpu = 0;

void setUp(void);
void watershed_callback(string directory, string filename, int mode);
void image_callback(int, void*);
void region_separation(void);
void singleImage(void);
void allImages(void);
void processImage(string directory, string filename, int mode);

void watershed_callback(string directory, string filename, int mode)
{
	cvtColor(imgGray, imgGray, COLOR_GRAY2BGR);
	int i, j, compCount = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(markerMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if( contours.empty() )
		return;
	Mat markers(markerMask.size(), CV_32S);
	markers = Scalar::all(0);
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
//	printf( "watershed execution time = %gms\n", t*1000./getTickFrequency() );

	Mat wshed(markers.size(), CV_8UC3);

	// paint the watershed image
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

	wshed = wshed*0.5 + imgGray*0.5;
	if(mode == 0)
        imshow( "watershed transform", wshed );
    else if(mode == 1)
    {
		imwrite("../../../data/allData/processedWatershed/"+directory + "-" +filename+"_1_original.png", imgGray);
		imwrite("../../../data/allData/processedWatershed/"+directory + "-" +filename+"_2_regions.png", wshed);
    }
}

void region_separation(void)
{
	Mat leftRegion, middleRegion, rightRegion, regionBin;
	matrixData leftRect, middleRect, rightRect;

	middleRegion = imgGray(Range::all(), Range(0,75));
	resizeCol(middleRegion, imgGray.cols - 75, Scalar(0,0,0));

	threshold( middleRegion, regionBin, thresh_window, 255, THRESH_BINARY );

	middleRect = maxRectInMat(regionBin);

	rectangle(markerMask, Point(middleRect.col-10, middleRect.row-10), Point(middleRect.col-middleRect.width+10,middleRect.row-middleRect.height+10), Scalar(255,0,0), 1);

	leftRegion = imgGray(Range::all(), Range(0,middleRect.col - middleRect.width));
	resizeCol(leftRegion, imgGray.cols - (middleRect.col - middleRect.width), Scalar(0,0,0));

	threshold( leftRegion, regionBin, thresh_window, 255, THRESH_BINARY );
    regionBin = removeSmallBlobs(regionBin, 400);
    regionBin = FillHoles(regionBin);

    line(regionBin, Point(0,0), Point(0,regionBin.rows), Scalar(0,0,0), 1);

	leftRect = maxRectInMat(regionBin);

	if(leftRect.col < 90)
        rectangle(markerMask, Point(leftRect.col-3, leftRect.row-3), Point(leftRect.col-leftRect.width+3,leftRect.row-leftRect.height+3), Scalar(255,0,0), 1);

    imgGray.copyTo(rightRegion);

    obtainRegionInMat(rightRegion, 120, imgGray.cols, 0, imgGray.rows, Scalar(255,255,255));
    threshold( rightRegion, regionBin, thresh_window, 255, THRESH_BINARY_INV );
    regionBin = FillHoles(regionBin);

    rightRect= maxRectInMat(regionBin);

    rectangle(markerMask, Point(rightRect.col, rightRect.row), Point(rightRect.col-rightRect.width,rightRect.row-rightRect.height), Scalar(255,0,0), 1);

    rectangle(markerMask, Point(10, 10), Point(11,11), Scalar(255,0,0), 1);
    rectangle(markerMask, Point(90,125), Point(90,125), Scalar(255,0,0), 1);
    rectangle(markerMask, Point(90,10), Point(230,230), Scalar(255,0,0), 1);

    Mat wshed(markerMask.size(), CV_8UC3);
    wshed = markerMask*0.5 + imgGray*0.5;

	if(mode == 0)
        imshow( "1", wshed );
}

void image_callback(int, void*)
{
    src_file = "../../../data/TestData/"+fileNames.at(image_window);
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

//	double t = (double)getTickCount();
	region_separation();

	watershed_callback(directory, filename, mode);

//	t = (double)getTickCount() - t;
//	printf( "All execution time = %gms\n", t*1000./getTickFrequency() )
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

	vector<vector<string>> cvsContents = getCsvContent("../../../data/allData/surveyFilesToProcess.csv");
	vector<string> filenames;

	for(int i = 0; i < cvsContents.size(); i++)
	{
		filenames.clear();
		string directory = cvsContents[i][0];

		for(int j = 1; j< cvsContents[i].size(); j++)
		{
			img0 = imread("../../../data/allData/originals/"+directory+"/"+cvsContents[i][j]+".png");
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
