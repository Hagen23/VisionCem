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

#include "misc_procedures.h"
#include "leftRegion.h"
#include "rightRegion.h"
#include "tissue.h"

using namespace cv;
using namespace gpu;
using namespace std;

Mat src, leftRegion, rightRegion, tissue_mat;

vector<string> fileNames;

int image_window = 24;

static void imageSwitching(int, void*)
{
	unsigned long AAtime=0, AAtimeCpu = 0;
	AAtimeCpu = getTickCount();
	src = imread("../data/"+fileNames.at(image_window));

	leftRegion = obtainLeftRegion(src);
	rightRegion = obtainRightRegion(src);
	tissue_mat = obtainTissue(src, leftRegion);

	ProccTimePrint(AAtimeCpu , "cpu");

  imshow( "Source", src );
	imshow("Left", leftRegion);
	imshow("Right", rightRegion);
	imshow("Tissue", tissue_mat);
}

void createNames(vector<string> & input)
{
	for(int i = 1; i<= 48; i++)
		input.push_back(to_string(i)+".png");
}

/** @function main */
int main( int argc, char** argv )
{
	createNames(fileNames);
  /// Create Window
  namedWindow( "Source", 1);
	createTrackbar("Image", "Source",&image_window ,47,imageSwitching);
  imageSwitching( 0, 0 );

  waitKey(0);
  return(0);
}

