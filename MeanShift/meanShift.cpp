//#include <time.h>   
//#include <opencv2\opencv.hpp>   
//#include <opencv2\gpu\gpu.hpp>   
//#include <string>   
//#include <stdio.h>   

#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include <cv.h>
#include <highgui.h>
#include <imgproc/imgproc.hpp>
#include <gpu/gpu.hpp>

using namespace cv;
using namespace std;

void ProccTimePrint( unsigned long Atime , string msg)   
{   
 unsigned long Btime=0;   
 float sec, fps;   
 Btime = getTickCount();   
 sec = (Btime - Atime)/getTickFrequency();   
 fps = 1/sec;   
 printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(),  sec, fps );   
} 


int main(int argc, char** argv)
{
 unsigned long AAtime=0;
 Mat img;
 //image load
 if(argc > 1) 
 img = imread( argv[1], 1 );
 else
 img = imread("../data/15.png");
 
 Mat outImg, outimg2;

 //cpu version meanshift
 AAtime = getTickCount();
 pyrMeanShiftFiltering(img, outImg, 30, 30, 3);
 ProccTimePrint(AAtime , "cpu");


 //gpu version meanshift
 gpu::GpuMat pimgGpu, imgGpu, outImgGpu;
 AAtime = getTickCount();
 pimgGpu.upload(img);
 //gpu meanshift only support 8uc4 type.
 gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);
 gpu::meanShiftFiltering(imgGpu, outImgGpu, 30, 30);
 outImgGpu.download(outimg2);
 ProccTimePrint(AAtime , "gpu");

 //show image
 imshow("origin", img);
 imshow("MeanShift Filter cpu", outImg);
 imshow("MeanShift Filter gpu", outimg2);


 waitKey();

	return 0;
}



