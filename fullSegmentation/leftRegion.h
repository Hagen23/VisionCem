#ifndef left_region
#define left_region

#include "misc_procedures.h"

Mat obtainLeftRegion(Mat img)
{
	int 					spatial_window = 10;
	int 					color_window = 26;
	int 					cluster_size = 2700;
	int 					binary_threshold = 90;
	int 					contrast_threshold = 50;
	int 					brightness_threshold = 15;

	Mat 					img_local, leftRegion, gris_mSFilteringImgHost, mSSegRegionsImgHost;
	gpu::GpuMat 	pimgGpu, imgGpu;

	//blur(img, img_local, Size(5,5), Point(-1,-1));
	fastNlMeansDenoising(img,img_local, 20);

	cvtColor( img_local, gris_mSFilteringImgHost, COLOR_RGB2GRAY );

	leftRegion = img_local(Range::all(), Range(0,65));

	resizeCol(leftRegion, img.cols - 65, 0, Scalar(150,150,150));
	adjustBrightnessContrast(leftRegion, contrast_threshold*0.1, brightness_threshold);
	
	pimgGpu.upload(leftRegion);

	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);

	gpu::meanShiftSegmentation(imgGpu, mSSegRegionsImgHost, spatial_window,color_window, cluster_size);
	
	cvtColor( mSSegRegionsImgHost, mSSegRegionsImgHost, COLOR_RGB2GRAY );
	threshold( mSSegRegionsImgHost, mSSegRegionsImgHost, binary_threshold, 255,  CV_THRESH_BINARY); 

	return mSSegRegionsImgHost;
}

#endif
