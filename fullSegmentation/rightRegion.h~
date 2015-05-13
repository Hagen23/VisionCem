#ifndef right_region
#define right_region

#include "misc_procedures.h"

Mat obtainRightRegion(Mat img)
{
	int 	spatial_window_tissue = 8;
	int 	color_window_tissue = 20;
	int 	cluster_window_tissue = 340;
	int 	binary_threshold_tissue = 120;

	int 	contrast_threshold_tissue = 50;
	int 	brightness_threshold_tissue = 50;
	int 	regionSeparator = 100;

	Mat 					img_local, rightRegion, mSFilteringImgHost, gris_mSFilteringImgHost, bin_mSFilteringImgHost;
	gpu::GpuMat 	pimgGpu, imgGpu, mSFilteringImgGPU;
	
	fastNlMeansDenoising(img,img_local, 15);
	
	adjustBrightnessContrastV3(img_local, contrast_threshold_tissue*0.1, brightness_threshold_tissue);

	pimgGpu.upload(img_local);

	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);

	// For transductor
	gpu::meanShiftFiltering(imgGpu, mSFilteringImgGPU, spatial_window_tissue, color_window_tissue);

	mSFilteringImgGPU.download(mSFilteringImgHost);

	cvtColor( mSFilteringImgHost, gris_mSFilteringImgHost, COLOR_RGB2GRAY );
 threshold( gris_mSFilteringImgHost, bin_mSFilteringImgHost, binary_threshold_tissue, 255,  CV_THRESH_BINARY ); 

	rightRegion = bin_mSFilteringImgHost(Range(10,bin_mSFilteringImgHost.rows -10) , Range(regionSeparator, bin_mSFilteringImgHost.cols));
	resizeCol(rightRegion, regionSeparator, 20, Scalar(255,255,255), Point(regionSeparator,10));

	morph(rightRegion);
	
	return rightRegion;
}

#endif
