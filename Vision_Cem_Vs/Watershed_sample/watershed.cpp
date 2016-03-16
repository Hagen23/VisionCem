#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
	cout << "\nThis program demonstrates the famous watershed segmentation algorithm in OpenCV: watershed()\n"
		"Usage:\n"
		"./watershed [image_name -- default is fruits.jpg]\n" << endl;


	cout << "Hot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tw or SPACE - run watershed segmentation algorithm\n"
		"\t\t(before running it, *roughly* mark the areas to segment on the image)\n"
		"\t  (before that, roughly outline several markers on the image)\n";
}
Mat markerMask, img;
Point prevPt(-1, -1);

Scalar colors[6] = 
{
	Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0,0,255),
	Scalar(255, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 255)
};

Scalar currentColor;

static void onMouse(int event, int x, int y, int flags, void*)
{
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == CV_EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(markerMask, prevPt, pt, currentColor, 5, 8, 0);
		line(img, prevPt, pt, currentColor, 5, 8, 0);
		prevPt = pt;
		imshow("image", img);
		imshow("markerMask", markerMask*1000);
	}
}

int main(int argc, char** argv)
{
	char* filename = argc >= 2 ? argv[1] : (char*)"flower.png";
	Mat img0 = imread(filename, 1), imgGray;

	if (img0.empty())
	{
		cout << "Couldn'g open image " << filename << ". Usage: watershed <image_name>\n";
		return 0;
	}
	help();
	namedWindow("image", WINDOW_NORMAL);
	namedWindow("watershed transform", WINDOW_NORMAL);
	namedWindow("markers", WINDOW_NORMAL);
	namedWindow("markerMask", WINDOW_NORMAL);

	img0.copyTo(img);
	cvtColor(img, imgGray, COLOR_BGR2GRAY);

	markerMask = Mat::zeros(img0.size(), CV_32SC1);
	//cvtColor(img, markerMask, COLOR_BGR2GRAY);
	//cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);

	imshow("image", img);
	setMouseCallback("image", onMouse, 0);

	currentColor = colors[0];

	for (;;)
	{
		int c = waitKey(0);

		if ((char)c == 27)
			break;

		if ((char)c == '1')
			currentColor = colors[0];

		if ((char)c == '2')
			currentColor = colors[1];

		if ((char)c == '3')
			currentColor = colors[2];

		if ((char)c == '4')
			currentColor = colors[3];

		if ((char)c == '5')
			currentColor = colors[4];

		if ((char)c == '6')
			currentColor = colors[05];


		if ((char)c == 'r')
		{
			markerMask = Scalar::all(0);
			img0.copyTo(img);
			imshow("image", img);
		}

		if ((char)c == 'w' || (char)c == ' ')
		{
			int i, j, compCount = 0;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;

			//findContours(markerMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

			//if (contours.empty())
			//	continue;
			//Mat markers(markerMask.size(), CV_32S);
			//markers = Scalar::all(0);
			//int idx = 0;
			//for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
			//{
			//	Scalar color = colors[idx % 6];
			//	drawContours(markers, contours, idx, color, -1, 8, hierarchy, INT_MAX);
			//}			

			//imshow("markers", markerMask);

			//if (compCount == 0)
			//	continue;

			//vector<Vec3b> colorTab;
			//for (i = 0; i < compCount; i++)
			//{
			//	int b = theRNG().uniform(0, 255);
			//	int g = theRNG().uniform(0, 255);
			//	int r = theRNG().uniform(0, 255);

			//	colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			//}

			//double t = (double)getTickCount();
			watershed(img0, markerMask);
			//t = (double)getTickCount() - t;
			//printf("execution time = %gms\n", t*1000. / getTickFrequency());

			Mat wshed(markerMask.size(), CV_8UC3);

			// paint the watershed image
			for (i = 0; i < markerMask.rows; i++)
			for (j = 0; j < markerMask.cols; j++)
			{
				int index = markerMask.at<int>(i, j);
				if (index == -1)
					wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				else if (index <= 0 || index > compCount)
					wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				else
					wshed.at<Scalar>(i, j) = colors[index % 6];
			}

			Mat Temp;
			imgGray.convertTo(Temp, CV_8UC3);
			wshed = wshed*0.5 + imgGray*0.5;
			imshow("watershed transform", wshed);
		}
	}

	return 0;
}
