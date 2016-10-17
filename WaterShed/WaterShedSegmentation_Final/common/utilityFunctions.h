/**
*	Author:	Octavio Navarro Hinojosa
*	Date:	June 2015
*	
*	Utility functions for the watershed segmentation.
*/

#ifndef UTILITY
#define UTILITY

#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stack>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

/**
* A class that stores the rectangle data.
*/
struct matrixData
{
	int area, row, col, width, height;

	matrixData()
	{
		area = 0;
		row = 0;
		col = 0;
		width = 0;
		height = 0;
	}

	void copyFrom(matrixData input)
	{
		area = input.area;
		width = input.width;
		height = input.height;
	}

	void printData(void)
	{
		printf("Area %d row %d col %d width %d, height %d \n\n",area, row, col, width, height);
	}
};

/**
* Creates and stores filenames based on the information of the images' filenames.
* @param input The vector that stores the created filenames.
*/
void createNames(vector<string> & input)
{
	for(int i = 1; i<= 48; i++)
		input.push_back(to_string(i)+".png");
}

/**
* Segments a region of an image.
* @param m The image to be segmented. The final regmentation is also stored here.
* @param beginCol The column index from which to start segmentation.
* @param beginRow The row index from which to start segmentation.
* @param endCol The column index that ends the segmentation.
* @param endRow The row index that ends the segmentation.
* @param s The color to fill the rest of the image with.
*/
void obtainRegionInMat(Mat& m, int beginCol, int endCol, int beginRow, int endRow, const Scalar s)
{
	Mat mask = cvCreateMat(m.rows, m.cols, m.type());
	mask.setTo(Scalar(0,0,0));

	for(int i=beginCol; i<endCol; i++)
		 for(int j=beginRow; j<endRow; j++)
		     mask.at<uchar>(Point(i,j)) = 255;

	Mat temporaryImg(m.rows, m.cols, m.type());
	temporaryImg.setTo(s);
	m.copyTo(temporaryImg,mask);

	m = temporaryImg;
}

/**
* Segments a region of an image based just on column indexes.
* @param m The image to be segmented. The final regmentation is also stored here.
* @param beginCol The column index from which to start segmentation.
* @param endCol The column index that ends the segmentation.
* @param s The color to fill the rest of the image with.
*/
void obtainRegionInMat(Mat& m, int beginCol, int endCol, const Scalar s)
{
	Mat mask = cvCreateMat(m.rows, m.cols, m.type());
	mask.setTo(Scalar(0,0,0));

	for(int i=beginCol; i<endCol; i++)
		 for(int j=0; j<mask.rows; j++)
		     mask.at<uchar>(Point(i,j)) = 255;

	Mat temporaryImg(m.rows, m.cols, m.type());
	temporaryImg.setTo(s);
	m.copyTo(temporaryImg,mask);

	m = temporaryImg;
}

/**
* Adds additional columns to an image. 
* @param m The image to be added columns.
* @param sz How many columns to add.
* @param s The color to fill the added columns of the image with.
*/
void resizeCol(Mat& m, size_t sz, const Scalar& s)
{
    Mat tm(m.rows, m.cols + sz, m.type());
    tm.setTo(s);
    m.copyTo(tm(Rect(Point(0, 0), m.size())));
    m = tm;
}
/**
* Prints how much time had passed since a determined moment.
* @param Atime The time from which to determine how long it passed. 
* @param msg A message to print.
*/
void ProccTimePrint( unsigned long Atime , string msg)
{
 unsigned long Btime=0;
 float sec, fps;
 Btime = getTickCount();
 sec = (Btime - Atime)/getTickFrequency();
 fps = 1/sec;
 printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(),  sec, fps );
}

/** 
* Processess a cvs file and saves the contents on a vector.
* @param filename The filename of the cvs file to process.
*/
vector<vector<string>> getCsvContent(string filename)
{
	ifstream file ( filename ,  ios::in);
	string value, value_aux;
	vector<vector<string>> cvsContents;

	while (!file.eof() )
	{
		vector<string> line;
		size_t position0 = 0, position1 = 0;
		getline(file, value);
		while(position1 !=std::string::npos)
		{
			position1 = value.find(",", position1+1);
			value_aux = value.substr(position0, position1-position0);
			if(!value_aux.empty())
				line.push_back(value_aux);
			position0 = position1+1;
		}
		if (!line.empty())
			cvsContents.push_back(line);
	}
	return cvsContents;
}

/** 
* Removes blobs in an image.
* @param m the image from which to remove blobs.
* @param maxBlobArea Blobs that have this area, or more, are filled. 
* @param s Color to fill the blobs with.
* @return drawing A Mat with the blobs removed.
*/
void removeSmallBlobs(Mat &m, float maxBlobArea, Scalar s = Scalar(255, 255, 255))
{
	vector<vector<Point> > contours, endContours;
	vector<Vec4i> hierarchy;

	findContours( m, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	for( int i = 0; i < contours.size(); i++ )
	{
		double area = contourArea(contours[i]);
		if(area > maxBlobArea)
			endContours.push_back(contours[i]);
	}

	vector<vector<Point> > contours_poly( endContours.size() );

	for( int i = 0; i < endContours.size(); i++ )
		approxPolyDP( Mat(endContours[i]), contours_poly[i], 0, true );

	m = Mat::zeros( m.size(), m.type() );

	for( int i = 0; i< endContours.size(); i++ )
	{
		Scalar color = s;
		drawContours( m, contours_poly, i, color, 1, 8 );
	}
}

/**
* Finds the data of largest area of an histogram of an image.
* I modified the original code found here: http://tech-queries.blogspot.mx/2011/09/find-largest-sub-matrix-with-all-1s-not.html
* @param arr The input image data. Flat bidimentional array.
* @param index The row index. Calculate maximum area in S for each row.
* @param colSize The matrix column size.
* @result outputData MatrixData with the location of the largest area histogram.
*/
matrixData largestArea(unsigned char *arr, int index, int colSize)
{
	int *area = new int[colSize]; 
	int  i, t, maxWidth, maxHeight;
	matrixData outputData;
	stack<int> St;  

	for (i=0; i<colSize; i++)
	{
		while (!St.empty())
		{
			if(arr[index+i] <= arr[index+St.top()])
				St.pop();
			else
				break;
		}
		if(St.empty())
			t = -1;
		else
			t = St.top();
		area[i] = i - t - 1;
		St.push(i);
	}

	while (!St.empty())
	St.pop();

	for (i=colSize-1; i>=0; i--)
	{
		while (!St.empty())
		{
			if(arr[index+i] <= arr[index + St.top()])
				St.pop();
			else
				break;
		}
		if(St.empty())
			t = colSize;
		else
			t = St.top();
			
		//calculating Ri, after this step area[i] = Li + Ri
		area[i] += t - i -1;
		St.push(i);
	}

	int max = 0;
	//Calculating Area[i] and find max Area
	for (i=0; i<colSize; i++)
	{
		int width = area[i] +1, height = arr[index+i];
		area[i] = arr[index+i] * (area[i] + 1);
		if (area[i] > max)
		{
			outputData.area = max = area[i];
			//outputData.col = maxCol = i;
			outputData.width = maxWidth = width;
			outputData.height = maxHeight = height;
		}
	}

//	outputData.printData();
	return outputData;
}

/**
* Trasposes a matrix of chars.
* @param A the initial matrix.
* @param result The trasposed matrix.
* @param rowSize The matrix row size.
* @param colSize The matrix columns size.
*/
void trasposeMatrix(unsigned char *A, unsigned char *result, int rowSize, int colSize)
{
	for(int i = 0; i < rowSize; i++)
	{
		for(int j = 0; j < colSize; j++)
		{
			result[j*rowSize+i] = A[i*rowSize + j];
		}
	}
}

/**
* Looks for the largest rectangle in an image based on the histogram of  said image.
* I modified the original code found here: http://tech-queries.blogspot.mx/2011/09/find-largest-sub-matrix-with-all-1s-not.html . 
* I consider a trasposed matrix in order to obtain the rectangle's row and column point.
* @param input The image data. This is used to obtain the histogram of columns.
* @param input_trasposed The trasposed image data. This is needed to obtain the histogram of the rows. 
* @param rowSize The image row size.
* @param colSize The image column size.
* @return A matrixData object with the information of the largest rectangle.
*/
matrixData find_max_matrix(unsigned char *input, unsigned char *input_trasposed, int rowSize, int colSize)
{
 matrixData finalData, dataRow, dataCol;
 int maxRow, maxCol, cur_maxRow, cur_maxCol;
 cur_maxRow = 0, cur_maxCol = 0;
 unsigned char *A, *B;

 A = new unsigned char[rowSize * colSize];
 B = new unsigned char[rowSize * colSize];

 memcpy ( A, input, rowSize*colSize );
 memcpy ( B, input_trasposed, rowSize*colSize );
 
 //Calculate Auxilary matrix histograms
 for (int i=1; i<rowSize; i++)
     for(int j=0; j<colSize; j++)
     {
         if(A[i*rowSize+j] == (unsigned char)255)
             A[i*rowSize+j] = A[(i-1)*rowSize+j] + 1;
         if(B[i*rowSize+j] == (unsigned char)255)
             B[i*rowSize+j] = B[(i-1)*rowSize+j] + 1;
     }

 //Calculate maximum area in S for each row
 for (int i=0; i<rowSize; i++)
 {
     dataRow = largestArea(A, i*rowSize, colSize);
     dataCol = largestArea(B, i*rowSize, colSize);
     maxRow = dataRow.area;
     maxCol = dataCol.area;

     if (maxRow > cur_maxRow)
     {
         cur_maxRow = maxRow;
         finalData.copyFrom(dataRow);
         finalData.row = i;
     }

     if (maxCol >= cur_maxCol)
     {
         cur_maxCol = maxCol;
         finalData.col = i;
     }
 }
 return finalData;
}

/**
* Obtains the largest area rectangle in a binary image.
* @param Image The binary image to process.
* @return The rectangle information in a matrixData object. 
*/
matrixData maxRectInMat(Mat& Image)
{
    int channels = Image.channels();

    int nRows = Image.rows;
    int nCols = Image.cols * channels;

	unsigned char *image_data, *image_data_trasposed;
	image_data_trasposed = new unsigned char[nRows*nCols];

	image_data = Image.ptr();

	if(Image.isContinuous())
	{
		trasposeMatrix(image_data, image_data_trasposed, nRows, nCols);
		return find_max_matrix(image_data, image_data_trasposed, nRows, nCols);
	}
	return matrixData();
}

/**
* Fills the holes in an image.
* @param _src The image to be processed.
* @return dst The image with its holes filled.
*/
Mat FillHoles(Mat _src, Scalar color = Scalar(255))
{
    CV_Assert(_src.type()==CV_8UC1);
    Mat dst;
    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(_src,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
    //Scalar color=Scalar(255);
    dst=Mat::zeros(_src.size(),CV_8UC1);

    for(int i=0;i<contours.size();i++)
    {
        drawContours(dst,contours,i,color,-1,8,hierarchy,0,cv::Point());
    }
    return dst;
}

/**
* Deprecated
*/
void removeAllSmallerBlobs(Mat& m, float minArea)
{
	float maxBlobArea = 0.0, secondMaxArea = 0.0;
	int 	maxBlobIndex = 0, secondIndex = 0;

	Mat m_in(m);

	vector<vector<Point> > contours, endContours;
  vector<Vec4i> hierarchy;

	findContours( m, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	cout << "Found " << contours.size() << " blobs " << endl;

	for( int i = 0; i < contours.size(); i++ )
	{
			float area = contourArea(contours[i]);
			if(area > minArea)
			{
				if(area > maxBlobArea)
				{
					maxBlobArea = area;
					maxBlobIndex = i;
				}
				else
					if(area > secondMaxArea)
					{
						secondMaxArea = area;
						secondIndex = i;
					}
			}
			cout << "blob " << i << " area " << area << endl;
	}

	cout << "secondIndex " << secondIndex << " secondMaxArea " << secondMaxArea << endl;
	endContours.push_back(contours[secondIndex]);

	vector<vector<Point> > contours_poly( endContours.size() );

  for( int i = 0; i < endContours.size(); i++ )
		approxPolyDP( Mat(endContours[i]), contours_poly[i], 0, true );

  m = Mat::zeros( m.size(), CV_8UC3 );

  for( int i = 0; i< endContours.size(); i++ )
	{
		Scalar color = Scalar( 255,255,255 );
		drawContours( m, contours_poly, i, color, CV_FILLED, 8 );
	}
}

/**
* Converts an image data array to a matlab mat file
* @param filename The filename where the mat file will be stored
* @param data The data array to be transformed. Templated to allow different data types
* @param width The width of the data array
* @param height The height of the data array
*/
template <class T>
void toMatlabMat(string filename, T *data, int width, int height)
{
	ofstream file(filename + ".txt", ofstream::out | ofstream::trunc);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
			file << data[j*height + i] << " ";
		
		file << "\n";
	}
	file << flush;
	file.close();
}
#endif
