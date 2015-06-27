#ifndef UTILITY
#define UTILITY

#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <stack>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

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

void createNames(vector<string> & input)
{
	for(int i = 1; i<= 48; i++)
		input.push_back(to_string(i)+".png");
}

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

void resizeCol(Mat& m, size_t sz, const Scalar& s)
{
    Mat tm(m.rows, m.cols + sz, m.type());
    tm.setTo(s);
    m.copyTo(tm(Rect(Point(0, 0), m.size())));
    m = tm;
}

void ProccTimePrint( unsigned long Atime , string msg)   
{   
 unsigned long Btime=0;   
 float sec, fps;   
 Btime = getTickCount();   
 sec = (Btime - Atime)/getTickFrequency();   
 fps = 1/sec;   
 printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(),  sec, fps );   
} 

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

Mat removeSmallBlobs(Mat m, float maxBlobArea)
{
	Mat drawing;

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

  drawing = Mat::zeros( m.size(), CV_8UC3 );

  for( int i = 0; i< endContours.size(); i++ )
	{
		Scalar color = Scalar( 255,255,255 );
		drawContours( drawing, contours_poly, i, color, CV_FILLED, 8 );
	}

	return drawing;
}

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

matrixData largestArea(unsigned char *arr, int index, int colSize)  
{  
	int *area = new int[colSize]; //initialize it to 0  
	int n, i, t, maxCol, maxWidth, maxHeight;
	matrixData outputData;
	stack<int> St;  //include stack for using this #include<stack>  
	bool done;  

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
		//Calculating Li  
		area[i] = i - t - 1;  
		St.push(i);  
	}  

	//clearing stack for finding Ri  
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
 
 //finalData.printData();
 return finalData;  
}  

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
#endif
