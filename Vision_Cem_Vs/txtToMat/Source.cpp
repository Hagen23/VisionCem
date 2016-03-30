#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mat.h>

using namespace std;

vector<vector<string>> getCsvContent(string filename)
{
	ifstream file(filename, ios::in);
	string value, value_aux;
	vector<vector<string>> cvsContents;

	while (!file.eof())
	{
		vector<string> line;
		size_t position0 = 0, position1 = 0;
		getline(file, value);
		while (position1 != std::string::npos)
		{
			position1 = value.find(",", position1 + 1);
			value_aux = value.substr(position0, position1 - position0);
			if (!value_aux.empty())
				line.push_back(value_aux);
			position0 = position1 + 1;
		}
		if (!line.empty())
			cvsContents.push_back(line);
	}
	return cvsContents;
}


void createNames(vector<string> & input)
{
	for (int i = 1; i <= 48; i++)
		input.push_back(to_string(i) + ".png");
}

template <class T>
void toMatlabMat(string filename, string variableName, T *data, int width, int height)
{
	MATFile *pmat;
	mxArray *pData;
	pmat = matOpen(filename.c_str(), "w");

	if (pmat != NULL)
	{
		pData = mxCreateDoubleMatrix(width, height, mxREAL);
		if (pData != NULL)
		{
			memcpy((void *)(mxGetPr(pData)), (void *)data, sizeof(T)*width*height);
			int status = matPutVariable(pmat, variableName.c_str(), pData);

			if (status == 0)
			{
				mxDestroyArray(pData);
				if (matClose(pmat) != 0)
					printf("Error closing file %s\n", filename);

			}
		}
	}
}

template <class T> T* setDataFromFile(string filename)
{
	size_t last = 0; size_t next = 0; 
	ifstream myfile(filename);
	static vector<T> floats;
	string line;
	
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			while ((next = line.find(" ", last)) != string::npos)
			{
				floats.push_back(stod(line.substr(last, next - last)));
				last = next + 1;
			}
			cout << line.substr(last) << endl;
			next = last = 0;
		}
		myfile.close();
	}

	T *data = floats.data();
	floats.clear(); 
	return data;
}

int main()
{
	string line;
	string filePath = "../../data/allData/processedWatershed/Matlab_Mats/";

	vector<string> fileNames;
	createNames(fileNames);

	vector<vector<string>> cvsContents = getCsvContent("../../data/allData/surveyFilesToProcess.csv");

	for (int i = 0; i < cvsContents.size(); i++)
	{
		string directory = cvsContents[i][0];

		for (int j = 1; j< cvsContents[i].size(); j++)
		{
			string filename = filePath + directory + "-" + cvsContents[i][j] + ".txt";
			double *data = setDataFromFile<double>(filename);
			toMatlabMat<double>((filePath + directory + "-" + cvsContents[i][j] + ".mat").c_str(), directory + "_" + cvsContents[i][j], 
				data, 240, 240);
			data = NULL;
		}
	}
	//vector<string> filenames;

	//for (int i = 0; i < 48; i++)
	//{

	//}

	//string matFile = "C:/Hagen/Research/Proyectos/AccousticModellingCEM/Codigo/VisionCem/data/allData/processedWatershed/Matlab_Mats/35.mat";
	//string floatsFile = "C:/Hagen/Research/Proyectos/AccousticModellingCEM/Codigo/VisionCem/data/allData/processedWatershed/Matlab_Mats/35.txt";
	//ifstream myfile("C:/Hagen/Research/Proyectos/AccousticModellingCEM/Codigo/VisionCem/data/allData/processedWatershed/Matlab_Mats/35.txt");
	//vector<float> floats;
	//if (myfile.is_open())
	//{
	//	while (getline(myfile, line))
	//	{

	//		cout << line << '\n';
	//	}
	//	myfile.close();
	//}

	//double *data = new double[240 * 240]();
	//for (int j = 0; j < 240; j++)
	//for (int i = 0; i < 240; i++)
	//{
	//	float f = (double)(i * 240 + j);
	//	data[j * 240 + i] = f;
	//}

	//double *data = setDataFromFile<double>(floatsFile);
	//toMatlabMat<double>(matFile.c_str(), data, 240, 240);

	return 0;
}