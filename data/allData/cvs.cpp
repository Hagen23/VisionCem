#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

vector<vector<string>> getCsvContent(string filename)
{
	ifstream file ( filename ); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	string value;
	vector<vector<string> > cvsContents;
	while ( file.good() )
	{
		vector<string> line;
		while(getline ( file, value, ',' ))
			line.push_back(value);
		cvsContents.push_back(line);
	}
	return cvsContents;
}

int main()
{
	vector<vector<string> > cvsContents = getCsvContent("surveyFilesToProcess.csv");
	for(auto v : cvsContents)
	{
		for(auto s : v)
			cout << s << " ";
		cout << endl;
	}
	return 1;
}
