#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>

#include "utility.h"

// type Matrix: vector<vector<float>> 

// following functions are defined in utility.h 
//
// generate_data: generate a random matrix and save it.
// read_matrix:   read matrix from the file.
// write_matrix:  write matrix from the file. 

using namespace std;

int main()
{
	srand(time(NULL));

	Matrix X;
	Matrix K;

	// generate random data for test, comment below two lines to use your own input data.
	generate_data(200, 200, "X.txt");
	generate_data(40, 40, "K.txt");
	
	//------- you don't need to change here ----------//
	// read input matrices from the file
	read_matrix(X, "X.txt");
	read_matrix(K, "K.txt");

	// initialize the output matrix
	Matrix Y;
	Y.resize(X.size() - K.size() +1);
	for (int a = 0; a < X.size() - K.size() + 1; a++)
	{
		Y[a].resize(X[0].size() - K[0].size() + 1);
	}
	//-----------------------------------------------//
	
	//-----------------------------------------------//
	// If needed, you can do pre-processing here. (Ex: Rearrange X or K))
	// Don't touch Y.
	//
	//-----------------------------------------------//

	cout << "Start 2D-Convolution" << endl;
	cout << "[ " << X.size() << " x " << X[0].size() << " ] * [ " << K.size() << " x " << K[0].size()  << " ] = [ "
		<< Y.size() << " x " << Y[0].size() << " ]" << endl;
	chrono::system_clock::time_point StartTime = chrono::system_clock::now();
		
	//---------------2D Convolution---------------//
		
	// change this
	for (int a = 0; a< X[0].size() - K[0].size() + 1; a++)
		for (int b = 0; b< K[0].size(); b++)
			for (int c = 0; c < X.size() - K.size() + 1; c++)
				for (int d = 0; d< K.size(); d++)											
							Y[c][a] += X[c+d][a+b] * K[d][b];
								
	//---------------2D Convolution---------------//

	chrono::system_clock::time_point EndTime = chrono::system_clock::now();
	chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
	cout << "2D-Convolution done" << endl;
	cout << "Time : " << micro.count() << endl;

	write_matrix(Y, "Y.txt");

	return 0;	
}