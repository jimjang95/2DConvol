#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <omp.h>

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

	cout << "Start Partition" << endl;
	//cout << "[ " << X.size() << " x " << X[0].size() << " ] * [ " << K.size() << " x " << K[0].size()  << " ] = [ "
	//	<< Y.size() << " x " << Y[0].size() << " ]" << endl;
	chrono::system_clock::time_point StartTime = chrono::system_clock::now();
		
	//---------------2D Convolution---------------//
		
	// change this
	omp_set_num_threads(4);

	int x = X.size();
	int k = K.size();
	int fSize = sizeof(float);
	int yRowSize = x - k + 1;
	int yRowHalf = yRowSize >> 1;
	int yRowQuad = yRowHalf >> 1;
	int yColSize = x - k + 1;
	int yColHalf = yColSize >> 1;
	int yColQuad = yColHalf >> 1;


	//---------------2D Convolution---------------//
	float* Xvec = (float*)malloc(x * x * fSize);
	float* Kvec = (float*)malloc(k * k * fSize);
	float* tmpY = (float*)calloc(yRowSize * yColSize, fSize);
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < x; j++)
			Xvec[x * i + j] = X[i][j];
	}

	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++)
			Kvec[k * i + j] = K[i][j];
	}

	// change this
	// Partitioning Y
	// partition #01 - 2사분면 (0 ~ 1/2, 0 ~ 1/2)
#pragma omp parallel for
		for (int c = 0; c < yColHalf; c++) 
			for (int a = 0; a < yRowHalf; a += yRowQuad) {
				// multiple pointers to go through every row of K
				//in one iteration
				//(basically loop unrolling K)
				float** tmps = (float**)malloc(k * sizeof(float*));
				for (int i = 0; i < k; i++) {
					tmps[i] = &Kvec[i * k];
				}
				float* yStart = &tmpY[c * yRowSize + a];
				for (int b = 0; b < k; b++) {
					for (int i = 0; i < yRowQuad; i++) {
						for (int j = 0; j < k; j++) {
							// fix number
							*yStart += Xvec[x * (c + j) + a] * *tmps[j];
						}
						yStart++;
					}
					//한 줄 했으니까 yStart는 다시 줄 처음으로
					yStart = &tmpY[c * yRowSize + a];
					
					//tmps들도 한 칸 옆으로
					for (int i = 0; i < k; i++) {
						tmps[i]++;
					}
				}
				free(tmps);
			}

	//partition #02 - 1사분면 (1/2 ~ 1, 0 ~ 1/2)
#pragma omp parallel for
		for (int c = 0; c < yColHalf; c++) 
			for (int a = yRowHalf; a < yRowSize - 1; a += yRowQuad) {
				// multiple pointers to go through every row of K
				//in one iteration
				//(basically loop unrolling K)
				float** tmps = (float**)malloc(k * sizeof(float*));
				for (int i = 0; i < k; i++) {
					tmps[i] = &Kvec[i * k];
				}
				float* yStart = &tmpY[c * yRowSize + a];
				for (int b = 0; b < k; b++) {
					for (int i = 0; i < yRowQuad; i++) {
						for (int j = 0; j < k; j++) {
							// fix number
							*yStart += Xvec[x * (c + j) + a] * *tmps[j];
						}
						yStart++;
					}
					//한 줄 했으니까 yStart는 다시 줄 처음으로
					yStart = &tmpY[c * yRowSize + a];

					//tmps들도 한 칸 옆으로
					for (int i = 0; i < k; i++) {
						tmps[i]++;
					}
				}
				free(tmps);
			}

	//partition #03 - 3사분면 (0 ~ 1/2, 1/2 ~ 1)
#pragma omp parallel for
		for (int c = yColHalf; c < yColSize - 1; c++)
			for (int a = 0; a < yRowHalf; a += yRowQuad) {
				// multiple pointers to go through every row of K
				//in one iteration
				//(basically loop unrolling K)
				float** tmps = (float**)malloc(k * sizeof(float*));
				for (int i = 0; i < k; i++) {
					tmps[i] = &Kvec[i * k];
				}
				float* yStart = &tmpY[c * yRowSize + a];
				for (int b = 0; b < k; b++) {
					for (int i = 0; i < yRowQuad; i++) {
						for (int j = 0; j < k; j++) {
							// fix number
							*yStart += Xvec[x * (c + j) + a] * *tmps[j];
						}
						yStart++;
					}
					//한 줄 했으니까 yStart는 다시 줄 처음으로
					yStart = &tmpY[c * yRowSize + a];

					//tmps들도 한 칸 옆으로
					for (int i = 0; i < k; i++) {
						tmps[i]++;
					}
				}
				free(tmps);
			}

	//partition #04 - 4사분면 (1/2 ~ 1, 1/2 ~ 1)
#pragma omp parallel for
		for (int c = yColHalf; c < yColSize - 1; c++)
			for (int a = yRowHalf; a < yRowSize - 1; a+= yRowQuad) {
				// multiple pointers to go through every row of K
				//in one iteration
				//(basically loop unrolling K)
				float** tmps = (float**)malloc(k * sizeof(float*));
				for (int i = 0; i < k; i++) {
					tmps[i] = &Kvec[i * k];
				}
				float* yStart = &tmpY[c * yRowSize + a];
				for (int b = 0; b < k; b++) {
					for (int i = 0; i < yRowQuad; i++) {
						for (int j = 0; j < k; j++) {
							// fix number
							*yStart += Xvec[x * (c + j) + a] * *tmps[j];
						}
						yStart++;
					}
					//한 줄 했으니까 yStart는 다시 줄 처음으로
					yStart = &tmpY[c * yRowSize + a];

					//tmps들도 한 칸 옆으로
					for (int i = 0; i < k; i++) {
						tmps[i]++;
					}
				}
				free(tmps);
			}

	//마지막 1줄 - both row / col

	//free()들
	free(Xvec);
	free(Kvec);
	free(tmpY);
								
	//---------------2D Convolution---------------//

	chrono::system_clock::time_point EndTime = chrono::system_clock::now();
	chrono::microseconds microTest = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
	cout << "partition done" << endl;
	cout << "Time : " << microTest.count() << endl;

	//////////////////////////////////////////////////
	////---------------2D Convolution---------------//
	cout << "Start Full" << endl;
	//cout << "[ " << x << " x " << x << " ] * [ " << k << " x " << k  << " ] = [ "
	//	<< Y.size() << " x " << Y[0].size() << " ]" << endl;
	StartTime = chrono::system_clock::now();
	//FULL Y
//#pragma omp parallel for
	for (int a = 0; a< yRowSize; a++)    // X 가로 길이 - K 가로 길이 + 1
		for (int b = 0; b< k; b++)              // K 가로 길이
			for (int c = 0; c < yColSize; c++)     // X 세로 길이 - K 세로 길이 + 1
				for (int d = 0; d< k; d++)			  // K 세로 길이
					Y[c][a] += X[c + d][a + b] * K[d][b];
	EndTime = chrono::system_clock::now();
	chrono::microseconds microBase = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
	cout << "full done" << endl;
	cout << "Time : " << microBase.count() << endl;
	cout << "Result: " << (float) microBase.count() / (float) microTest.count() << " times faster" << endl;
	////---------------2D Convolution---------------//
	//////////////////////////////////////////////////
	write_matrix(Y, "Y.txt");

	return 0;	
}
