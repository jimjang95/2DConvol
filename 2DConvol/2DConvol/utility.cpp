#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "utility.h"

using namespace std;

int generate_data( int rows, int cols, string filename)
{
	Matrix M;
		
	M.resize(rows);
	for (int b = 0; b < rows; b++)
		M[b].resize(cols, 0);
		
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			float temp = -5 + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 10);
			M[i][j] = temp;
		}
	}
	
	write_matrix(M, filename);

	return 0;
}

int write_matrix(Matrix &m, string filename)
{
	ofstream output_matrix;
	output_matrix.open(filename);
	if (!output_matrix.good())
	{
		cout << "could not open " << filename << endl;
		return -1;
	}
	int rows = m.size();
	int cols = m[0].size();

	output_matrix << rows << " " << cols << endl;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			output_matrix << m[i][j] << " ";
		}
		output_matrix << endl;
	}
	output_matrix.close();
	return 0;
}

int read_matrix(Matrix &m, string filename)
{
	ifstream input_matrix;
	input_matrix.open(filename);
	if (!input_matrix.good())
	{
		cout << "could not open " << filename << endl;
		return -1;
	}
	int rows = 0;
	int cols = 0;
	input_matrix >> rows;
	input_matrix >> cols;

	m.resize(rows);

	for (int i = 0; i < rows; i++)
	{
		m[i].resize(cols, 0);
		for (int j = 0; j < cols; j++)
		{
			input_matrix >> m[i][j];
		}
	}
	input_matrix.close();
	if (rows != m.size() || cols != m[0].size())
	{
		cout << "input failed" << endl;
		return -1;
	}
	return 0;
}