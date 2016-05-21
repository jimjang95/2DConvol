#include <vector>
#include <string>

using namespace std;

typedef  vector<vector<float>> Matrix;
	
int read_matrix(Matrix &m, string filename);
int write_matrix(Matrix &m, string filename);

int generate_data(int rows, int cols, string filename);