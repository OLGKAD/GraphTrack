#include <opencv2/opencv.hpp>
#include <iostream>
#include "PCA.hpp"
#include <stdlib.h>     /* srand, rand */
#include <vector>
#include "util.hpp"

using namespace cv;
using namespace std;

vector<int> mat_to_vector(Mat mat) {
    vector<int> array;
    if (mat.isContinuous()) {
        array.assign(mat.datastart, mat.dataend);
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), mat.ptr<int>(i), mat.ptr<int>(i)+mat.cols);
        }
    }
    return array;
}

// cout a vector
void print_vector(vector<int> vector1) {
    for (vector<int>::const_iterator i = vector1.begin(); i != vector1.end(); ++i)
        std::cout << (int) *i << ' ';
    cout << endl;
}

vector<int> add_vectors(const vector<int>& a, const vector<int>& b)
{
    assert(a.size() == b.size());
    
    vector<int> result;
    result.reserve(a.size());
    
    transform(a.begin(), a.end(), b.begin(),
              std::back_inserter(result), std::plus<int>());
    return result;
}

vector<int> subtract_vectors(const vector<int>& a, const vector<int>& b)
{
    assert(a.size() == b.size());
    
    vector<int> result;
    result.reserve(a.size());
    
    transform(a.begin(), a.end(), b.begin(),
              std::back_inserter(result), std::minus<int>());
    return result;
}

// flaten a Mat into a 1D array. Mat might have 3 (or more) channels. All values should be flattened into a single row. 
Mat flatten(Mat mat) {
    Mat bgr[3];   //destination array
    split(mat,bgr);//split source
    hconcat(bgr, 3, mat);
//    cout << mat.rows << endl; // 20
//    cout << mat.cols << endl; // 60
    Mat temp[1200] = {};
    Mat result;
    Mat entry;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            temp[i * mat.cols + j] = mat.row(i).col(j);
        }
    }
    hconcat( temp, 1200, result );
    return result;
}


