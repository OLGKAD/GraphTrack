#include <opencv2/opencv.hpp>
#include <iostream>
#include "PCA.hpp"
#include <stdlib.h>     /* srand, rand */
#include <vector>
#include "util.hpp"

using namespace cv;
using namespace std;

void write_mat_to_file(Mat mat, string filename) {
    FileStorage file("debug_log/" + filename, FileStorage::WRITE);
    
    file << filename << mat; // the matrix in the file will be named the same as the filename here.
}
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
Mat flatten2(Mat patch) {
    Mat bgr[3];   //destination array
    Mat mat(patch.rows, patch.cols * 3, CV_32FC1);

    split(patch,bgr);//split source
    hconcat(bgr, 3, mat);
    Mat temp[75] = {};
    Mat result(1, 75, CV_32FC1);
    Mat row_pointer;
    for (int i = 0; i < mat.rows; ++i) {
        row_pointer = mat.row(i);
/////////////////////////////////////////// SPEED UP: save pointer to mat.row(i), and then access temp_row.col(j)
        for (int j = 0; j < mat.cols; ++j) {
            temp[i * mat.cols + j] = row_pointer.col(j);
        }
    }
    hconcat( temp, 75, result );
    return result;
}

// another version of "flatten". As of now, it's faster
/////////////////////////////////////// SOLUTION: seems like the only way to speed things up is to understand that "CONVOLUTIONS" part
Mat flatten(Mat patch) {
    Mat bgr[3];   //destination array
    split(patch,bgr);//split source
    Mat result;
    // stack the channels into a new mat:
    // this loop really SLOWS things down
    for (int i=0; i<3; i++)
        result.push_back(bgr[i]);
    result = result.reshape(1,1);

    return result;
    
//    Mat m1 = Mat(1,75, CV_32FC1, double(2));
//    return m1;
}

//Mat res; // stack the channels into a new mat:
//for (size_t i=0; i<chans.size(); i++)
//res.push_back(chans[i]);
//
//res = res.reshape(1,1); // flat 1d, again.

// multiply a vector by a matrix
//Mat transform_vector(Mat matrix, Mat vector) {
//    Mat result;
//    Mat temp[matrix.rows];
//    
//    for (int i = 0; i < matrix.rows; i++) {
//        for (int j = 0; j < matrix.cols; j++) {
//            temp 
//        }
//    }
//    
//}

// not written by me. Delete when not needed
std::string GetMatDepth(const cv::Mat& mat)
{
    const int depth = mat.depth();
    
    switch (depth)
    {
        case CV_8U:  return "CV_8U";
        case CV_8S:  return "CV_8S";
        case CV_16U: return "CV_16U";
        case CV_16S: return "CV_16S";
        case CV_32S: return "CV_32S";
        case CV_32F: return "CV_32F";
        case CV_64F: return "CV_64F";
        default:
            return "Invalid depth type of matrix!";
    }
}

// not written by me. Delete when not needed
std::string GetMatType(const cv::Mat& mat)
{
    const int mtype = mat.type();
    
    switch (mtype)
    {
        case CV_8UC1:  return "CV_8UC1";
        case CV_8UC2:  return "CV_8UC2";
        case CV_8UC3:  return "CV_8UC3";
        case CV_8UC4:  return "CV_8UC4";
            
        case CV_8SC1:  return "CV_8SC1";
        case CV_8SC2:  return "CV_8SC2";
        case CV_8SC3:  return "CV_8SC3";
        case CV_8SC4:  return "CV_8SC4";
            
        case CV_16UC1: return "CV_16UC1";
        case CV_16UC2: return "CV_16UC2";
        case CV_16UC3: return "CV_16UC3";
        case CV_16UC4: return "CV_16UC4";
            
        case CV_16SC1: return "CV_16SC1";
        case CV_16SC2: return "CV_16SC2";
        case CV_16SC3: return "CV_16SC3";
        case CV_16SC4: return "CV_16SC4";
            
        case CV_32SC1: return "CV_32SC1";
        case CV_32SC2: return "CV_32SC2";
        case CV_32SC3: return "CV_32SC3";
        case CV_32SC4: return "CV_32SC4";
            
        case CV_32FC1: return "CV_32FC1";
        case CV_32FC2: return "CV_32FC2";
        case CV_32FC3: return "CV_32FC3";
        case CV_32FC4: return "CV_32FC4";
            
        case CV_64FC1: return "CV_64FC1";
        case CV_64FC2: return "CV_64FC2";
        case CV_64FC3: return "CV_64FC3";
        case CV_64FC4: return "CV_64FC4";
            
        default:
            return "Invalid type of matrix!";
    }
}


