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
    Mat temp[75] = {};
    Mat result;
    Mat entry;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            temp[i * mat.cols + j] = mat.row(i).col(j);
        }
    }
    hconcat( temp, 75, result );
    return result;
}

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


