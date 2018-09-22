#include <opencv2/opencv.hpp>
#include <iostream>
#include "PCA.hpp"
#include "util.hpp"
#include <iterator>
#include <algorithm>
#include <functional>

using namespace cv;
using namespace std;

Mat mean_of_mats(Mat mat[]) {
    Mat mean(mat[2000].size(), CV_64FC1, Scalar(0));
    cout << mat[0] << endl;
//    cout << size(mat) << endl;
//    for (int i = 0; i < sizeof(mat); i++) {
//        mean += mat[i];
//    }
//    mean /= sizeof(mat);
    return mean;
}

// given an array of type vector<int> return the average (mean) of all the elements
vector<int> vector_mean(vector<vector<int>> vectors) {
    vector<int> mean(vectors[0].size(), 0);;
    for (int i = 0; i < vectors.size(); i++) {
//        print_vector(mean);
        mean = add_vectors(mean, vectors[i]);
//        cout << "hey" << endl;
//        print_vector(mean);
    }
    // divide mean by a number of vectors in the array
    transform(mean.begin(), mean.end(), mean.begin(), bind(multiplies<float>(), placeholders::_1, 1.0 / vectors.size()));
    return mean;
}

// shift vectors. Substruct the MEAN from every vector
vector<vector<int>> shift_vectors_by_mean(vector<vector<int>> vectors) {
    vector<int> mean = vector_mean(vectors);
    vector<vector<int>> result;
    result.reserve(vectors.size());
    
    for (int i = 0; i < vectors.size(); i++) {
//        print_vector(vectors[i]);
        result.push_back(subtract_vectors(vectors[i], mean));
//        print_vector(result[i]);
    }
    return result;
}


