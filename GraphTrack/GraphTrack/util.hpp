//
//  util.hpp
//  GraphTrack
//
//  Created by Kadyrakunov Olzhas on 19/09/2018.
//  Copyright © 2018 Olzhas Kadyrakunov. All rights reserved.
//

#ifndef util_hpp
#define util_hpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <vector>
using namespace cv;
using namespace std;


#include <stdio.h>
vector<int> mat_to_vector(Mat mat);
void print_vector(vector<int> vector1);
vector<int> add_vectors(const vector<int>& a, const vector<int>& b);
vector<int> subtract_vectors(const vector<int>& a, const vector<int>& b);
Mat flatten(Mat mat);
#endif /* util_hpp */
