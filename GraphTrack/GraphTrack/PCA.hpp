//
//  PCA.hpp
//  GraphTrack
//
//  Created by Kadyrakunov Olzhas on 18/09/2018.
//  Copyright © 2018 Olzhas Kadyrakunov. All rights reserved.
//

#ifndef PCA_hpp
#define PCA_hpp


using namespace cv;
using namespace std;
#include <stdio.h>

Mat mean_of_mats(Mat mat[]);
vector<int> vector_mean(vector<vector<int>> vectors);
vector<vector<int>> shift_vectors_by_mean(vector<vector<int>> vectors);
#endif /* PCA_hpp */
