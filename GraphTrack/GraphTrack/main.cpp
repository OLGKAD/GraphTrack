#include <opencv2/opencv.hpp>
#include <iostream>
#include "PCA.hpp"
#include <stdlib.h>     /* srand, rand */
#include <vector>
#include "util.hpp"
#include <ctime>

using namespace cv;
using namespace std;

// GLOBAL VARIABLES
Mat frames[270] = {}; // video has 270 frames
int width = 720;
int height = 528;
int patch_width = 20;
int patch_height = 20;
int patches_per_frame = 115;
//Mat pca_patches[115 * 270] = {}; // 115 patches from each frame; each patch has size 20 x 60 (RGB for each pixel)
//vector<vector<int>> pca_patches (115 * 270);
// to save time only 20 frames will be processed (out of 270 => should be 115 * 270)
Mat pca_patches[115 * 20] = {};

int main(int argc, const char * argv[]) {
    VideoCapture cap("media/Megamind.avi");
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video camera" << endl;
        cin.get(); //wait for any key press
        return -1;
    }
    
    int frameNumber = 0;
    while (true)
    {
        Mat frame;
        bool isSuccess = cap.read(frame); // read a new frame from the video camera
        if (isSuccess == false || frameNumber >= 20)
        {
            cout << "Video file ended" << endl;
            break;
        }
        frames[frameNumber] = frame;
        Mat patch;
        for (int i = 0; i < patches_per_frame; i++) {
            srand(time(0));
            int patch_x = rand() % (width / patch_width);
            int patch_y = rand() % (height / patch_height);
            pca_patches[frameNumber * patches_per_frame + i] = flatten(frame(Rect( patch_x * patch_width, patch_y * patch_height, patch_width, patch_height))).clone();
        }
        frameNumber += 1;
    }

    Mat patches;
    vconcat( pca_patches, 115 * 20, patches );
    
    PCA pca(patches, // pass the data
            Mat(), // we do not have a pre-computed mean vector,
            // so let the PCA engine to compute it
            PCA::DATA_AS_ROW, // indicate that the vectors
            // are stored as matrix rows
            // (use PCA::DATA_AS_COL if the vectors are
            // the matrix columns)
            16 // specify, how many principal components to retain
            );
    
    Mat eigenvalues = pca.eigenvalues;
    Mat eigenvectors = pca.eigenvectors; // the top 16 are already chosen. So, x_new = eigenvectors * x_old
//    cout << eigenvalues.rows << endl; // 16
//    cout << eigenvalues.cols << endl; // 1
//    cout << eigenvalues << endl;
//    cout << eigenvectors.rows << endl; // 16
//    cout << eigenvectors.cols << endl; // 1200
}

