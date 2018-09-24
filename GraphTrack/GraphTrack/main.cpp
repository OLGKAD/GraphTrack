#include <opencv2/opencv.hpp>
#include <iostream>
#include "PCA.hpp"
#include <stdlib.h>     /* srand, rand */
#include <vector>
#include "util.hpp"
#include <ctime>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

// GLOBAL VARIABLES
Mat frames[270] = {}; // video has 270 frames
int width = 720;
int height = 528;
int patch_width = 5;
int patch_height = 5;
int patches_per_frame = 15000;
//Mat pca_patches[15000 * 270] = {}; // 15000 patches from each frame; each patch has size 20 x 60 (RGB for each pixel)
//vector<vector<int>> pca_patches (15000 * 270);
// to save time only 20 frames will be processed (out of 270 => should be 15000 * 270)
int lastFrame = 10;
int currentFrame = 0;
string window_name = "my_window";
Mat pca_patches[15000 * 10] = {}; // HAS TO BE 20 000 * 10, at least
Mat compressed_patches[(528 - 4)][(720 - 4)] = {};
Mat positive_patches[5] = {};
Mat negative_patches[5 * 20] = {}; // in each frame where interest points were marked (5), we select a number of negative examples (20 - CHOSEN RANDOMLY)
//VideoCapture cap;


PCA computePCA_basis() {
    cout << "Running PCA" << endl;
    VideoCapture cap("media/Megamind.avi");
    
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video" << endl;
        cin.get(); //wait for any key press
    }
    
    int frameNumber = 0;
    while (true)
    {
        Mat frame;
        bool isSuccess = cap.read(frame); // read a new frame from the video
        if (isSuccess == false || frameNumber >= 10)
        {
            cout << "Video file ended" << endl;
            break;
        }
//        imshow(window_name, frame);
//        char c=(char)waitKey(250);
//        if(c==27)
//            break;
        
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
    
    Mat patches(15000 * 10, 75, CV_32FC1);
    vconcat( pca_patches, 15000 * 10, patches );
    
    PCA pca(patches, Mat(), PCA::DATA_AS_ROW, 16);

    return pca;
}

// use the PCA basis (16 eigenvectors) to project each patch vector in every frame to a 16-vector space
void compress_all_patches(PCA pca) {
    cout << "All patches in the video are being compressed" << endl;
    Mat eigenvalues = pca.eigenvalues;
    Mat eigenvectors = pca.eigenvectors; // the top 16 are already chosen. So, x_new = eigenvectors * x_old
    Mat mean = pca.mean;
    Mat transposed_mean;
    transpose(mean, transposed_mean);
    
    Mat frame_temp;
    Mat frame;
    Mat transposed_patch(75, 1, CV_32FC1);
    Mat compressed_patch(75, 1, CV_32FC1);
    for (int i = 0; i < lastFrame; i++) {
//        cout << frames[i].rows << endl; //
//        cout << frames[i].cols << endl;
//        cout << mean.rows << endl; //
//        cout << mean.cols << endl;
        frame_temp = frames[i];
//        cout << frame_temp << endl;
//        cout << endl;
        frame_temp.convertTo(frame, CV_32FC1);
//        cout << frame << endl;
        for (int j = 0; j < height - 4; j++) {
            cout << "frame number: " << i << ", loop number~: " << j * (width - 4) << endl;
            for (int k = 0; k < width - 4; k++) {
//                if (j == 0 & k == 0)
                transpose(flatten(frame(Rect( k, j, patch_width, patch_height))), transposed_patch); // very fast when commented out => flatten (or transpose) is the BOTTLENECK. When patch size was changed from 10x10 to 5x5, time/frame went from 35 sec -> 10 sec
//
                cout << GetMatType(eigenvectors) << endl; // CV_32FC1
//                cout << GetMatType(transposed_patch) << endl; // CV_8UC1
                compressed_patch = eigenvectors * (transposed_patch - transposed_mean); // when commented out, it takes ~35 sec / frame, when not - about the same! => the BOTTLENECK IS the "flatten" function.
                
                compressed_patches[j][k] = compressed_patch.clone();
            }
        }
    }
    
}
void on_trackbar(int, void* args) {
    //    cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame);
    VideoCapture cap = (*((VideoCapture*)(args)));
    cap.set(CAP_PROP_POS_FRAMES,currentFrame);
    Mat frame;
    cap.read(frame);
    imshow(window_name, frame);
    cout << currentFrame << endl;
}

void mark_interest_points() {
    cout << "Mark interest points" << endl;
    VideoCapture cap("media/Megamind.avi");
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video" << endl;
        cin.get(); //wait for any key press
    }
    namedWindow(window_name); //create a window
    Mat frame;
    cap.set(CAP_PROP_POS_FRAMES,1);
    cap.read(frame);
    imshow(window_name, frame);
    createTrackbar( "frames", "my_window", &currentFrame, lastFrame, on_trackbar, &cap );
    char c=(char)waitKey(25000);

    
    cap.release();
    destroyAllWindows();
}
// each patch will be represented using 16 numbers (instead of 300), and everything will be stored in compressed_video array;
void compress_video() {
    PCA pca = computePCA_basis();
    compress_all_patches(pca);
}



// Right now, video is being read for PCA, played (in window), and manipulated (trackbar) at the same time. Should happen like that.
// 1. First it should be read for PCA, and then each patch is represented using 16 bytes using pca.eigenvectors.
// 2. Then the user uses trackpad to mark interest points.
// 3. Then preses "esc" and the video is played.
int main(int argc, const char * argv[]) {
    compress_video();
    mark_interest_points();
}

