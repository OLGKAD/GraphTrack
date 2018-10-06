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
Mat frames_unnormalized[269] = {};
Mat frames[269] = {}; // video has 269 frames (270, but the first frame is blank)
int width = 256; // 256; // 720;
int height = 240; // 240; //528;
int patch_width = 5;
int patch_height = 5;
int patches_per_frame = 25000; // 1/8 of all patches, needed for PCA
int lastFrame = 15;
int currentFrame = 0;
string window_name = "my_window";
Mat pca_patches[25000 * 15] = {};
Mat compressed_patches[15][(240 - 4)][(256 - 4)] = {};
Mat positive_patches[5] = {}; // one per frame in only some frames
int positve_patch_counter = 0; // the next positve patch will be saved at this index in positive_patches
Mat negative_patches[5 * 20] = {}; // in each frame where interest points were marked (5), we select a number of negative examples (20 - CHOSEN RANDOMLY)
int negative_patches_per_frame = 20;
int negative_patch_counter = 0; // the next negative patch will be saved at this index in negative_patches
Mat candidate_nodes[15][250] = {}; // 250 candidates patches, per each frame
int nodes_per_frame = 250;


/* computes the average color of the video: sum all the pixel RGB values (over all pixels, over all frames) and divide by the #pixels (width * height * frames).  AND
 * Substracts the average color from every pixel in the video, before it's used in PCA.
 */
void compute_average_color() {
    cout << "computing average color" << endl;
    Mat average_frame(height, width, CV_32FC3, Scalar(0));
    float average_color;
    Mat temp1;
    for (int i = 0; i < lastFrame; i++) {
        frames_unnormalized[i].convertTo(temp1, CV_32FC3);
//        cout << average_frame.type() << endl; // 21
//        cout << temp1.type() << endl; // 21
        average_frame += temp1;
    }
    write_mat_to_file(average_frame, "sum_of_frames");
    average_frame /= lastFrame;
//    write_mat_to_file(average_frame, "average_frame");
    average_color = sum(average_frame)[0] / (height * width);
    write_mat_to_file(average_frame, "average_frame");
    cout << "average color is: " << average_color << endl;
    // now subtract average color from every pixel of every frame
    Mat temp2;
    for (int i = 0; i < lastFrame; i++) {
        frames_unnormalized[i].convertTo(temp2, CV_32FC3);
//        cout << temp.rows << endl; // 528
//        cout << temp.cols << endl; // 720
//        write_mat_to_file(temp2, "temp"); // the very last temp will be saved as the file is written at every iteration
        frames[i] = temp2 - Scalar(average_color, average_color, average_color); // ERROR here
    }
    
    namedWindow("temp");
//    cout << average_frame.row(0).col(0) << endl;
    imshow("temp", average_frame / 255); // before displaying it multiplies all pixel values by 255. (that's how imshow works)
    waitKey(0);
    destroyAllWindows();
    
}

// fills out the "frames" array
void read_video() {
    cout << "Reading the video" << endl;
    VideoCapture cap("media/drop.avi");
    
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video" << endl;
        cin.get(); //wait for any key press
    }
    
    int frameNumber = -1;
    Mat frame;
    while (true)
    {
        // skip the first frame, since it's blank
        if (frameNumber == -1) {
            frameNumber += 1;
            cap.read(frame);
            cout << frame.rows << endl;
            cout << frame.cols << endl;
            continue;
        }
        
        bool isSuccess = cap.read(frame); // read a new frame from the video
        if (isSuccess == false || frameNumber >= lastFrame)
        {
            break;
        }
        frames_unnormalized[frameNumber] = frame;
        frameNumber += 1;
    }
    cap.release();
    write_mat_to_file(frames_unnormalized[0], "first_video_frame_before_normalizing");
    compute_average_color();
    write_mat_to_file(frames[0], "first_video_frame");
}

PCA computePCA_basis() {
    cout << "Running PCA" << endl;
    
    Mat frame;
    Mat frame_temp;
    for (int i = 0; i < lastFrame; i++) {
        frame_temp = frames[i];
        frame_temp.convertTo(frame, CV_32FC3);
        Mat patch;
        for (int j = 0; j < patches_per_frame; j++) {
            srand(time(0));
            int patch_x = rand() % (width / patch_width);
            int patch_y = rand() % (height / patch_height);
            pca_patches[i * patches_per_frame + j] = flatten(frame(Rect( patch_x * patch_width, patch_y * patch_height, patch_width, patch_height))).clone();
        }
    }
    
    Mat patches(patches_per_frame * lastFrame, patch_width * patch_height * 3, CV_32FC1);
    vconcat( pca_patches, patches_per_frame * lastFrame, patches );
    
    PCA pca(patches, Mat(), PCA::DATA_AS_ROW, 16);
    
    return pca;
}

// use the PCA basis (16 eigenvectors) to project each patch vector in every frame to a 16-vector space
void compress_all_patches(PCA pca) {
    cout << "All patches in the video are being compressed" << endl;
    Mat eigenvalues = pca.eigenvalues;
    Mat eigenvectors = pca.eigenvectors; // the top 16 are already chosen. So, x_new = eigenvectors * x_old
    Mat mean = pca.mean;   // ERROR: PCA.mean != actual mean, as it wasn't computed over all patches.
    cout << "MEAN (should be zero: " << mean << endl;
    Mat transposed_mean;
    transpose(mean, transposed_mean);
    
    Mat frame_temp;
    Mat frame;
    Mat transposed_patch(patch_height * patch_width * 3, 1, CV_32FC1);
    Mat compressed_patch(patch_height * patch_width * 3, 1, CV_32FC1);
    for (int i = 0; i < lastFrame; i++) {
        cout << "frame number: " << i << endl;
        frame_temp = frames[i];
//        cout << frame_temp << endl;
//        cout << endl;
        frame_temp.convertTo(frame, CV_32FC3);
//        cout << frame << endl;
        for (int j = 0; j < height - patch_height + 1; j++) {
//            cout << "frame number: " << i << ", loop number~: " << j * (width - 4) << endl;
            for (int k = 0; k < width - patch_width + 1; k++) {
//                if (j == 0 & k == 0)
                transpose(flatten(frame(Rect( k, j, patch_width, patch_height))), transposed_patch);
                compressed_patch = eigenvectors * transposed_patch;
//                compressed_patch = eigenvectors * (transposed_patch - transposed_mean);
                
                compressed_patches[i][j][k] = compressed_patch.clone();
//                if (i == 0 && j == 0 && k == 0) {
//                    cout << transposed_mean << endl; //
//                    cout << transposed_patch << endl; //
//                    cout << eigenvectors << endl; //
//                    cout << compressed_patches[i][j][k] << endl; //
//                }
            }
        }
        
    }
    
}
void compress_all_patches_new(PCA pca) {
    Mat eigenvalues = pca.eigenvalues;
    Mat eigenvectors = pca.eigenvectors;
    Mat frame_temp;
    Mat kernel_one = eigenvectors.row(0).reshape(1,5);
    for (int i = 0; i < lastFrame; i++) {
        frame_temp = frames[i];
    }
}

void on_mouse_click(int event, int x, int y, int flags, void* userdata)
{
    Mat &img = *((Mat*)(userdata));
    if  ( event == EVENT_LBUTTONDOWN )
    {
//        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        circle(img,Point(x,y),10,Scalar(255,255,255), CV_FILLED, 1);
        imwrite("media/ex.jpg", img);
        imshow(window_name, img);
        //        void circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)¶
        
        // save the positive patch
        positive_patches[positve_patch_counter] = compressed_patches[currentFrame][y][x];
        positve_patch_counter += 1;
        // save the negative patches. THIS PART IS INCOMPLETE
        int x_temp;
        int y_temp;
        for (int i = 0; i < negative_patches_per_frame; i++) {
            srand(time(0));
            x_temp = rand() % (width - patch_width + 1);
            y_temp = rand() % (height - patch_height + 1);
            negative_patches[negative_patch_counter] = compressed_patches[currentFrame][y_temp][x_temp];
            negative_patch_counter += 1;
        }
    }
}

void on_trackbar(int, void* args) {
    Mat frame = frames[currentFrame];
    imshow(window_name, frame);
    setMouseCallback(window_name, on_mouse_click, &frame); ///
    waitKey(0);
    cout << "current frame: " << currentFrame << endl;
}

void mark_interest_points() {
    cout << "Mark interest points" << endl;
    namedWindow(window_name); //create a window
    Mat frame;
    frame = frames[0];
    setMouseCallback(window_name, on_mouse_click, &frame);
    imshow(window_name, frame);
    createTrackbar( "frames", "my_window", &currentFrame, lastFrame, on_trackbar);
    
    /// Wait until user press some key
    waitKey(0);

    cout << "the wait is over" << endl;
    destroyAllWindows();
}


// given coordinates of a patch in copressed_patches, it returns its distance (sum of absoulute differences) to the closest positive patch
int distance_to_positive_patches(int frame, int y, int x) {
    int closest_distance = 1000;
    int this_distance;
    
    for (int i = 0; i < positve_patch_counter; i++) {
        this_distance = sum(abs(positive_patches[i] - compressed_patches[frame][y][x]))[0];
        if (this_distance < closest_distance) {
            closest_distance = this_distance;
        }
    }
    
    return closest_distance;
}

// candidate selection
// selects 250-250 nodes (patches) in each frame, resembling the interest points the most.
void select_candidates() {
    cout << "Candidate nodes are being chosen" << endl;
    /* we'll have (w-4) * (h-4) values, and we'll need the 250 smallest. Sorting will cost O(w*h * log(w*h)). Finding 250 smallest costs O(w*h * 250), which is faster.
     So, we'll first find the min, then the min of the rest, etc.
     */
    Mat distances(height - patch_height + 1, width - patch_width + 1, CV_32FC1);
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    for (int i = 0; i < lastFrame; i++) {
        // fill up "distances"
        for (int j = 0; j < height - patch_height + 1; j++) {
            for (int k = 0; k < width - patch_width + 1; k++) {
                distances.row(j).col(k) = distance_to_positive_patches(i, j, k);
            }
        }
        for (int j = 0; j < nodes_per_frame; j++) {
            minMaxLoc( distances, &minVal, &maxVal, &minLoc, &maxLoc );
            candidate_nodes[i][j] = compressed_patches[i][minLoc.y][minLoc.x];
            distances.row(minLoc.y).col(minLoc.x) = maxVal;
        }
        
    }
}

// each patch will be represented using 16 numbers (instead of 300), and everything will be stored in compressed_video array;
void compress_video() {
    PCA pca = computePCA_basis();
    write_mat_to_file(pca.eigenvectors, "PCA_eigenvectors");
    compress_all_patches(pca); // runs at about 5-10 sec / frame, depending on a day... . Fine for now.
    write_mat_to_file(compressed_patches[0][0][0], "first_compressed_patch");
}


// 1. First it should be read for PCA, and then each patch is represented using 16 bytes using pca.eigenvectors.
// 2. Then the user uses trackpad to mark interest points.
// 3. Then preses "esc" and the video is played.
int main(int argc, const char * argv[]) {
//    PCA pca = computePCA_basis(); // here only for testing
    read_video();
    compress_video();
    mark_interest_points();
    select_candidates();
    cout << candidate_nodes[0][0] << endl;
}

