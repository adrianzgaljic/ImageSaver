#include <iostream>
#include <fstream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/face.hpp"
#include "opencv2/core/core.hpp"


using namespace cv;
using namespace std;
using namespace cv::face;



vector<vector<Point2f>> faceDetector(const Mat& image) {
    std::vector<Rect> faces;
    const string cascade_name = "/Users/adrianzgaljic/Desktop/react-native-opencv3-tests/HoughCircles/node_modules/react-native-opencv3/android/build/intermediates/library_assets/debug/out/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    if (not face_cascade.load(cascade_name)) {
        cerr << "Cannot load cascade classifier from file: " <<    cascade_name << endl;
    }
    Mat gray;
    // The cascade classifier works best on grayscale images
    if (image.channels() > 1) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    // Histogram equalization generally aids in face detection
    equalizeHist(gray, gray);
    faces.clear();
    // Run the cascade classifier
    face_cascade.detectMultiScale(
            gray,
            faces,
            1.4, // pyramid scale factor
            3,   // lower thershold for neighbors count
            // here we hint the classifier to only look for one face
            CASCADE_SCALE_IMAGE + CASCADE_FIND_BIGGEST_OBJECT);


    const string facemark_filename = "/Users/adrianzgaljic/Desktop/lbfmodel.yml";
    Ptr<Facemark> facemark = createFacemarkLBF();
    facemark->loadModel(facemark_filename);
    vector<vector<Point2f>> shapes;
    facemark->fit(image, faces, shapes);
    return shapes;
}

void customReproject(const cv::Mat& disparity, cv::Mat& out3D, cv::Mat& outColor){

    // 3-channel matrix for containing the reprojected 3D world coordinates
    out3D = cv::Mat::zeros(disparity.size(), CV_32FC3);
    outColor = cv::Mat::zeros(disparity.size(), CV_32FC3);

    // Getting the interesting parameters from Q, everything else is zero or one
    float Q03 = -509.091339111328125;
    float Q13 = -512.0567626953125;
    float Q23 = 1465.2501220703125;
    float Q32 = 0.009868702407;
    float Q33 = 2.50484309702818564;

    // Transforming a single-channel disparity map to a 3-channel image representing a 3D surface
    for (int i = 0; i < disparity.rows; i++)
    {
        const uchar* disp_ptr = disparity.ptr<uchar>(i);
        cv::Vec3f* out3D_ptr = out3D.ptr<cv::Vec3f>(i);
        cv::Vec3f* outColor_ptr = outColor.ptr<cv::Vec3f>(i);

        for (int j = 0; j < disparity.cols; j++)
        {
            const float w = (int)disp_ptr[j] * Q32 + Q33;

            cv::Vec3f& point = out3D_ptr[j];
            cv::Vec3f& pointColor = outColor_ptr[j];
            point[0] = (static_cast<float>(j) + Q03)/w;
            point[1] = (static_cast<float>(i) + Q13)/w;
            point[2] =  Q23/w;
            if (j>640 and j<700 and i>470 and i<590){
                pointColor[0] = 255;
                pointColor[1] = 0;
                pointColor[2] = 0;
            } else {
                pointColor[0] = 0;
                pointColor[1] = 255;
                pointColor[2] = 0;
            }

        }
    }
}

int main() {

    Mat disparity = imread("/Users/adrianzgaljic/Desktop/NERO/pointclouds/ensenso snimke fantoma/test 5.2.2020./depthgray.png", 0);
    Mat img = imread("/Users/adrianzgaljic/Desktop/NERO/pointclouds/ensenso snimke fantoma/test 5.2.2020./r.png", 1);
    Mat l = imread("/Users/adrianzgaljic/Desktop/NERO/pointclouds/ensenso snimke fantoma/test 5.2.2020./l.png", 0);
    Mat r = imread("/Users/adrianzgaljic/Desktop/NERO/pointclouds/ensenso snimke fantoma/test 5.2.2020./r.png", 0);

    Matx44d Q = cv::Matx44d(
            1.0, 0.0, 0.0,     -509.091339111328125,
            0.0, 1.0, 0.0, -512.0567626953125,
            0.0, 0.0, 0.0, 1465.2501220703125,
            0.0, 0.0, 0.009868702407, 2.50484309702818564
    );
    Mat xyz;
    Mat color;

    //reprojectImageTo3D(disparity, xyz, Q, true);
    customReproject(disparity, xyz, color);
    std::ofstream outFile("output.ply");
    if (!outFile.is_open())
    {
        std::cerr << "ERROR: Could not open "  << std::endl;
    }

    outFile << "ply" << endl;
    outFile << "format ascii 1.0" << endl;
    outFile << "element vertex "  << xyz.rows*xyz.cols << endl;
    outFile << "property float x" << endl;
    outFile << "property float y" << endl;
    outFile << "property float z" << endl;
    outFile << "property uchar red" << endl;
    outFile << "property uchar green" << endl;
    outFile << "property uchar blue" << endl;
    outFile << "end_header" << endl;

    for (int i=0; i<xyz.rows; i++){
        const cv::Vec3f* image3D_ptr = xyz.ptr<cv::Vec3f>(i);
        const cv::Vec3f* imageColor_ptr = color.ptr<cv::Vec3f>(i);

        for (int j = 0; j < xyz.cols; j++)
        {
            outFile << image3D_ptr[j][0] << " " << image3D_ptr[j][1] << " " << image3D_ptr[j][2] << " " << imageColor_ptr[j][0] << " " << imageColor_ptr[j][1] << " " << imageColor_ptr[j][2] << std::endl;
        }
    }
    outFile.close();

    Mat disp, disp8;
    Ptr<StereoBM> sbm = StereoBM::create(16, 19);
    sbm->compute(l,r,disp);

    normalize(disp, disp8, 0, 255, 32, CV_8U);

    imshow("Disp", disp8);

    waitKey(0);


}