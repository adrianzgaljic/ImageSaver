#include <iostream>
#include <fstream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/face.hpp"


using namespace cv;
using namespace std;
using namespace cv::face;


void showDisparity(Mat img, Mat img2) {
    Mat disp, disp8;
    Ptr<StereoBM> sbm = StereoBM::create(16,9);
    sbm->setNumDisparities(16);
    sbm->setPreFilterSize(5);
    sbm->setPreFilterCap(61);
    sbm->setMinDisparity(9);
    sbm->setTextureThreshold(507);
    sbm->setUniquenessRatio(0);
    sbm->setSpeckleWindowSize(0);
    sbm->setSpeckleRange(8);
    sbm->setDisp12MaxDiff(1);
    sbm->compute(img, img2, disp);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    sgbm->setNumDisparities(192);
    sgbm->setPreFilterCap(4);
    sgbm->setMinDisparity(-64);
    sgbm->setUniquenessRatio(1);
    sgbm->setSpeckleWindowSize(150);
    sgbm->setSpeckleRange(2);
    sgbm->setDisp12MaxDiff(10);
    //sgbm->setP1(600);
    //sgbm->setP2(2400);
    sgbm->compute(img, img2, disp);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

    imshow("left", img);
    imshow("right", img2);
    imshow("disp", disp8);
    waitKey(0);
}


vector<vector<Point2f>> faceDetector(const Mat& image) {
    std::vector<Rect> faces;
    const string cascade_name = "/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    if (not face_cascade.load(cascade_name)) {
        cerr << "Cannot load cascade classifier from file: " <<    cascade_name << endl;
    }
    cout << "Casscade classifier loaded" << endl;
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


    const string facemark_filename = "/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/lbfmodel.yml";
    Ptr<Facemark> facemark = createFacemarkLBF();
    facemark->loadModel(facemark_filename);
    vector<vector<Point2f>> shapes;
    facemark->fit(image, faces, shapes);
    return shapes;
}


int main() {
    cout << "ok" << endl;
    Mat img = imread("/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/face point clouds + images/image_left_1.jpg", 1);
    vector<Rect> faces;
    vector<vector<Point2f>> shapes = faceDetector(img);
    cout << "done" << endl;
    cout << shapes.at(0).size() << endl;

    // Draw the detected landmarks
    for (int j=0; j<18; j++){
        circle(img, Point(shapes.at(0).at(j).x, shapes.at(0).at(j).y), 4, Scalar(255, 0, 255), 5);
    }
    for (int j=17; j<27; j++){
        circle(img, Point(shapes.at(0).at(j).x, shapes.at(0).at(j).y), 4, Scalar(0, 255, 255), 5);
    }
    for (int j=28; j<37; j++){
        circle(img, Point(shapes.at(0).at(j).x, shapes.at(0).at(j).y), 4, Scalar(0, 255, 0), 5);
    }
    for (int j=36; j<48; j++){
        circle(img, Point(shapes.at(0).at(j).x, shapes.at(0).at(j).y), 4, Scalar(0, 255, 0), 5);
    }
    for (int j=48; j<68; j++){
        circle(img, Point(shapes.at(0).at(j).x, shapes.at(0).at(j).y), 4, Scalar(0, 0, 255), 5);
    }


    imshow("img", img);
    waitKey(0);

    return 0;
}
