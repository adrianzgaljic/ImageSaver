#include <iostream>
#include <fstream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/face.hpp"


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


int main() {
    cout << "ok" << endl;
    Mat img = imread("/Users/adrianzgaljic/Desktop/NERO/pointclouds/ensenso snimke fantoma/test 5.2.2020./r.png", 1);


    vector<Rect> faces;
    vector<vector<Point2f>> shapes = faceDetector(img);


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
