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


void showDisparity(Mat img, Mat img2) {
    Mat disp, disp8;
    /*
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
    normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);
    */

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,32,3);

    sgbm->setNumDisparities(96);

    sgbm->setPreFilterCap(4);
    sgbm->setMinDisparity(10);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(150);
    sgbm->setSpeckleRange(2);
    sgbm->setDisp12MaxDiff(10);

    //sgbm->setP1(600);
    //sgbm->setP2(2400);
    sgbm->compute(img, img2, disp);
    normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);

    imshow("left", img);
    imshow("right", img2);
    imshow("disp", disp8);
    waitKey(0);
}


vector<vector<Point2f>> faceDetector(const Mat& image) {
    imshow("face", image);
    std::vector<Rect> faces;
    const string cascade_name = "/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    if (not face_cascade.load(cascade_name)) {
        cerr << "Cannot load cascade classifier from file: " <<    cascade_name << endl;
    }
    Mat img = Mat::zeros(200, 200, CV_8UC3);
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
            pointColor[0] = 255;
            pointColor[1] = 0;
            pointColor[2] = 0;
        }
    }
}


Point3f reprojectDot(Point2i point, Mat disparity){

    float Q03 = -509.091339111328125;
    float Q13 = -512.0567626953125;
    float Q23 = 1465.2501220703125;
    float Q32 = 0.009868702407;
    float Q33 = 2.50484309702818564;
    Point3f pointOut;
    // Transforming a single-channel disparity map to a 3-channel image representing a 3D surface
    //cout << disparity << endl;
    const uchar* disp_ptr = disparity.ptr<uchar>(point.y);
    const float w = (int)disp_ptr[point.x] * Q32 + Q33;
    pointOut.x = (static_cast<float>(point.x) + Q03)/w;
    pointOut.y = (static_cast<float>(point.y) + Q13)/w;
    pointOut.z =  Q23/w;

    return pointOut;
}



int main() {

    Mat disparity = imread("/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/face point clouds + images/manual.png", 0);
    Mat img = imread("/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/face point clouds + images/manual_disparity/manual_right.png", 1);
    Mat l = imread("/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/face point clouds + images/manual_disparity/manaul_left.png", 0);
    Mat r = imread("/Users/adrianzgaljic/Desktop/doktorski/point clouds and images/face point clouds + images/manual_disparity/manual_right.png", 0);

    Matx44d Q = cv::Matx44d(
            1.0, 0.0, 0.0,     -509.091339111328125,
            0.0, 1.0, 0.0, -512.0567626953125,
            0.0, 0.0, 0.0, 1465.2501220703125,
            0.0, 0.0, 0.009868702407, 2.50484309702818564
    );


    Mat xyz;
    Mat color;
    //customReproject(disparity, xyz, color);



    std::ofstream outFile("output_features.ply");


    if (!outFile.is_open())
    {
        std::cerr << "ERROR: Could not open "  << std::endl;
    }

    outFile << "ply" << endl;
    outFile << "format ascii 1.0" << endl;
    outFile << "element vertex "  <<  68 << endl; //xyz.rows*xyz.cols << endl; //xyz.rows*xyz.cols << endl;
    outFile << "property float x" << endl;
    outFile << "property float y" << endl;
    outFile << "property float z" << endl;
    outFile << "property uchar red" << endl;
    outFile << "property uchar green" << endl;
    outFile << "property uchar blue" << endl;
    outFile << "end_header" << endl;

    /*
    float minX = 99999;
    float minY = 99999;
    float minZ = 99999;
    float maxX = -999;
    float maxY = -999;
    float maxZ = -999;
    for (int i=0; i<xyz.rows; i++){
        const cv::Vec3f* image3D_ptr = xyz.ptr<cv::Vec3f>(i);
        const cv::Vec3f* imageColor_ptr = color.ptr<cv::Vec3f>(i);
        for (int j = 0; j < xyz.cols; j++){
            Scalar colour = r.at<uchar>(Point(j, i));
            //cout << "color: " << colour.val[0] << endl;
            float p_x = image3D_ptr[j][0];
            float p_y = image3D_ptr[j][1];
            float p_z = image3D_ptr[j][2];
            if (p_x > maxX){
                maxX = p_x;
            }
            if (p_x < minX){
                minX = p_x;
            }
            if (p_y > maxY){
                maxY = p_y;
            }
            if (p_y < minY){
                minY = p_y;
            }
            if (p_z > maxZ){
                maxZ = p_z;
            }
            if (p_z < minZ){
                minZ = p_z;
            }
            outFile << -image3D_ptr[j][0] << " " << -image3D_ptr[j][1] << " " << -image3D_ptr[j][2] << " " << colour.val[0] << " " << colour.val[0] << " " << colour.val[0] << std::endl;

            //outFile << image3D_ptr[j][0] << " " << image3D_ptr[j][1] << " " << image3D_ptr[j][2] << " " << imageColor_ptr[j][0] << " " << imageColor_ptr[j][1] << " " << imageColor_ptr[j][2] << std::endl;
        }
    }
    cout << "min max" << maxX-minX << ",   " << maxY - minY << ",   " << maxZ-minZ << endl;
    */
    vector<vector<Point2f>> shapes = faceDetector(img);
    //cout << shapes.at(0).size() << endl;
    Point3f point;
    for (int i=0; i<68; i++){
        //cout << "points: " << shapes.at(0).at(i).x << ", " << shapes.at(0).at(i).y << endl;
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
        point = reprojectDot(Point2i(shapes.at(0).at(i).x, shapes.at(0).at(i).y), disparity);
        cout << "point " <<i << ": " << point.x << "," << point.y << "," << point.z << endl;
        outFile << -point.x << " " << -point.y << " " << -point.z << " " << 0 << " " << 0 << " " << 255 << std::endl;
    }
    outFile.close();


    imshow("img", disparity);
    imshow("imgL", img);

    waitKey(0);


}