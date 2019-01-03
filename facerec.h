#ifndef FACEREC_H
#define FACEREC_H
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <QtMultimedia>
#include <QFile>

class FaceRec
{
private:
    // my private variables
    // Smart EigenFaceRecognizer pointer "faceRecognizer"
    cv::Ptr<cv::face::EigenFaceRecognizer> faceRecognizer;

    std::vector<cv::Mat> eigenVector;
    std::vector<cv::Mat> grayVector;
    std::vector<  int  > labels;       // photo index

    //cv::Mat grayFace;
    cv::Mat meanFace;
    cv::Mat grayFace;

    int faceCount = 0;

    // my private methods
    cv::Mat norm(cv::InputArray arr_in); //normalize images
    cv::Mat rgbMat2GrayMat(cv::Mat src);

    int getFaceRecognizerPredictedFace(cv::Mat img_in);

    void initFaceRecognizer(int i = 0, double d = DBL_MAX);
    void setFaceRecognizerEigenvectors();
    void setFaceRecognizerEigenvalues();
    void setFaceRecognizerMean();
    void trainFaceRecognizer();

public:
    // my public variables
    // Haar face cascade classifier
    cv::CascadeClassifier faceCascade;
    std::vector<cv::Mat> facesVector;

    // my public methods
    FaceRec();

    cv::Mat getFaceImg(QRectF boundingRect, cv::Mat frame);
    cv::Mat getEigenFace(int i);
    cv::Mat getMeanFace();

    int whoAmI(cv::Mat img_in);

    void pushBackMat(cv::Mat img_in);
    void loadFaceCascadeClassifier();
};

#endif // FACEREC_H
