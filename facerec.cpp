#include "facerec.h"

//==============================================================
// FaceRec constructor
//==============================================================

FaceRec::FaceRec()
{
}

//==============================================================
// WHO AM I ???
//==============================================================

int FaceRec::whoAmI(cv::Mat img_in) // predicting the face match
{
    return( getFaceRecognizerPredictedFace( img_in ) );
}

//==============================================================
// PUSH_BACK FACE VECTOR METHODS
//==============================================================

void FaceRec::pushBackMat( cv::Mat colorImg )
{
    grayFace = rgbMat2GrayMat( colorImg );

    facesVector.push_back( colorImg );  
    grayVector.push_back( grayFace );
    labels.push_back( faceCount );

    initFaceRecognizer();
    trainFaceRecognizer();

    setFaceRecognizerMean();
    setFaceRecognizerEigenvectors();

    faceCount++;
}

//==============================================================
// EigenFaceRecognizer
//==============================================================

void FaceRec::initFaceRecognizer(int i, double d)
{
    // INITIALIZE FACERECOGNIZER POINTER
    // set faceRecognizer ptr to created eigen face recognizer
    // create( int principal components, double threshold value )

    faceRecognizer = cv::face::EigenFaceRecognizer::create( i, d );
}

void FaceRec::trainFaceRecognizer()
{
    // TRAIN FACERECOGNIZER
    // using vector of faces after each capture

    faceRecognizer->train( grayVector, labels );
}

void FaceRec::setFaceRecognizerMean()
{
    // GET AVERAGE FACE
    // return normaized mean from raceRecognizer

    meanFace = norm( faceRecognizer->getMean() );
}

void FaceRec::setFaceRecognizerEigenvalues()
{
    cv::Mat eigenvalues = faceRecognizer->getEigenValues();
}

void FaceRec::setFaceRecognizerEigenvectors()
{
    // SAVE EIGEN FACES
    cv::Mat eigenvectors = faceRecognizer->getEigenVectors();

    eigenVector.clear();
    for ( int i = 0; i < cv::min(10, eigenvectors.cols); i++ )
    {
        // get eigenvector #i
        cv::Mat ev = eigenvectors.col(i).clone();

        // reshape to original size & normalize to [0 ... 255]
        cv::Mat grayscale = norm( ev );
        eigenVector.push_back(grayscale);
    }
}

int FaceRec::getFaceRecognizerPredictedFace(cv::Mat img_in)
{
    cv::Mat grayscale = rgbMat2GrayMat( img_in );
    return( faceRecognizer->predict( grayscale ) );
}

//==============================================================
// Normalize Images from their vectorized form
//==============================================================

cv::Mat FaceRec::norm(cv::InputArray arr_in) // normalize image
{
    cv::Mat src = arr_in.getMat();
    cv::Mat dst;

    switch( src.channels() )
    {
    case 1:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        break;
    case 2:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }

    return( dst.reshape(1,200) );
}

//==============================================================
// IMAGE CONVERSIONS, RESHAPING, RESIZING, RETURNING
//==============================================================

cv::Mat FaceRec::getFaceImg(QRectF r, cv::Mat frame)
{
    // get face bounding rect and establish region of interest
    cv::Rect ROI( int(r.x()),
                  int(r.y()),
                  int(r.width()),
                  int(r.height())
                  );

    // crop frame to ROI; resize; recolor
    cv::Mat croppedImg(frame, ROI);
    cv::resize(croppedImg, croppedImg, cv::Size(200,200));

    return( croppedImg );
}

cv::Mat FaceRec::getMeanFace()
{
    cv::Mat img = meanFace;
    cv::resize( img, img, cv::Size(350,350) );
    return( img );
}

cv::Mat FaceRec::getEigenFace(int i)
{
    size_t index = size_t( i );
    cv::Mat img = eigenVector[ index ];
    cv::resize( img, img, cv::Size(350,350) );
    return( img );
}

cv::Mat FaceRec::rgbMat2GrayMat(cv::Mat src)
{
    cv::Mat dst;
    cv::cvtColor( src, dst, CV_RGB2GRAY);
    return(dst);
}

//==============================================================
// LOAD HAAR CASCADE CLASSIFIERS (face & kat)
//==============================================================

void FaceRec::loadFaceCascadeClassifier()
{
    QTemporaryFile temp;
    QFile xml("/Users/MrChieferton/Desktop/haarcascade_frontalface_alt.xml");

    if ( xml.open(QFile::ReadOnly | QFile::Text) && temp.open() )
    {
        temp.write( xml.readAll() );
    }
    if ( !faceCascade.load (temp.fileName().toStdString()) )
    {
        qWarning("ERROR: cannot load face haar cascade files!");
    }
    else
    {
        qWarning("SUCCESS: Haar Face Cascade Classifier files loaded!");
    }
}
