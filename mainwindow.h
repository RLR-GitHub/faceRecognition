#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QDebug>
#include <QImage>
#include <QPainter>
#include <QMainWindow>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>

#include <facerec.h>
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    FaceRec faceRec;

private:
    Ui::MainWindow *ui;

    cv::VideoCapture webcam;
    std::vector<Rect> faces;

    int faceIndex;

    bool liveWebcam;
    bool matchFound;
    bool faceFound;
    bool showEigen; // false until whoAmI clicked
    bool pause;

    QImage::Format grayForm = QImage::Format_Grayscale8;
    QImage::Format rgbForm = QImage::Format_RGB888;

    QGraphicsScene *display = new QGraphicsScene;
    QGraphicsScene *eigen = new QGraphicsScene;
    QGraphicsScene *scene = new QGraphicsScene;
    QGraphicsRectItem *myMatchRect;
    QGraphicsRectItem *myFaceRect;

    QGraphicsPixmapItem pixmap;
    QRectF matchedFace;
    QImage qFrame;
    Mat frame;

    // my private methods
    bool webcamError();

    QRectF boundFace(size_t i = 0);
    QImage rgbMat2QImage(Mat img);
    QImage grayMat2QImage(Mat img);

    void addFace();
    void findFace();
    void initWebcam();
    void windowSetup();
    void initBooleanValues();
    void initCascadeClassifiers();
    void setMatchedFace(int index);
    void paintEvent(QPaintEvent *event);
    void clr(QGraphicsScene * s_in, QGraphicsRectItem * i_in, bool b_in);

private slots:
    void on_START_pressed();              // start  button
    void on_PAUSE_pressed();              // pause display
    void on_whoAmI_pressed();             // match display
    void on_CAPTURE_pressed();            // capture image
    void on_TERMINATE_pressed();          // finish button
    void on_meanFace_windowIconChanged(); // eigen display
};

#endif // MAINWINDOW_H
