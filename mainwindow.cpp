#include "mainwindow.h"
#include "ui_mainwindow.h"

/* Change / reset size array to 6 and to reset on overflow???
 * Rename functions for more clean representation of code
 * Add pass by reference arguments using & for good practice
 */

//==============================================================
// MAINWINDOW CONSTRUCTOR
//==============================================================

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->meanFace->setScene( eigen );             // eigen faces
    ui->graphicsView->setScene( scene );         // live webcam
    ui->displayFaces->setScene( display );       // face vector

    ui->graphicsView->scene()->addItem(&pixmap); // webcam item

    eigen->setSceneRect(0,0,350,350);
    display->setSceneRect(0,0,1200,200);
}

//==============================================================
// SETUP MAINWINDOW
//==============================================================

void MainWindow::windowSetup()
{
    initCascadeClassifiers();              // load haar cascades
    initBooleanValues();                   // set boolean values
    initWebcam();                          // setup live webcam
}

//==============================================================
// INITIALIZERS
//==============================================================

void MainWindow::initCascadeClassifiers()
{
    // load haar face cascade xml file
    faceRec.loadFaceCascadeClassifier();
}

void MainWindow::initBooleanValues()
{
    pause = false;                  // pause webcam image
    showEigen = false;              // show matching eigen
    faceFound = false;              // face found in webcam
    matchFound = false;             // match found for image
}

void MainWindow::initWebcam()
{
    webcam.open(0);
    if ( !webcamError() ) liveWebcam = true;

    while ( liveWebcam )
    {
        // get webcam frame (rgb Mat)
        if ( !pause ) webcam >> frame;
        qFrame = rgbMat2QImage( frame );

        // map webcam frame, fit to display, & find face
        pixmap.setPixmap( QPixmap::fromImage( qFrame ) );
        ui->graphicsView->fitInView( &pixmap );
        findFace();

        // ensure GUI thread & other events are processed
        qApp->processEvents();
    }
}

//==============================================================
// STORING & DISPLAYING FOUND FACES
//==============================================================

void MainWindow::addFace() // LOAD CAPTURED IMAGE INTO FACE ARRAY
{
    size_t i = faceRec.facesVector.size() - 1;
    QImage img = rgbMat2QImage( faceRec.facesVector[i] );

    display->addPixmap(QPixmap::fromImage(img))->moveBy(200*(i%6),0);
}

void MainWindow::on_meanFace_windowIconChanged() // eigen window
{
    QImage img;
    if ( !showEigen ) img = grayMat2QImage( faceRec.getMeanFace() );
    else img = grayMat2QImage( faceRec.getEigenFace( faceIndex ) );

    eigen->addPixmap( QPixmap::fromImage( img ) );
}


//==============================================================
// ACTION BUTTONS
//==============================================================

void MainWindow::on_CAPTURE_pressed()      // store webcam face
{
    if ( !faces.empty() )
    {
        Mat img = faceRec.getFaceImg( boundFace(), frame );
        faceRec.pushBackMat( img );
        showEigen = false;
        addFace();

        on_meanFace_windowIconChanged();
    }
}

void MainWindow::on_whoAmI_pressed()       // match webcam face
{
    Mat img = faceRec.getFaceImg( boundFace(), frame );
    faceIndex = faceRec.whoAmI( img );
    setMatchedFace( faceIndex );
    showEigen = true;

    on_meanFace_windowIconChanged();
}

//==============================================================
// PAINT EVENTS
//==============================================================

void MainWindow::paintEvent(QPaintEvent *event)
{
    // Clear bounding face rectangle and reposition new rectangle
    clr( scene, myFaceRect, faceFound );
    if ( !faces.empty() )
    {
        myFaceRect = scene->addRect( boundFace(),
                                     QPen( QColor(Qt::red), 8 )
                                     );
        faceFound = true;
    }
    else faceFound = false;

    // Clear bounding match rectangle and reposition new rectangle
    clr( display, myMatchRect, matchFound );
    myMatchRect = display->addRect( matchedFace,
                                    QPen( QColor(Qt::green), 10 )
                                    );
    matchFound = true;

    Q_UNUSED(event);
    update();
}

//==============================================================
// WEBCAM FRAME METHODS
//==============================================================

void MainWindow::findFace() // FIND FACE FROM WEBCAM FRAME
{
    faceRec.faceCascade.detectMultiScale( frame,
                                          faces,
                                          1.1,
                                          2,
                                          0|CV_HAAR_SCALE_IMAGE,
                                          Size( 200, 200 ) //***
                                          );
}

void MainWindow::setMatchedFace(int index) // MATCH FACE ARRAY
{
    matchedFace = QRectF( 200 * ( index % 6 ) , 0, 200, 200 );
}

QRectF MainWindow::boundFace(size_t i) // BOUND WEBCAM FRAME
{
    return( QRectF( faces[i].x,
                    faces[i].y,
                    faces[i].width,
                    faces[i].height
                    ) );
}

void MainWindow::clr( QGraphicsScene * s_in,
                      QGraphicsRectItem * i_in,
                      bool b_in
                      )
{
    if ( !s_in->items().isEmpty() && b_in ) s_in->removeItem(i_in);
}

//==============================================================
// CONVERSIONS (cv swaps r & b values by default)
// CV_RGB2GRAY ; CV_BGR2GRAY ; ...
//==============================================================

QImage MainWindow::rgbMat2QImage(Mat img)
{
    cv::Mat src;
    cv::cvtColor( img, src, CV_BGR2RGB);

    return( QImage( src.data,
                    src.cols,
                    src.rows,
                    int(src.step),
                    rgbForm
                    ) );
}

QImage MainWindow::grayMat2QImage(Mat img)
{
    return( QImage( img.data,
                    img.cols,
                    img.rows,
                    int(img.step),
                    grayForm
                    ) );
}

//==============================================================
// TOGGLE BUTTONS
//==============================================================

void MainWindow::on_PAUSE_pressed() // Toggle the pause button
{
    if ( pause == true ) pause = false;
    else pause = true;
}

//==============================================================
// START & END BUTTONS
//==============================================================

void MainWindow::on_START_pressed()
{
    windowSetup();
    qWarning("SUCCESS: Program initiated...");
}

void MainWindow::on_TERMINATE_pressed()
{
    liveWebcam = false;
    webcam.release();
    QApplication::quit();
    qWarning("SUCCESS: Program terminated successfully!");
}

//==============================================================
// ERROR CHECKING
//==============================================================

bool MainWindow::webcamError() // WEBCAM ERROR
{
    if ( !webcam.isOpened() )
    {
        qWarning("ERROR: VideoCapture is NOT working");
        return true;
    }
    else
    {
        qWarning("SUCCESS: VideoCapture is working...");
        return false;
    }
}

//==============================================================
// MAINWINDOW DESTRUCTOR
//==============================================================

MainWindow::~MainWindow()
{
    delete ui;
}
