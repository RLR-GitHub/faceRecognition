#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    //from the QApplication determine primary screen size
    QApplication app(argc, argv);
    int height = app.primaryScreen()->geometry().height();
    int width = app.primaryScreen()->geometry().width();

    //from the MainWindow display the user interface
    MainWindow window;
    window.resize(width, height);
    window.show();

    return app.exec();
}
