#include <iostream>


#include "opencv4/opencv2/opencv.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

int main(){
    int width = 1000;
    int height = 1000;

    Mat map = imread("111.png");

    Mat mapr;

    resize(map, mapr, Size(900, 300));

    int a = 0;
    int b = 0;

    int c = 0;
    int d = 900;

    int e = 0;
    int f = 0;

    int img_h = map.rows;
    int img_w = map.cols;

    for(int i=0; i<32; i++){
        //a++
        line(mapr, Point(a, b), Point(a, d), Scalar(255, 255, 255), 2, 8);
        a = a + 30;

        //e++
        line(mapr, Point(c, e), Point(d, e), Scalar(255, 255,255), 2, 8);
        e = e + 30;
    }

    circle(mapr, Point(30, 270), 5, Scalar(0,0, 255), 5, 8);

    cout<<"before"<<endl;

    line(mapr, Point(30, 0), Point(30, 270), Scalar(0, 0, 255), 2, 18);

    line(mapr, Point(30, 270), Point(900, 270), Scalar(0, 0, 255), 2, 18);
    imshow("Map", mapr);
    imwrite("Map_check.png", mapr);


    waitKey(0);

    cout<<"after"<<endl;



    return 0;
}