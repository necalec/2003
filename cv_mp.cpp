#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <omp.h>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

Vec3b gray_funk(Vec3b color) {
    float gray = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2];
    uchar gray_value = static_cast<uchar>(gray);
    return Vec3b(gray_value, gray_value, gray_value);
}
Vec3b sepia_funk(Vec3b color) {
    float tb = 0.272 * color[0] + 0.534 * color[1] + 0.131 * color[2];
    if (tb > 255) tb = 255;
    float tr = 0.393 * color[0] + 0.769 * color[1] + 0.189 * color[2];
    if (tr > 255) tr = 255;
    float tg = 0.349 * color[0] + 0.686 * color[1] + 0.168 * color[2];
    if (tg > 255) tg = 255;
    uchar tb_value = static_cast<uchar>(tb);
    uchar tr_value = static_cast<uchar>(tr);
    uchar tg_value = static_cast<uchar>(tg);
    return Vec3b(tb_value, tg_value, tr_value);
}
Vec3b negativ_funk(Vec3b color) {
    float tb = 255 - color[0];
    float tg = 255 - color[1];
    float tr = 255 - color[2];
    uchar tb_value = static_cast<uchar>(tb);
    uchar tr_value = static_cast<uchar>(tr);
    uchar tg_value = static_cast<uchar>(tg);
    return Vec3b(tb_value, tg_value, tr_value);
}

Mat sobel_funk(const Mat& src) {
    int rows = src.rows;
    int cols = src.cols;

    Mat grad_x(rows, cols, CV_32F, Scalar(0));
    Mat grad_y(rows, cols, CV_32F, Scalar(0));

    float sobel_x[3][3] = { {-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1} };

    float sobel_y[3][3] = { {1, 2, 1},
                             {0, 0, 0},
                             {-1, -2, -1} };

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            float x = 0, y = 0;
            for (int m = -1; m <= 1; ++m) {
                for (int n = -1; n <= 1; ++n) {
                    x += src.at<uchar>(i + m, j + n) * sobel_x[m + 1][n + 1];
                    y += src.at<uchar>(i + m, j + n) * sobel_y[m + 1][n + 1];
                }
            }
            grad_x.at<float>(i, j) = x;
            grad_y.at<float>(i, j) = y;
        }
    }

    Mat value;
    value = abs(grad_x) + abs(grad_y);
    value = value.sqrt;

    normalize(value, value, 0, 255, NORM_MINMAX, CV_8U);

    return value;
}

int main()
{
    string way = "C:/Users/necal/Downloads/cokacola.jpg";

    Mat main = imread(way, 1);

    if (main.empty()) {
        cout << "nothing!" << endl;
        return 0;
    }
    else {
        cout << "norm" << endl;
    }

    Mat gray(main.rows, main.cols, CV_8UC3);
    Mat sepia(main.rows, main.cols, CV_8UC3);
    Mat negativ(main.rows, main.cols, CV_8UC3);
    Mat sobel(main.rows, main.cols, CV_8UC3);
    Mat sobel_2(main.rows, main.cols, CV_8UC3);

    sobel = sobel_funk(main);

    for (int i = 0; i < main.rows; i++) {
        for (int j = 0; j < main.cols; j++) {
            gray.at<Vec3b>(i, j) = gray_funk(main.at<Vec3b>(i, j));
            sepia.at<Vec3b>(i, j) = sepia_funk(main.at<Vec3b>(i, j));
            negativ.at<Vec3b>(i, j) = negativ_funk(main.at<Vec3b>(i, j));
            sobel_2.at<Vec3b>(i, j) = negativ_funk(sobel.at<Vec3b>(i, j));
        }
    }

    namedWindow("1", WINDOW_NORMAL);
    imshow("1", gray);

    namedWindow("2", WINDOW_NORMAL);
    imshow("2", sepia);

    namedWindow("3", WINDOW_NORMAL);
    imshow("3", negativ);

    namedWindow("4", WINDOW_NORMAL);
    imshow("4", sobel);

    namedWindow("5", WINDOW_NORMAL);
    imshow("5", sobel_2);

    waitKey(0);

    return 0;
}