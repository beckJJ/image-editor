#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <tchar.h>
#include <Windows.h>
#include <CommCtrl.h>
#include <conio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define PROC_INFO_MENU 1
#define PROC_OPEN_FILE 2
#define PROC_COPY_IMG 3
#define PROC_H_FLIP 4
#define PROC_V_FLIP 5
#define PROC_H_MIRROR 6
#define PROC_V_MIRROR 7
#define PROC_LUMINANCE 8
#define PROC_QUANTIZATION 9
#define PROC_EXIT_MENU 10
#define PROC_SAVE_FILE 11
#define PROC_HISTOGRAM 12
#define PROC_BRIGHTNESS 13
#define PROC_CONTRAST 14
#define PROC_NEGATIVE 15
#define PROC_HISTOGRAM_EQ 16
#define PROC_OPEN_TARGET 17
#define PROC_HISTOGRAM_MATCH 18
#define PROC_ZOOM_OUT 19
#define PROC_ZOOM_IN 20
#define PROC_ROTATE_CCW 21
#define PROC_ROTATE_CW 22
#define PROC_CONV_GAUSS 23
#define PROC_CONV_LAPLACE 24
#define PROC_CONV_GEN_HIGH 25
#define PROC_CONV_PRE_HX 26
#define PROC_CONV_PRE_HY 27
#define PROC_CONV_SOB_HX 28
#define PROC_CONV_SOB_HY 29
#define PROC_CONV_CUSTOM 30

#define IMG_OPEN_FILE 1
#define IMG_COPY 2
#define IMG_H_FLIP 3
#define IMG_V_FLIP 4
#define IMG_H_MIRROR 5
#define IMG_V_MIRROR 6
#define IMG_LUMINANCE 7
#define IMG_QUANTIZATION 8
#define IMG_HISTOGRAM 9
#define IMG_BRIGHTNESS 10
#define IMG_CONTRAST 11
#define IMG_NEGATIVE 12
#define IMG_HISTOGRAM_EQ 13
#define IMG_HISTOGRAM_MATCH 14
#define IMG_ZOOM_OUT 15
#define IMG_ZOOM_IN 16
#define IMG_ROTATE_CCW 17
#define IMG_ROTATE_CW 18
#define IMG_CONV_GAUSS 19
#define IMG_CONV_LAPLACE 20
#define IMG_CONV_GEN_HIGH 21
#define IMG_CONV_PRE_HX 22
#define IMG_CONV_PRE_HY 23
#define IMG_CONV_SOB_HX 24
#define IMG_CONV_SOB_HY 25
#define IMG_CONV_CUSTOM 36

#define OPEN_SRC 1
#define OPEN_TARGET 2

#define ROT_CCW 1
#define ROT_CW 2

#define BUFFER_SIZE 100
#define NORMALIZATION_CONSTANT 200.0
#define HIST_MARGIN_TOP 50
#define HIST_MARGIN_LEFT 20

using namespace cv;

// Global variables
HWND quantBox;
HWND brightnessBox;
HWND contrastBox;
HWND zoomOut_sx;
HWND zoomOut_sy;
HWND conv[3][3];
char filename[FILENAME_MAX];
Mat img, img2;
Mat targetImg;

int luminance(Vec3b px)
{ //returns the luminance of the px received
    float lum;
    lum = (float)((0.299 * (float)px[0])
        + (0.587 * (float)px[1])
        + (0.114 * (float)px[2]));
    return (int)lum;
}

void grayscale(Mat dest, const Mat src)
{
    int lum;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            lum = luminance(src.at<Vec3b>(i, j));
            for (int k = 0; k < 3; k++) {
                dest.at<Vec3b>(i, j)[k] = lum;
            }
        }
    }
}

void quant(Mat dest, const Mat src, int tones)
{
    int lum = luminance(src.at<Vec3b>(0, 0));
    int lum_max = lum;
    int lum_min = lum;
    int tam_int;
    int tb;
    int t_ini, t_final;

    // find lum_max and lum_min
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            lum = luminance(src.at<Vec3b>(i, j));
            if (lum_max < lum) {
                lum_max = lum;
            }
            else if (lum_min > lum) {
                lum_min = lum;
            }
        }
    }
    tam_int = (lum_max - lum_min + 1);
    if (tones >= tam_int || tones <= 0) {
        grayscale(dest, src);
        return;
    }
    tb = tam_int / tones;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            lum = luminance(src.at<Vec3b>(i, j));
            for (int k = 0;; k++) {
                t_ini = (int)(lum_min - 0.5 + (k * tb));
                t_final = (int)(lum_min - 0.5 + ((k + 1) * tb));
                if (lum >= t_ini && lum < t_final) {
                    int new_lum = (t_ini + t_final) / 2;
                    for (int l = 0; l < 3; l++) {
                        dest.at<Vec3b>(i, j)[l] = new_lum;
                    }
                    break;
                }
            }
        }
    }
}

void imgcpy(Mat dest, const Mat src)
{
    for (int i = 0; i < src.rows; i++) {
        dest.row(i) = (src.row(i) + 0);
    }
}

void hflip(Mat dest, const Mat src)
{
    for (int i = 0; i < src.cols; i++) {
        dest.col(i) = (src.col(src.cols - i - 1) + 0);
    }
}

void vflip(Mat dest, const Mat src)
{
    for (int i = 0; i < src.rows; i++) {
        dest.row(i) = (src.row(src.rows - i - 1) + 0);
    }
}

void printHistogram(float hist[256], char *window_name)
{
    Mat histogram(300, 300, CV_8UC3);
    for (int i = 0; i < histogram.rows; i++) {
        for (int j = 0; j < histogram.cols; j++) {
            for (int k = 0; k < 3; k++) {
                histogram.at<Vec3b>(i, j)[k] = 255;
            }
        }
    }
    for (int i = 0; i < 256; i++) {
        while (hist[i] > 0) {
            histogram.at<Vec3b>((int)(NORMALIZATION_CONSTANT - hist[i]) + HIST_MARGIN_TOP,
                i + HIST_MARGIN_LEFT)[0] = 0;
            histogram.at<Vec3b>((int)(NORMALIZATION_CONSTANT - hist[i]) + HIST_MARGIN_TOP,
                i + HIST_MARGIN_LEFT)[1] = 0;
            hist[i]--;
        }
    }
    // draw black line below histogram
    for (int i = 0; i < 256; i++) {
        for (int k = 0; k < 3; k++) {
            histogram.at<Vec3b>(252, i + HIST_MARGIN_LEFT)[k] = 0;
        }
    }
    for (int k = 0; k < 3; k++) {
        histogram.at<Vec3b>(253, HIST_MARGIN_LEFT)[k] = 0;
        histogram.at<Vec3b>(254, HIST_MARGIN_LEFT)[k] = 0;
        histogram.at<Vec3b>(255, HIST_MARGIN_LEFT)[k] = 0;

        histogram.at<Vec3b>(253, HIST_MARGIN_LEFT + 128)[k] = 0;
        histogram.at<Vec3b>(254, HIST_MARGIN_LEFT + 128)[k] = 0;
        histogram.at<Vec3b>(255, HIST_MARGIN_LEFT + 128)[k] = 0;

        histogram.at<Vec3b>(253, HIST_MARGIN_LEFT + 255)[k] = 0;
        histogram.at<Vec3b>(254, HIST_MARGIN_LEFT + 255)[k] = 0;
        histogram.at<Vec3b>(255, HIST_MARGIN_LEFT + 255)[k] = 0;
    }
    // draw numbers below the line
    putText(histogram, "0", Point(15, 270), FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 0);
    putText(histogram, "128", Point(133, 270), FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 0);
    putText(histogram, "255", Point(260, 270), FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 0);
    cv::imshow(window_name, histogram);
    cv::waitKey(1);
}

void histogram(const Mat src, char *window_name)
{
    float hist[256] = { 0 };
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            hist[luminance(src.at<Vec3b>(i, j))]++;
        }
    }
    // encontrar maior elemento
    float biggest = 0;
    for (int i = 0; i < 256; i++) {
        if (hist[i] > biggest) {
            biggest = hist[i];
        }
    }
    if (biggest == 0) { // divisao por 0
        return;
    }
    // normzalizar
    for (int i = 0; i < 256; i++) {
        hist[i] /= biggest;
        hist[i] *= NORMALIZATION_CONSTANT;
    }

    printHistogram(hist, window_name);
}

void brightness(Mat dest, const Mat src, int key)
{
    if (key > 255 || key < -255) {
        return;
    }
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                if (dest.at<Vec3b>(i, j)[k] + key < 0) {
                    dest.at<Vec3b>(i, j)[k] = 0;
                }
                else if (dest.at<Vec3b>(i, j)[k] + key > 255) {
                    dest.at<Vec3b>(i, j)[k] = 255;
                }
                else {
                    dest.at<Vec3b>(i, j)[k] += key;
                }
            }
        }
    }
}

void contrast(Mat dest, const Mat src, float key)
{
    if (key <= 0 || key > 255) {
        return;
    }
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                if (dest.at<Vec3b>(i, j)[k] * key < 0) {
                    dest.at<Vec3b>(i, j)[k] = 0;
                }
                else if (dest.at<Vec3b>(i, j)[k] * key > 255) {
                    dest.at<Vec3b>(i, j)[k] = 255;
                }
                else {
                    dest.at<Vec3b>(i, j)[k] *= key;
                }
            }
        }
    }
}

void negative(Mat dest, const Mat src)
{
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dest.at<Vec3b>(i, j)[k] = 255 - dest.at<Vec3b>(i, j)[k];
            }
        }
    }
}

void histogramEq(Mat dest, const Mat src)
{
    double hist[256] = { 0 };
    double hist_cum[256] = { 0 };
    double alfa;
    int numPx = 0;
    Mat graySrc(src.size(), src.type());

    grayscale(graySrc, src);
    // histogram
    for (int i = 0; i < graySrc.rows; i++) {
        for (int j = 0; j < graySrc.cols; j++) {
            hist[luminance(graySrc.at<Vec3b>(i, j))]++;
            numPx++;
        }
    }
    // scaling factor
    alfa = 255.0 / numPx;
    // cumulative histogram
    hist_cum[0] = (alfa * hist[0]);
    for (int i = 1; i < 256; i++) {
        hist_cum[i] = hist_cum[i - 1] + alfa * hist[i];
    }
    for (int i = 0; i < graySrc.rows; i++) {
        for (int j = 0; j < graySrc.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dest.at<Vec3b>(i, j)[k] = hist_cum[graySrc.at<Vec3b>(i, j)[k]];
            }
        }
    }
}

int findTargetShadeLevel(const float hist[256], int shade)
{
    if (hist[shade]) {
        return hist[shade];
    }
    else {
        int i = 0;
        while (i < 256) {
            if ((shade + i <= 255) && hist[shade + i]) {
                return hist[shade + i];
            }
            else if ((shade - i >= 0) && hist[shade + i]) {
                return hist[shade - i];
            }
            else {
                i++;
            }
        }
    }
}

void histogramMatch(Mat dest, const Mat src, const Mat target)
{
    float hist_src[256] = { 0 };
    float hist_target[256] = { 0 };
    float hist_src_cum[256] = { 0 };
    float hist_target_cum[256] = { 0 };
    float hist_match[256] = { 0 };
    int numPxSrc = 0;
    int numPxTarget = 0;
    float alfaSrc;
    float alfaTarget;
    Mat graySrc(src.size(), src.type());
    Mat grayTarget(target.size(), target.type());

    grayscale(graySrc, src);
    grayscale(grayTarget, target);
    // histogram src
    for (int i = 0; i < graySrc.rows; i++) {
        for (int j = 0; j < graySrc.cols; j++) {
            hist_src[luminance(graySrc.at<Vec3b>(i, j))]++;
            numPxSrc++;
        }
    }
    // histogram target
    for (int i = 0; i < grayTarget.rows; i++) {
        for (int j = 0; j < grayTarget.cols; j++) {
            hist_target[luminance(grayTarget.at<Vec3b>(i, j))]++;
            numPxTarget++;
        }
    }
    // scaling factors
    alfaSrc = 255.0 / numPxSrc;
    alfaTarget = 255.0 / numPxTarget;
    // cumulative histogram src
    hist_src_cum[0] = alfaSrc * hist_src[0];
    for (int i = 1; i < 256; i++) {
        hist_src_cum[i] = hist_src_cum[i - 1] + alfaSrc * hist_src[i];
    }
    // cumulative histogram target
    hist_target_cum[0] = alfaTarget * hist_target[0];
    for (int i = 1; i < 256; i++) {
        hist_target_cum[i] = hist_target_cum[i - 1] + alfaTarget * hist_target[i];
    }
    // find hist_match
    for (int i = 0; i < 256; i++) {
        hist_match[i] = findTargetShadeLevel(hist_target_cum, i);
    }
    // create dest img
    for (int i = 0; i < graySrc.rows; i++) {
        for (int j = 0; j < graySrc.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dest.at<Vec3b>(i, j)[k] = hist_match[graySrc.at<Vec3b>(i, j)[k]];
            }
        }
    }
}

void zoomOut(Mat dest, const Mat src, const int sx, const int sy)
{
    if (!(sx && sy)) {
        return;
    }
    Mat newImg(Size(src.cols / sy, src.rows / sx), src.type());
    int sumPx[3] = { 0 };
    int avgPx[3] = { 0 };

    int i, j, k, l, m;
    int countPx = 0;
    for (i = 0, k = 0; i < src.rows; i += sx, k++) {
        for (j = 0, l = 0; j < src.cols; j += sy, l++) {
            for (int x = 0; x < sx; x++) {
                for (int y = 0; y < sy; y++) {
                    if (((i + x)  < src.rows) && ((j + y) < src.cols)) {
                        for (m = 0; m < 3; m++) {
                            sumPx[m] += src.at<Vec3b>(i + x, j + y)[m];
                        }
                        countPx++;
                    }
                }
            }
            if (countPx) {
                for (m = 0; m < 3; m++) {
                    avgPx[m] = sumPx[m] / countPx;
                }
                if (k < newImg.rows && l < newImg.cols) {
                    for (m = 0; m < 3; m++) {
                        newImg.at<Vec3b>(k, l)[m] = avgPx[m];
                    }
                }
            }
            countPx = 0;
            for (m = 0; m < 3; m++) {
                sumPx[m] = 0;
            }
        }
        countPx = 0;
        for (m = 0; m < 3; m++) {
            sumPx[m] = 0;
        }
    }
    img2 = newImg;
}

void zoomIn(Mat dest, const Mat src)
{
    Mat newImg(Size(src.cols * 2, src.rows * 2), src.type());

    int i, j, k, l, m;
    for (i = 0, k = 0; i < src.rows; i++, k += 2) {
        for (j = 0, l = 0; j < src.cols; j++, l += 2) {
            for (m = 0; m < 3; m++) {
                newImg.at<Vec3b>(k, l)[m] = src.at<Vec3b>(i, j)[m];
            }
        }
    }
    //interpolate rows
    int avg[3] = { 0 };
    for (l = 0; l < newImg.cols; l++) {
        for (k = 0; k < newImg.rows; k++) {
            if (k & 1) {
                if ((k - 1) >= 0 && (k + 1) < newImg.rows) {
                    for (m = 0; m < 3; m++) {
                        avg[m] = (newImg.at<Vec3b>(k - 1, l)[m] + newImg.at<Vec3b>(k + 1, l)[m]) / 2;
                        newImg.at<Vec3b>(k, l)[m] = avg[m];
                    }
                }
                else if ((k - 1) >= 0) {
                    for (m = 0; m < 3; m++) {
                        newImg.at<Vec3b>(k, l)[m] = newImg.at<Vec3b>(k - 1, l)[m];
                    }
                }
                else if ((k + 1) < newImg.rows) {
                    for (m = 0; m < 3; m++) {
                        newImg.at<Vec3b>(k, l)[m] = newImg.at<Vec3b>(k + 1, l)[m];
                    }
                }
            }
        }
    }
    //interpolate cols
    for (m = 0; m < 3; m++) {
        avg[m] = 0;
    }
    for (l = 0; l < newImg.rows; l++) {
        for (k = 0; k < newImg.cols; k++) {
            if (k & 1) {
                if ((k - 1) >= 0 && (k + 1) < newImg.cols) {
                    for (m = 0; m < 3; m++) {
                        avg[m] = (newImg.at<Vec3b>(l, k - 1)[m] + newImg.at<Vec3b>(l, k + 1)[m]) / 2;
                        newImg.at<Vec3b>(l, k)[m] = avg[m];
                    }
                }
                else if ((k - 1) >= 0) {
                    for (m = 0; m < 3; m++) {
                        newImg.at<Vec3b>(l, k)[m] = newImg.at<Vec3b>(l, k - 1)[m];
                    }
                }
                else if ((k + 1) < newImg.cols) {
                    for (m = 0; m < 3; m++) {
                        newImg.at<Vec3b>(l, k)[m] = newImg.at<Vec3b>(l, k + 1)[m];
                    }
                }
            }
        }
    }
    img2 = newImg;
}

void rotateImg(Mat dest, const Mat src, unsigned char direction)
{
    Mat newImg(Size(src.rows, src.cols), src.type());

    int i, j;
    for (i = 0; i < src.rows; i++) {
        for (j = 0; j < src.cols; j++) {
            if (direction == ROT_CCW) {
                newImg.at<Vec3b>(src.cols - j - 1, i) = src.at<Vec3b>(i, j);
            }
            else if (direction == ROT_CW) {
                newImg.at<Vec3b>(j, src.rows - i - 1) = src.at<Vec3b>(i, j);
            }
        }
    }
    img2 = newImg;
}

void convolution(Mat dest, const Mat src, float kernel[3][3], unsigned char special)
{
    float sum[3];
    Mat aux(src.size(), src.type());

    for (int x = 1; x < src.rows - 1; x++) {
        for (int y = 1; y < src.cols - 1; y++) {
            for (int i = 0; i < 3; i++) {
                sum[i] = (kernel[2][2] * src.at<Vec3b>(x - 1, y - 1)[i]) + //iA
                    (kernel[2][1] * src.at<Vec3b>(x - 1, y)[i]) + //hB
                    (kernel[2][0] * src.at<Vec3b>(x - 1, y + 1)[i]) + //gC
                    (kernel[1][2] * src.at<Vec3b>(x, y - 1)[i]) + //fD
                    (kernel[1][1] * src.at<Vec3b>(x, y)[i]) + //eE
                    (kernel[1][0] * src.at<Vec3b>(x, y + 1)[i]) + //dF
                    (kernel[0][2] * src.at<Vec3b>(x + 1, y - 1)[i]) + //cG
                    (kernel[0][1] * src.at<Vec3b>(x + 1, y)[i]) + //bH
                    (kernel[0][0] * src.at<Vec3b>(x + 1, y + 1)[i]);  //aI

                if (!special) {
                    sum[i] += 127;
                }

                if (sum[i] < 0) {
                    aux.at<Vec3b>(x, y)[i] = 0;
                }
                else if (sum[i] > 255) {
                    aux.at<Vec3b>(x, y)[i] = 255;
                }
                else {
                    aux.at<Vec3b>(x, y)[i] = sum[i];
                }
            }
        }
    }
    img2 = aux;
}

void imgOps(unsigned char operation, unsigned char openFlag)
{
    int posx = 0, posy = 0;
    int screenx = 0;

    screenx = GetSystemMetrics(SM_CXSCREEN);
    posx = screenx / 2;
    posy = 10;

    switch (operation) {
    case IMG_OPEN_FILE:
        FILE *file;
        if (!(file = fopen(filename, "r"))) {
            MessageBox(NULL, "Arquivo nao encontrado!", "ERRO", MB_OK);
            return;
        }
        fclose(file);
        if (openFlag == OPEN_SRC) {
            img = imread(filename);
            Mat imgDest(img.size(), img.type());
            img2 = imgDest;

            cv::imshow("SRC", img);
            cv::imshow("DEST", img2);
            // move windows
            cv::moveWindow("SRC", posx - img.cols - 10, posy);
            cv::moveWindow("DEST", posx + 10, posy);

            histogram(img, "HISTOGRAM SRC");
            histogram(img2, "HISTOGRAM DEST");
            cv::moveWindow("HISTOGRAM SRC", posx + img.cols + 20, posy);
            cv::moveWindow("HISTOGRAM DEST", posx + img.cols + 20, posy + 330);
            break;
        }
        else if (openFlag == OPEN_TARGET) {
            targetImg = imread(filename);
            cv::imshow("TARGET", targetImg);
            histogram(targetImg, "HISTOGRAM TARGET");
            cv::moveWindow("TARGET", posx - img.cols - 10, posy + img.rows + 20);
            cv::moveWindow("HISTOGRAM TARGET", posx - img.cols + targetImg.cols, posy + img.rows + 20);
        }
        break;
    case IMG_COPY: {
        Mat imgDest(img.size(), img.type());
        img2 = imgDest;
        imgcpy(img2, img);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_H_FLIP: {
        Mat temp_img(img2.size(), img2.type());
        hflip(temp_img, img2);
        imgcpy(img2, temp_img);
        cv::imshow("DEST", img2);
        break;
    }
    case IMG_V_FLIP: {
        Mat temp_img(img2.size(), img2.type());
        vflip(temp_img, img2);
        imgcpy(img2, temp_img);
        cv::imshow("DEST", img2);
        break;
    }
    case IMG_H_MIRROR:
        hflip(img2, img2);
        cv::imshow("DEST", img2);
        break;
    case IMG_V_MIRROR:
        vflip(img2, img2);
        cv::imshow("DEST", img2);
        break;
    case IMG_LUMINANCE: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        cv::imshow("DEST", img2);
        break;
    }
    case IMG_QUANTIZATION: {
        Mat imgDest(img.size(), img.type());
        img2 = imgDest;
        wchar_t ws_tones[BUFFER_SIZE] = { 0 };
        char s_tones[BUFFER_SIZE] = { 0 };
        int numTones = 0;
        GetWindowTextW(quantBox, ws_tones, BUFFER_SIZE);
        std::wcstombs(s_tones, ws_tones, BUFFER_SIZE);
        numTones = atoi(s_tones);
        quant(img2, img, numTones);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_BRIGHTNESS: {
        wchar_t ws_brightness[BUFFER_SIZE] = { 0 };
        char s_brightness[BUFFER_SIZE] = { 0 };
        int numBrightness = 0;
        GetWindowTextW(brightnessBox, ws_brightness, BUFFER_SIZE);
        std::wcstombs(s_brightness, ws_brightness, BUFFER_SIZE);
        numBrightness = atoi(s_brightness);
        brightness(img2, img2, numBrightness);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONTRAST: {
        wchar_t ws_contrast[BUFFER_SIZE] = { 0 };
        char s_contrast[BUFFER_SIZE] = { 0 };
        float numContrast = 0;
        GetWindowTextW(contrastBox, ws_contrast, BUFFER_SIZE);
        std::wcstombs(s_contrast, ws_contrast, BUFFER_SIZE);
        numContrast = atof(s_contrast);
        contrast(img2, img2, numContrast);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_NEGATIVE:
        negative(img2, img2);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    case IMG_HISTOGRAM_EQ:
        histogramEq(img2, img2);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    case IMG_HISTOGRAM_MATCH:
        histogramMatch(img2, img, targetImg);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    case IMG_ZOOM_OUT: {
        wchar_t ws_sx[BUFFER_SIZE] = { 0 };
        char s_sx[BUFFER_SIZE] = { 0 };
        int num_sx = 0;
        GetWindowTextW(zoomOut_sx, ws_sx, BUFFER_SIZE);
        std::wcstombs(s_sx, ws_sx, BUFFER_SIZE);
        num_sx = atoi(s_sx);
        wchar_t ws_sy[BUFFER_SIZE] = { 0 };
        char s_sy[BUFFER_SIZE] = { 0 };
        int num_sy = 0;
        GetWindowTextW(zoomOut_sy, ws_sy, BUFFER_SIZE);
        std::wcstombs(s_sy, ws_sy, BUFFER_SIZE);
        num_sy = atoi(s_sy);

        zoomOut(img2, img2, num_sx, num_sy);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_ZOOM_IN:
        zoomIn(img2, img2);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    case IMG_ROTATE_CCW:
        rotateImg(img2, img2, ROT_CCW);
        cv::imshow("DEST", img2);
        break;
    case IMG_ROTATE_CW:
        rotateImg(img2, img2, ROT_CW);
        cv::imshow("DEST", img2);
        break;
    case IMG_CONV_GAUSS: {
        float kernel[3][3] = { { 0.0625, 0.125, 0.0625 },
        { 0.125, 0.25, 0.125 },
        { 0.0625, 0.125, 0.0625 } };
        convolution(img2, img2, kernel, 1);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONV_LAPLACE: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        float kernel[3][3] = { { 0, -1, 0 },
        { -1, 4, -1 },
        { 0, -1, 0 } };
        convolution(img2, img2, kernel, 1);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONV_GEN_HIGH: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        float kernel[3][3] = { { -1, -1, -1 },
        { -1, 8, -1 },
        { -1, -1, -1 } };
        convolution(img2, img2, kernel, 1);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONV_PRE_HX: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        float kernel[3][3] = { { -1, 0, 1 },
        { -1, 0, 1 },
        { -1, 0, 1 } };
        convolution(img2, img2, kernel, 0);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONV_PRE_HY: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        float kernel[3][3] = { { -1, -1, -1 },
        { 0, 0, 0 },
        { 1, 1, 1 } };
        convolution(img2, img2, kernel, 0);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONV_SOB_HX: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        float kernel[3][3] = { { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 } };
        convolution(img2, img2, kernel, 0);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONV_SOB_HY: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        float kernel[3][3] = { { -1, -2, -1 },
        { 0, 0, 0 },
        { 1, 2, 1 } };
        convolution(img2, img2, kernel, 0);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    case IMG_CONV_CUSTOM: {
        Mat temp_img(img2.size(), img2.type());
        grayscale(temp_img, img2);
        imgcpy(img2, temp_img);
        float kernel[3][3] = { 0 };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                wchar_t ws_ker[BUFFER_SIZE] = { 0 };
                char s_ker[BUFFER_SIZE] = { 0 };
                GetWindowTextW(conv[i][j], ws_ker, BUFFER_SIZE);
                std::wcstombs(s_ker, ws_ker, BUFFER_SIZE);
                kernel[i][j] = atof(s_ker);
            }
        }

        convolution(img2, img2, kernel, 0);
        cv::imshow("DEST", img2);
        histogram(img2, "HISTOGRAM DEST");
        break;
    }
    }
    cv::waitKey(20);
}

int openFile(HWND hWnd, unsigned char openFlag)
{
    OPENFILENAME ofn;

    wchar_t LfileName[FILENAME_MAX * 2] = { 0 }; // wide char

    ZeroMemory(&ofn, sizeof(OPENFILENAME));

    ofn.lStructSize = sizeof(OPENFILENAME);
    ofn.hwndOwner = hWnd;
    ofn.lpstrFile = LfileName;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = FILENAME_MAX * 2;
    ofn.lpstrFilter = L"Image Files (.jpg, .png, .bmp)\0*.JPG;*.PNG*;.BMP\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_FILEMUSTEXIST;

    GetOpenFileName(&ofn);
    std::wcstombs(filename, LfileName, wcslen(LfileName) + 1);
    if (!filename[0]) {
        return 0;
    }

    imgOps(IMG_OPEN_FILE, openFlag);
    return 1;
}

int saveFile(HWND hWnd)
{
    OPENFILENAME sfn;

    char LfileName[FILENAME_MAX * 2] = { 0 }; // wide char

    ZeroMemory(&sfn, sizeof(OPENFILENAME));

    sfn.lStructSize = sizeof(OPENFILENAME);
    sfn.hwndOwner = hWnd;
    sfn.lpstrFile = LfileName;
    sfn.lpstrFile[0] = '\0';
    sfn.nMaxFile = FILENAME_MAX * 2;
    sfn.lpstrFilter = "JPEG Image\0*.JPG\0PNG Image\0*.PNG\0BMP Image\0*.BMP\0";
    sfn.nFilterIndex = 1;
    sfn.lpstrDefExt = "JPG\0PNG\0BMP\0";
    sfn.Flags = OFN_OVERWRITEPROMPT;

    GetSaveFileName(&sfn);
   // std::wcstombs(filename, LfileName, wcslen(LfileName) + 1);
    strcpy(filename, LfileName);
    if (!filename[0]) {
        return 0;
    }
    imwrite(filename, img2);
    return 1;
}

LRESULT CALLBACK WindowProcedure(HWND hWnd, UINT msg, WPARAM wp, LPARAM lp)
{
    static unsigned char ini = 0;
    static unsigned char t_ini = 0;
    int val;
    switch (msg) {
    case WM_COMMAND:
        switch (wp) {
        case PROC_INFO_MENU:
            MessageBox(NULL, L"", L"Information", MB_OK);
            break;
        case PROC_EXIT_MENU:
            val = MessageBoxW(NULL, L"Encerrar a Execucao?", L"", MB_YESNO | MB_ICONEXCLAMATION);
            if (val == IDYES) {
                DestroyWindow(hWnd);
            }
            break;
        case PROC_OPEN_FILE:
            if (openFile(hWnd, OPEN_SRC)) {
                ini = 1;
            }
            break;
        case PROC_COPY_IMG:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_COPY, NULL);
            }
            break;
        case PROC_H_FLIP:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_H_FLIP, NULL);
            }
            break;
        case PROC_V_FLIP:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_V_FLIP, NULL);
            }
            break;
        case PROC_H_MIRROR:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_H_MIRROR, NULL);
            }
            break;
        case PROC_V_MIRROR:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_V_MIRROR, NULL);
            }
            break;
        case PROC_LUMINANCE:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_LUMINANCE, NULL);
            }
            break;
        case PROC_QUANTIZATION:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_QUANTIZATION, NULL);
            }
            break;
        case PROC_SAVE_FILE:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                if (saveFile(hWnd)) {
                    MessageBox(NULL, L"Imagem destino salva com sucesso!", L"", MB_OK);
                }
            }
            break;
        case PROC_BRIGHTNESS:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_BRIGHTNESS, NULL);
            }
            break;
        case PROC_CONTRAST:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONTRAST, NULL);
            }
            break;
        case PROC_NEGATIVE:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_NEGATIVE, NULL);
            }
            break;
        case PROC_HISTOGRAM_EQ:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_HISTOGRAM_EQ, NULL);
            }
            break;
        case PROC_OPEN_TARGET:
            if (openFile(hWnd, OPEN_TARGET)) {
                t_ini = 1;
            }
            break;
        case PROC_HISTOGRAM_MATCH:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else if (!t_ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem alvo!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_HISTOGRAM_MATCH, NULL);
            }
            break;
        case PROC_ZOOM_OUT:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_ZOOM_OUT, NULL);
            }
            break;
        case PROC_ZOOM_IN:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_ZOOM_IN, NULL);
            }
            break;
        case PROC_ROTATE_CCW:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_ROTATE_CCW, NULL);
            }
            break;
        case PROC_ROTATE_CW:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_ROTATE_CW, NULL);
            }
            break;
        case PROC_CONV_GAUSS:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_GAUSS, NULL);
            }
            break;
        case PROC_CONV_LAPLACE:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_LAPLACE, NULL);
            }
            break;
        case PROC_CONV_GEN_HIGH:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_GEN_HIGH, NULL);
            }
            break;
        case PROC_CONV_PRE_HX:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_PRE_HX, NULL);
            }
            break;
        case PROC_CONV_PRE_HY:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_PRE_HY, NULL);
            }
            break;
        case PROC_CONV_SOB_HX:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_SOB_HX, NULL);
            }
            break;
        case PROC_CONV_SOB_HY:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_SOB_HY, NULL);
            }
            break;
        case PROC_CONV_CUSTOM:
            if (!ini) {
                MessageBox(NULL, L"Primeiro carregue uma imagem de origem!", L"ERRO", MB_OK);
            }
            else {
                imgOps(IMG_CONV_CUSTOM, NULL);
            }
            break;
        }

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProcW(hWnd, msg, wp, lp);
    }
}

void AddControls(HWND hWnd)
{
    CreateWindowW(
        L"BUTTON",
        L"OPEN SRC IMAGE",      // Button text 
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20,         // x position 
        20,         // y position 
        200,        // Button width
        25,        // Button height
        hWnd,     // Parent window
        (HMENU)PROC_OPEN_FILE, // procedure
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"COPY",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 100,
        90, 25,
        hWnd,
        (HMENU)PROC_COPY_IMG,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"H-FLIP",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 130,
        90, 25,
        hWnd,
        (HMENU)PROC_H_FLIP,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"V-FLIP",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        130, 130,
        90, 25,
        hWnd,
        (HMENU)PROC_V_FLIP,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"H-MIRROR",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 160,
        90, 25,
        hWnd,
        (HMENU)PROC_H_MIRROR,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"V-MIRROR",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        130, 160,
        90, 25,
        hWnd,
        (HMENU)PROC_V_MIRROR,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"LUMINANCE",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 190,
        200, 25,
        hWnd,
        (HMENU)PROC_LUMINANCE,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"QUANTIZE",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        130, 100,
        90, 25,
        hWnd,
        (HMENU)PROC_QUANTIZATION,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    quantBox = CreateWindowW(
        L"EDIT", L"16",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | ES_NUMBER | WS_BORDER | ES_RIGHT,
        130, 70,
        30, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"STATIC", L"TONES",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD,
        165, 73,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"SAVE DEST IMAGE",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 220,
        200, 25,
        hWnd,
        (HMENU)PROC_SAVE_FILE,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"OPEN TARGET IMAGE",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 20,
        200, 25,
        hWnd,
        (HMENU)PROC_OPEN_TARGET,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    brightnessBox = CreateWindowW(
        L"EDIT", L"32",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_RIGHT,
        260, 70,
        40, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"BRIGHTNESS",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        310, 70,
        150, 25,
        hWnd,
        (HMENU)PROC_BRIGHTNESS,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    contrastBox = CreateWindowW(
        L"EDIT", L"0.8",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_RIGHT,
        260, 100,
        70, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"CONTRAST",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        340, 100,
        120, 25,
        hWnd,
        (HMENU)PROC_CONTRAST,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"NEGATIVE",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 130,
        200, 25,
        hWnd,
        (HMENU)PROC_NEGATIVE,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"HISTOGRAM EQ",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 160,
        200, 25,
        hWnd,
        (HMENU)PROC_HISTOGRAM_EQ,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"HISTOGRAM MATCH",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 190,
        200, 25,
        hWnd,
        (HMENU)PROC_HISTOGRAM_MATCH,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    zoomOut_sy = CreateWindowW(
        L"EDIT", L"2",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        260, 220,
        20, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"STATIC", L"x",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD,
        285, 223,
        20, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    zoomOut_sx = CreateWindowW(
        L"EDIT", L"2",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        297, 220,
        20, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"ZOOM OUT",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        325, 220,
        135, 25,
        hWnd,
        (HMENU)PROC_ZOOM_OUT,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"ZOOM IN 2x",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 250,
        200, 25,
        hWnd,
        (HMENU)PROC_ZOOM_IN,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"ROTATE CCW",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 280,
        200, 25,
        hWnd,
        (HMENU)PROC_ROTATE_CCW,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"ROTATE CW",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 310,
        200, 25,
        hWnd,
        (HMENU)PROC_ROTATE_CW,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"STATIC", L"CONVOLUTION",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD,
        20, 325,
        110, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"GAUSSIAN",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 350,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_GAUSS,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"LAPLACIAN",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 380,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_LAPLACE,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"GENERIC HIGH-PASS",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 410,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_GEN_HIGH,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"PREWITT HX",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 440,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_PRE_HX,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"PREWITT HY",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 470,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_PRE_HY,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"SOBEL HX",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 500,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_SOB_HX,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"SOBEL HY",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 530,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_SOB_HY,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindowW(
        L"BUTTON", L"CUSTOM KERNEL",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        260, 440,
        200, 25,
        hWnd,
        (HMENU)PROC_CONV_CUSTOM,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[0][0] = CreateWindowW(
        L"EDIT", L"-0.5",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        260, 470,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[0][1] = CreateWindowW(
        L"EDIT", L"0.5",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        330, 470,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[0][2] = CreateWindowW(
        L"EDIT", L"-0.5",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        400, 470,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[1][0] = CreateWindowW(
        L"EDIT", L"1",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        260, 500,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[1][1] = CreateWindowW(
        L"EDIT", L"-2",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        330, 500,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[1][2] = CreateWindowW(
        L"EDIT", L"1",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        400, 500,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[2][0] = CreateWindowW(
        L"EDIT", L"-0.5",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        260, 530,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[2][1] = CreateWindowW(
        L"EDIT", L"0.5",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        330, 530,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    conv[2][2] = CreateWindowW(
        L"EDIT", L"-0.5",
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
        400, 530,
        60, 25,
        hWnd,
        NULL,
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

}

void AddMenus(HWND hWnd, HMENU hMenu)
{
    hMenu = CreateMenu();
    HMENU hFileMenu = CreateMenu();

    AppendMenu(hFileMenu, MF_STRING, PROC_EXIT_MENU, L"Exit");

    AppendMenu(hMenu, MF_POPUP, (UINT_PTR)hFileMenu, L"File");
    AppendMenu(hMenu, MF_STRING, NULL, L"Info");

    SetMenu(hWnd, hMenu);
}

int main(void)
{
    // hide cmd
    ShowWindow(GetConsoleWindow(), SW_HIDE);

    WNDCLASSW wc = { 0 }; // define window class
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProcedure;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = (HINSTANCE)GetModuleHandle(NULL);
    wc.hIcon = NULL;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
    wc.lpszMenuName = NULL;
    wc.lpszClassName = L"WindowClass";

    // register class
    if (!RegisterClassW(&wc)) {
        return 0;
    }

    // create main window
    HWND hWnd = CreateWindowW(L"WindowClass", L"Image Editor",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        100, 100,
        500, 630,
        NULL, NULL, NULL, NULL);
    if (!hWnd) {
        ShowWindow(GetConsoleWindow(), SW_SHOW);
        return 0;
    }

    // create menu and controls
    HMENU hMenu = { 0 };
    AddMenus(hWnd, hMenu);
    AddControls(hWnd);

    MSG msg = { 0 };

    UpdateWindow(hWnd);
    while (GetMessage(&msg, hWnd, 0, 0) != -1) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    ShowWindow(GetConsoleWindow(), SW_SHOW);
    return 0;
}