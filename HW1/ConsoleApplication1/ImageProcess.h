#pragma once

#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Vec2i Align(cv::Mat& img0, cv::Mat& img1);
float Diff(cv::Mat& img0, cv::Mat& img1, cv::Vec2i offset);
cv::Mat MTB(cv::Mat& img, int margin=0);
int Median(cv::Mat& img);

void HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV);

