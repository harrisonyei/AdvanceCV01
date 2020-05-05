#include "ImageProcess.h"
#include <iostream>

#define MIN_IMG_SIZE 64
#define MAX_IMG_STEP 4

using namespace cv;
using namespace std;

// align img1 to img0
Vec2i Align(Mat& img0, Mat& img1)
{
	if (img0.size != img1.size) {
		cout << " IMAGE SIZE NOT EQUAL" << endl;
		return 0;
	}

	int pyrHeight = min(log2f(img0.cols / MIN_IMG_SIZE), log2f(img0.rows / MIN_IMG_SIZE));

	Mat pyrImg0 = MTB(img0, 7);
	Mat pyrImg1 = MTB(img1, 7);

	vector<Mat> pyrImgs0;
	vector<Mat> pyrImgs1;

	// build pyramids
	pyrImgs0.push_back(pyrImg0.clone());
	pyrImgs1.push_back(pyrImg1.clone());
	for (int i = 1; i < pyrHeight; i++) {

		// gaussian filter down
		pyrDown(pyrImg0, pyrImg0, Size(pyrImg0.cols / 2, pyrImg0.rows / 2));
		pyrDown(pyrImg1, pyrImg1, Size(pyrImg1.cols / 2, pyrImg1.rows / 2));

		pyrImgs0.push_back(pyrImg0.clone());
		pyrImgs1.push_back(pyrImg1.clone());
	}

	// find offset
	Vec2i offset(0, 0);
	for (int i = pyrHeight-1; i >= 0; i--) {
		offset *= 2; 

		float minDiff = 9999;
		int minRow = 0;
		int minCol = 0;

		for (int row = -MAX_IMG_STEP; row <= MAX_IMG_STEP; row++) {
			for (int col = -MAX_IMG_STEP; col <= MAX_IMG_STEP; col++) {

				float diff = Diff(pyrImgs0[i], pyrImgs1[i], offset + Vec2i(col, row));

				if (diff < minDiff) {
					minRow = row;
					minCol = col;

					minDiff = diff;
				}

			}
		}
		offset = offset + Vec2i(minCol, minRow);
	}

	Mat originImg = img1.clone();
	for (int col = 0; col < img1.cols; col++) {
		for (int row = 0; row < img1.rows; row++) {
			if ((col + offset[0]) >= 0 && (col + offset[0]) < img1.cols &&
				(row + offset[1]) >= 0 && (row + offset[1]) < img1.rows) {
				img1.at<Vec3b>(row, col) = originImg.at<Vec3b>(row + offset[1], col + offset[0]);
			}
			else {
				img1.at<Vec3b>(row, col) = 0;
			}
		}
	}

	return offset;
}

float Diff(cv::Mat& img0, cv::Mat& img1, Vec2i offset)
{
	float diff = 0;
	int sampleCount = 1;
	for (int col = 0; col < img0.cols; col++) {
		for (int row = 0; row < img0.rows; row++) {

			if (col + offset[0] < 0 || col + offset[0] >= img1.cols||
				row + offset[1] < 0 || row + offset[1] >= img1.rows) {
				diff += 0;
			}
			else {
				uchar p0 = img0.at<uchar>(row, col);
				uchar p1 = img1.at<uchar>(row + offset[1], col + offset[0]);

				if (p0 == 100 || p1 == 100) {
					continue;
				}

				diff += abs(p0 - p1);
				sampleCount += 1;
			}
		}
	}
	return diff / sampleCount;
}

cv::Mat MTB(cv::Mat& img, int margin)
{
	Mat dst;
	cvtColor(img, dst, cv::COLOR_BGR2GRAY);
	int med = Median(dst);

	for (int col = 0; col < dst.cols; col++) {
		for (int row = 0; row < dst.rows; row++) {
			uchar& p = dst.at<uchar>(row, col);
			int diff = p - med;
			if (abs(diff) < margin) {
				p = 100;
			}
			else if (diff > 0) {
				p = 255;

			}
			else {
				p = 0;

			}
		}
	}

	return dst;
}

int Median(cv::Mat& img)
{
	unsigned int counter[256] = {0};
	for (int col = 0; col < img.cols; col++) {
		for (int row = 0; row < img.rows; row++) {
			uchar p = img.at<uchar>(row, col);
			counter[p] += 1;
		}
	}

	int medianIdx = (img.rows * img.cols) / 2;
	int idx = 0;
	for (int i = 0; i < 256; i++) {
		idx += counter[i];
		if (idx >= medianIdx) {
			return i;
		}
	}

	return 0;
}

void HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV) {
	float fC = fV * fS; // Chroma
	float fHPrime = fmod(fH / 60.0, 6);
	float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
	float fM = fV - fC;

	if (0 <= fHPrime && fHPrime < 1) {
		fR = fC;
		fG = fX;
		fB = 0;
	}
	else if (1 <= fHPrime && fHPrime < 2) {
		fR = fX;
		fG = fC;
		fB = 0;
	}
	else if (2 <= fHPrime && fHPrime < 3) {
		fR = 0;
		fG = fC;
		fB = fX;
	}
	else if (3 <= fHPrime && fHPrime < 4) {
		fR = 0;
		fG = fX;
		fB = fC;
	}
	else if (4 <= fHPrime && fHPrime < 5) {
		fR = fX;
		fG = 0;
		fB = fC;
	}
	else if (5 <= fHPrime && fHPrime < 6) {
		fR = fC;
		fG = 0;
		fB = fX;
	}
	else {
		fR = 0;
		fG = 0;
		fB = 0;
	}

	fR += fM;
	fG += fM;
	fB += fM;
}

