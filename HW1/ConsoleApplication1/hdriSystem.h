#pragma once

#include "ImageProcess.h"
#include <iostream>
#include <string>
#include <vector>

class HDRI {
	std::string dir; // 圖片資料夾路徑
	std::string subtitle; // 圖片資料夾路徑

	float shutter_offset = 1; // 照片之間快門時間
	float shutter_multiplier = 0.5; // 照片之間快門時間差距

	std::vector<cv::Mat> images; // 讀取進來的圖片
	std::vector<cv::Vec2i> image_offsets; // 讀取進來的圖片


public: 

	HDRI(std::string dir, std::string subtitle, float shutter_offset=1, float shutter_mul=0.5f);
	~HDRI();

	void ReadConfig(std::string filename);

	cv::Mat GetHDRI(bool ghostRemove=false, float ghost_threshold = 0.2f);

	cv::Mat ToneMapping(cv::Mat& hdri,float epsilon, float alpha=0.2f, float phi = 2, float a = 1, int maxIteration=50);

	cv::Mat RadianceMap(cv::Mat& hdri);

	cv::Mat GetImage(int idx);

	void ShowInputs();


};