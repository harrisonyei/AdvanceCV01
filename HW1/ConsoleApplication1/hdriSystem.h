#pragma once

#include "ImageProcess.h"
#include <iostream>
#include <string>
#include <vector>

class HDRI {
	std::string dir; // �Ϥ���Ƨ����|
	std::string subtitle; // �Ϥ���Ƨ����|

	float shutter_offset = 1; // �Ӥ������֪��ɶ�
	float shutter_multiplier = 0.5; // �Ӥ������֪��ɶ��t�Z

	std::vector<cv::Mat> images; // Ū���i�Ӫ��Ϥ�
	std::vector<cv::Vec2i> image_offsets; // Ū���i�Ӫ��Ϥ�


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