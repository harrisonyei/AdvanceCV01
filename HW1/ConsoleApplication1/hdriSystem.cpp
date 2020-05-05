#include "hdriSystem.h"
#include "fileIO.h"

#include <algorithm>

using namespace cv;
using namespace std;

#define GRAY(_R_, _G_, _B_) (((_R_) * 0.27) + ((_G_) * 0.67) + ((_B_) * 0.06))

#define LOOP_MAT(__mat__) for(int row=0;row<(__mat__).rows;row++)\
                                for(int col=0;col<(__mat__).cols;col++)

HDRI::HDRI(string dir, string subtitle, float shutter_offset, float shutter_mul)
{
    this->dir = dir;
    this->subtitle = subtitle;
    this->shutter_offset = shutter_offset;
    this->shutter_multiplier = shutter_mul;

    vector<string> filenames = get_filenames_in(dir, subtitle);

    this->images.clear();
    this->image_offsets.clear();

    for (string& filename : filenames) {
        Mat image;
        
        image = imread(filename, IMREAD_COLOR);
        if (!image.data)// Check for invalid input
        {
            cout << "Could not open or find the image : " << filename << std::endl;
            continue;
        }

        this->images.push_back(image);
        this->image_offsets.push_back(Vec2i(0, 0));
    }

}

HDRI::~HDRI()
{
}

void HDRI::ReadConfig(std::string filename)
{
}

Mat HDRI::GetHDRI(bool ghostRemove, float ghost_threshold)
{
    cout << endl << "Create HDRI Image Start." << endl;

    if (this->images.size() < 2) {
        cout << "Fail, Images must more than two." << endl;
        return Mat();
    }


    // record sample boundary
    int leftX = 0, bottomY = 0, rightX = 0, topY = 0;

    cout << "Align" << endl;
    for (int i = 1; i < this->images.size(); i++) {
        Vec2i offset = Align(this->images[0], this->images[i]);
        cout << "algin :  " << i << " / "<< this->images.size()-1 <<"           \r";

        leftX = max(leftX, -offset[0]);
        bottomY = max(bottomY, -offset[1]);

        rightX = max(rightX, offset[0]);
        topY = max(topY, offset[1]);
    }
    cout << endl << "Align Done" << endl;

    int cols = this->images[0].cols;
    int rows = this->images[0].rows;

    int colRange = cols - leftX - rightX;
    int rowRange = rows - bottomY - topY;

    cout << "Sampling" << endl;
    int sampleN = (255 / (this->images.size() - 1)) * 2;
    vector<Vec2i> sample_offsets;
    for (int i = 0; i < sampleN; i++) {
        int col = (rand() % colRange) + leftX;
        int row = (rand() % rowRange) + bottomY;
        sample_offsets.push_back(Vec2i(col, row));
    }
    cout << "Sampling Done" << endl;


    int n = 256;
    Mat A[3]; 
    Mat b[3];
    for (int i = 0; i < 3; i++) {
        A[i] = Mat::zeros(sampleN * this->images.size() + n + 1, n + sampleN, CV_32F);
        b[i] = Mat::zeros(A[i].rows, 1, CV_32F);
    }

    float log_shutter_off = log(shutter_offset);
    float log_shutter_mul = log(shutter_multiplier);

    // weight function
    auto w = [](int z) {
        float weight = (128 - abs(z - 128)) / 128.0f;
        return max(weight, 0.001f);
    };

    cout << "Initialize matrices" << endl;
    int k = 0; // Include the data - fitting equations
    for (int i = 0; i < sampleN; i++) {
        Vec2i& offset = sample_offsets[i];

        for (int j = 0; j < this->images.size(); j++) {

            Vec3b& color = this->images[j].at<Vec3b>(offset[1], offset[0]);

            for (int c = 0; c < 3; c++) {
                float wij = w(color[c]);

                A[c].at<float>(k, color[c]) = wij;
                A[c].at<float>(k, n + i)    = -wij; 
                b[c].at<float>(k, 0)        = wij * (log_shutter_off + log_shutter_mul * j);
            }

            k = k + 1;
        }
    }

    for (int c = 0; c < 3; c++) {
        A[c].at<float>(k, 128) = 1;// Fix the curve by setting its middle value to 0
    }

    k = k + 1;

    // smoothness constant
    const float l = 6.5f;

    for (int i = 0; i < n - 2; i++) {
        //Include the smoothness equations
        for (int c = 0; c < 3; c++) {
            A[c].at<float>(k, i) = l * w(i);
            A[c].at<float>(k, i + 1) = -2 * l * w(i);
            A[c].at<float>(k, i + 2) = l * w(i);
        }
        k = k + 1;
    }
    cout << "Initialize Done" << endl;

    cout << "Solve Ax=b" << endl;
    Mat x[3];
    Mat g[3];
    for (int c = 0; c < 3; c++) {
        cv::solve(A[c], b[c], x[c], DECOMP_SVD);// Solve the system using SVD
        //Mat lE = x.rowRange(n, x.rows);
        g[c] = x[c].rowRange(0, n);

        for (int i = 15; i > 0; i--) {
            if (g[c].at<float>(i) < g[c].at<float>(i-1)) {
                g[c].at<float>(i - 1) = g[c].at<float>(i);
            }
        }

        for (int i = n-15; i < n; i++) {
            if (g[c].at<float>(i-1) > g[c].at<float>(i)) {
                g[c].at<float>(i) = g[c].at<float>(i-1);
            }
        }
    }
    cout << "Solve Done" << endl;

    cout << endl << "Generating result image" << endl;
    Mat hdri(this->images[0].rows, this->images[0].cols, CV_32FC3);
    vector<Mat> ghostMask;
    // pick reference image
    if (ghostRemove) {
        int referenceImageIdx = -1;

        auto f = [](float E, Mat& g) {

            int lE = log(E);

            int select_i = 0;
            for (int i = 0; i < 255; i++) {
                if (g.at<float>(i) <= lE && g.at<float>(i + 1) >= lE) {
                    select_i = i;
                    break;
                }
            }

            float e0 = exp(g.at<float>(select_i));
            float e1 = exp(g.at<float>(select_i+1));

            float p = (E - e1) / (e0 - e1);
            return select_i * p + (select_i + 1) * (1 - p);
        };

        cout << endl << "[Ghost Removal] Picking Reference Image" << endl;
        float minLoss = 9999;
        for (int i = 0; i < this->images.size(); i++) {

            float loss = 0;
            int N = 0;
            Mat& image = this->images[i];
            LOOP_MAT(image) {
                Vec3b& color = image.at<Vec3b>(row, col);

                float g = GRAY(color[2], color[1], color[0]);
                if (g > 10 && g < 250) {
                    loss += (w(color[0]) + w(color[1]) + w(color[2]));
                    N += 1;
                }
            }
            loss /= N;

            if (loss < minLoss) {
                minLoss = loss;
                referenceImageIdx = i;
            }
        }

        cout << endl << "[Ghost Removal] Remove Invalid Pixel" << endl;

        for (int i = 0; i < this->images.size(); i++) {

            bool first_half = (i <= (this->images.size() / 2));

            int idx1 = first_half ? (i + 1) : (i - 1);

            Mat image0 = this->images[i];
            Mat image1 = this->images[idx1];

            float shutter_ratio = pow(this->shutter_multiplier, i - idx1);

            Mat mask(image0.rows, image0.cols, CV_8U);

            LOOP_MAT(image0) {

                uchar isGhost = 0;
                if (i != referenceImageIdx) {

                    Vec3b& color0 = image0.at<Vec3b>(row, col);
                    Vec3b& color1 = image1.at<Vec3b>(row, col);

                    float zk[3];
                    for (int c = 0; c < 3; c++) {
                        float Ej = exp(g[c].at<float>(color1[c]) - (log_shutter_off + log_shutter_mul * idx1));
                        zk[c] = f(Ej * shutter_ratio, g[c]);
                    }

                    float g_zk = GRAY(zk[2], zk[1], zk[0]);
                    float g_color = GRAY(color0[2], color0[1], color0[0]);

                    if (g_zk > 10 && g_zk < 240 || g_color > 10 && g_color < 240) {
                        if (abs(g_color - g_zk) > ghost_threshold) {
                            // this pixel is ghost
                            isGhost = 1;
                        }
                    }
                }

                mask.at<uchar>(row, col) = isGhost;

            }
        
            ghostMask.push_back(mask);
        }
    
        cout << "[Ghost Removal] Done" << endl;
    }

    LOOP_MAT(hdri) {
        float totalWeight[3] = { 0 };
        float totalLE[3] = { 0 };

        for (int j = 0; j < this->images.size(); j++) {
            Vec3b& color = this->images[j].at<Vec3b>(row, col);

            if (!ghostRemove || ghostMask[j].at<uchar>(row, col) == 0) {
                for (int c = 0; c < 3; c++) {
                    float wij = w(color[c]);
                    totalWeight[c] += wij;
                    totalLE[c] += (wij * (g[c].at<float>(color[c]) - (log_shutter_off + log_shutter_mul * j)));
                }
            }
            
        }

        hdri.at<Vec3f>(row, col) = Vec3f(exp(totalLE[0] / totalWeight[0]),
                                            exp(totalLE[1] / totalWeight[1]),
                                                exp(totalLE[2] / totalWeight[2]));
    }
    cout << "HDRI Done" << endl;

    /*Mat tmp(hdri.rows, hdri.cols, CV_8U);
    LOOP_MAT(tmp) {
        Vec3f& c = hdri.at<Vec3f>(row, col);

        tmp.at<uchar>(row, col) = min(255, (int)(GRAY(c[2], c[1], c[0]) * 10));
    }*/

    return hdri;
}

// alpha : global luminance factor
// Lwhite: white color threshold
cv::Mat HDRI::ToneMapping(cv::Mat& hdri,float epsilon, float alpha, float phi, float a, int maxIteration)
{
    cout << endl << "Tone Mapping" << endl;
    const float delta = 0.00001f;

    cout << endl << "Global" << endl;

    int N = (hdri.rows * hdri.cols);
    float log_total_Lw = 0;
    LOOP_MAT(hdri) {
        Vec3f LwV = hdri.at<Vec3f>(row, col);
        log_total_Lw += log(delta + GRAY(LwV[2], LwV[1], LwV[0]));
    }
    float avg_Lw = exp(log_total_Lw / N);

    Mat Lm = hdri.clone() * (alpha / avg_Lw); // median

    Mat Lblur = Lm.clone(); // blur
    Mat Ls    = Lm.clone(); // blur max s

    cout << endl << "Local" << endl;
    int iteration = maxIteration;
    int power_phi = pow(2, phi);
    GaussianBlur(Lm, Lblur, Size(3, 3), 1, 1, BorderTypes::BORDER_REFLECT);
    for (int s = 1; iteration > 0; s += 1, iteration -=1) {
        Mat Lblur2;
        GaussianBlur(Lm, Lblur2, Size(s * 2 + 3, s * 2 + 3), 1, 1, BorderTypes::BORDER_REFLECT);
        int alterPixel = 0;
        LOOP_MAT(Lblur) {
            Vec3f& c0 = Lblur.at<Vec3f>(row, col);
            Vec3f& c1 = Lblur2.at<Vec3f>(row, col);

            float lb0 = GRAY(c0[2], c0[1], c0[0]);
            float lb1 = GRAY(c1[2], c1[1], c1[0]);

            float Vs = (lb0 - lb1) / (power_phi * a / (s * s) + lb0);
            if (abs(Vs) < epsilon && abs(Vs) > delta) {
                Ls.at<Vec3f>(row, col) = c0;
                alterPixel += 1;
            }

        }

        if (alterPixel <= 0) {
            break;
        }

        Lblur2.copyTo(Lblur);
    }

    cout << "Iter : " << maxIteration - iteration << endl;

    cout << endl << "to LDR" << endl;
    Mat result(Lm.rows, Lm.cols, CV_8UC3);

    LOOP_MAT(Lm) {
        Vec3f& LmV = Lm.at<Vec3f>(row, col); // Lm value
        Vec3f& LsV = Ls.at<Vec3f>(row, col); // Ls value

        Vec3b& color = result.at<Vec3b>(row, col);

        for (int c = 0; c < 3; c++) {
            float Ld = (LmV[c] / (1 + LsV[c]));
            int v = (Ld * 255);
            if (v > 255) {
                v = 255;
            }
            else if (v < 0) {
                v = 0;
            }
            color[c] = v;
        }
    }
    cout << endl << "Done" << endl;

    return result;
}

cv::Mat HDRI::RadianceMap(cv::Mat& hdri)
{
    cv::Mat radiance(hdri.rows, hdri.cols, CV_32F);

    float maxR = -99999, minR = 99999;

    LOOP_MAT(hdri) {
        Vec3f& v = hdri.at<Vec3f>(row, col);

        float r = log(GRAY(v[2], v[1], v[0]));
        radiance.at<float>(row, col) = r;

        maxR = std::max(maxR, r);
        minR = std::min(minR, r);
    }

    cv::Mat result(hdri.rows, hdri.cols, CV_8UC3);
    LOOP_MAT(radiance) {
        float norm_r = (radiance.at<float>(row, col) - minR) / (maxR - minR);
        float r, g, b;
        float h = (1 - norm_r) * 270, s = 1, v = 1;

        HSVtoRGB(r, g, b, h, s, v);

        result.at<Vec3b>(row, col) = Vec3b(b * 255, g * 255, r * 255);
    }

    return result;
}

cv::Mat HDRI::GetImage(int idx)
{
    if (idx >= images.size()) {
        cout << " Index Out of Range\n";
        return cv::Mat();
    }

    return images[idx];
}

void HDRI::ShowInputs()
{
    namedWindow("Display window", WINDOW_AUTOSIZE);
    for (int i = 0; i < this->images.size(); i++) {
        cv::Mat image = this->images[i];
        imshow("Display window " + to_string(i), image);                   // Show our image inside it.
    }

    waitKey(0);
}
