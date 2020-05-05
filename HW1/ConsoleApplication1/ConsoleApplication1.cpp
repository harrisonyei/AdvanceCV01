

#include <iostream>
#include <string>

#include "hdriSystem.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
    if (argc < 8) {
        cout << "Incorrect arguments! : [ Directory ] [ File Type(jpg, nef, png) ] [shutter offset] [shutter multiplier] [ghost removal (0/1)] [ghost removal threshold] [Output filename]" << std::endl;
        return -1;
    }

    srand(time(NULL));

    string dir      = argv[1];
    string subtitle = argv[2];
    float shutterOffset = atof(argv[3]);
    float shutterMultiplier = atof(argv[4]);
    bool ghostRemoval = atoi(argv[5]);
    float ghostRemovalThreshold = atof(argv[6]);

    string exportPath = argv[7];

    HDRI* hdriProcessor = new HDRI(dir, subtitle, shutterOffset, shutterMultiplier);

    Mat hdri = hdriProcessor->GetHDRI(ghostRemoval, ghostRemovalThreshold);
    Mat image = hdriProcessor->ToneMapping(hdri, 0.06, 0.2, 5, 0.56, 50);

    imwrite(exportPath, image);

    imshow("Display window", image);               // Show our image inside it.
    waitKey(0);
    
    return 0;
}

