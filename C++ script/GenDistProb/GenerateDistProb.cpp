#include <string.h>
#include <iostream> 
#include <time.h> 
#include <math.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "zxhImageGipl.h" 
#include "zxhImageModelingLinear.h"

using namespace std;

int main(int argc, char* argv[])
{
	string mainfold, Datafold, TargetImageName, ScarLabelName, LABloodPoolGaussianBlurName, \
		WallLabelName, WallScarLabelName;

	mainfold = argv[1];
	string  casename = argv[2];

	//mainfold = "C:\\LeiLi\\2020_MICCAI\\LA2020\\Data_60\\train_data";
	//string  casename = "patient_1";

	string PathName = mainfold + "\\" + casename;
	TargetImageName = PathName + "\\enhanced.nii.gz";
	LABloodPoolGaussianBlurName = PathName + "\\LA_label_GauiisanBlur_M.nii.gz";
	WallLabelName = PathName + "\\LAwall_gd_new.nii.gz";
	ScarLabelName = PathName + "\\LA_MeshWallLabel_M.nii.gz";
	WallScarLabelName = PathName + "\\scarSegImgM_wall.nii.gz";

	zxhImageData SourceImage, LABloodPoolGaussianBlur, WallLabel, ScarLabel, WallScarLabel;

	zxh::OpenImageSafe(&SourceImage, TargetImageName);
	zxh::OpenImageSafe(&LABloodPoolGaussianBlur, LABloodPoolGaussianBlurName);
	zxh::OpenImageSafe(&WallLabel, WallLabelName);
	zxh::OpenImageSafe(&ScarLabel, ScarLabelName);

	WallScarLabel.CloneFrom(&WallLabel);

	const int * Size = SourceImage.GetImageSize();
	float distValue;
	for (int scx = 0; scx < Size[0]; scx++)
	{
		for (int scy = 0; scy < Size[1]; scy++)
		{
			for (int scz = 0; scz < Size[2]; scz++)
			{
				//WallScarLabel.SetPixelByGreyscale(scx, scy, scz, 0, 421);
				ZXHPixelTypeDefault ScarLabelValue = ScarLabel.GetPixelGreyscaleClosest(scx, scy, scz, 0);
				if (ScarLabelValue == 0)//PV
					continue;
				//------------------------------Get New input world coordinate-------------------------------- 
				float InputWorldCoord[] = { scx, scy, scz, 0 };
				SourceImage.ImageToWorld(InputWorldCoord);

				zxhImageModelingLinear GradientMod;
				GradientMod.SetImage(&LABloodPoolGaussianBlur);
				float  pwGrad[4] = { 0 };
				GradientMod.GetPixelGradientByWorld(pwGrad, InputWorldCoord[0], InputWorldCoord[1], InputWorldCoord[2], 0);
				float ix = pwGrad[0];
				float iy = pwGrad[1];
				float iz = pwGrad[2];
				float mag = sqrt(ix*ix + iy*iy + iz*iz);
				if (mag < ZXH_FloatInfinitesimal)
				{
					std::cout << "error: magnitude " << mag << " too small for node " << "\n";
					return -1;
				}
				float Ia = 0.01*ix / mag;
				float Ib = 0.01*iy / mag;
				float Ic = 0.01*iz / mag;
				int NumSteps = 1000;

				if (ScarLabelValue == 421)
				{				
					for (int step = 0; step <= NumSteps; step++) // 0.01mm for each step
					{
						// forward step
						float  NewInputWorldCoord[4] = { InputWorldCoord[0] + step*Ia, InputWorldCoord[1] + step*Ib, InputWorldCoord[2] + step*Ic, 0 };
						float NewInputImageCoord[] = { NewInputWorldCoord[0], NewInputWorldCoord[1], NewInputWorldCoord[2], 0 };
						SourceImage.WorldToImage(NewInputImageCoord);
						int cx = zxh::round(NewInputImageCoord[0]);
						int cy = zxh::round(NewInputImageCoord[1]);
						int cz = zxh::round(NewInputImageCoord[2]);
						bool bIsInsideImage = SourceImage.InsideImage(cx, cy, cz, 0);
						if (!bIsInsideImage)
							continue;

						WallScarLabel.SetPixelByGreyscale(cx, cy, cz, 0, 421);

						// backward step
						float  BackInputWorldCoord[4] = { InputWorldCoord[0] - step*Ia, InputWorldCoord[1] - step*Ib, InputWorldCoord[2] - step*Ic, 0 };
						float BackInputImageCoord[] = { BackInputWorldCoord[0], BackInputWorldCoord[1], BackInputWorldCoord[2], 0 };
						SourceImage.WorldToImage(BackInputImageCoord);
						//int  cx = zxh::round(BackInputImageCoord[0]);
						//int  cy = zxh::round(BackInputImageCoord[1]);
						//int  cz = zxh::round(BackInputImageCoord[2]);
						cx = zxh::round(BackInputImageCoord[0]);
						cy = zxh::round(BackInputImageCoord[1]);
						cz = zxh::round(BackInputImageCoord[2]);
						bIsInsideImage = SourceImage.InsideImage(cx, cy, cz, 0);
						if (!bIsInsideImage)
							continue;

						WallScarLabel.SetPixelByGreyscale(cx, cy, cz, 0, 421);
					}
				}
				if (ScarLabelValue == 422)
				{
					for (int step = 0; step <= NumSteps; step++) // 0.01mm for each step
					{
						// forward step
						float  NewInputWorldCoord[4] = { InputWorldCoord[0] + step*Ia, InputWorldCoord[1] + step*Ib, InputWorldCoord[2] + step*Ic, 0 };
						float NewInputImageCoord[] = { NewInputWorldCoord[0], NewInputWorldCoord[1], NewInputWorldCoord[2], 0 };
						SourceImage.WorldToImage(NewInputImageCoord);
						int cx = zxh::round(NewInputImageCoord[0]);
						int cy = zxh::round(NewInputImageCoord[1]);
						int cz = zxh::round(NewInputImageCoord[2]);
						bool bIsInsideImage = SourceImage.InsideImage(cx, cy, cz, 0);
						if (!bIsInsideImage)
							continue;

						WallScarLabel.SetPixelByGreyscale(cx, cy, cz, 0, 422);

						// backward step
						float  BackInputWorldCoord[4] = { InputWorldCoord[0] - step*Ia, InputWorldCoord[1] - step*Ib, InputWorldCoord[2] - step*Ic, 0 };
						float BackInputImageCoord[] = { BackInputWorldCoord[0], BackInputWorldCoord[1], BackInputWorldCoord[2], 0 };
						SourceImage.WorldToImage(BackInputImageCoord);
						//int  cx = zxh::round(BackInputImageCoord[0]);
						//int  cy = zxh::round(BackInputImageCoord[1]);
						//int  cz = zxh::round(BackInputImageCoord[2]);
						cx = zxh::round(BackInputImageCoord[0]);
						cy = zxh::round(BackInputImageCoord[1]);
						cz = zxh::round(BackInputImageCoord[2]);
					    bIsInsideImage = SourceImage.InsideImage(cx, cy, cz, 0);
						if (!bIsInsideImage)
							continue;

						WallScarLabel.SetPixelByGreyscale(cx, cy, cz, 0, 422);
					}
				}

			}
		}
	}

	for (int scx = 0; scx < Size[0]; scx++)
	{
		for (int scy = 0; scy < Size[1]; scy++)
		{
			for (int scz = 0; scz < Size[2]; scz++)
			{
				ZXHPixelTypeDefault WallLabelValue = WallLabel.GetPixelGreyscaleClosest(scx, scy, scz, 0);
				ZXHPixelTypeDefault WallScarLabelValue = WallScarLabel.GetPixelGreyscaleClosest(scx, scy, scz, 0);
				if ((WallScarLabelValue != 422) && (WallLabelValue>0))
					WallScarLabel.SetPixelByGreyscale(scx, scy, scz, 0, 421);
				if (WallLabelValue == 0)
					WallScarLabel.SetPixelByGreyscale(scx, scy, scz, 0, 0);

			}
		}
	}

	const char *SegResultName = WallScarLabelName.data();
	zxh::SaveImage(&WallScarLabel, SegResultName);

	return 1;
}
