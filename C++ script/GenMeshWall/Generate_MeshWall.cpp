#include <string.h>
#include <iostream> 

#include <time.h> 
#include <math.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "STLreader.h"
#include "zxhImageGipl.h"
using namespace std;

int main(int argc, char* argv[])
{
	string mainfold, Datafold, TargetImageName, MeshWallName, mesh_name;

	//mainfold = "c:\\leili\\2020_miccai\\la2020\\train_data";
	//string  casename = "patient_1";

	mainfold = argv[1];
	string  casename = argv[2];

	string PathName = mainfold + "\\" + casename;
	TargetImageName = PathName + "\\enhanced.nii.gz";
	mesh_name = PathName + "\\LA_Mesh_M.stl";
	MeshWallName = PathName + "\\LA_MeshWall_M.nii.gz";



	zxhImageData SourceImage, MeshWallImage;
	zxh::OpenImageSafe(&SourceImage, TargetImageName);
	MeshWallImage.NewImage(SourceImage.GetImageInfo());


	const int * Size = SourceImage.GetImageSize();
	string type;

	STLreader *stlreader = new STLreader(mesh_name);
	vtkSmartPointer<vtkPolyData> LAMesh = stlreader->decimated;	//load in the mesh
	const int iNumOfMeshPoints = LAMesh->GetNumberOfPoints(); //the number of mesh point

	for (int ptId = 0; ptId < iNumOfMeshPoints; ptId++)
	{
		float MeshNode_P2I_Coor[] = { LAMesh->GetPoint(ptId)[0], LAMesh->GetPoint(ptId)[1], LAMesh->GetPoint(ptId)[2], 0 };
		SourceImage.GetImageInfo()->PhysicalToImage(MeshNode_P2I_Coor);//物理坐标转成图像坐标
		int scx = zxh::round(MeshNode_P2I_Coor[0]);
		int scy = zxh::round(MeshNode_P2I_Coor[1]);
		int scz = zxh::round(MeshNode_P2I_Coor[2]);
		//int scz = zxh::round(Size[2] - MeshNode_P2I_Coor[2]);
		
		bool bIsInsideImage = SourceImage.InsideImage(scx, scy, scz, 0);
		if (!bIsInsideImage)//放弃这种点
			continue;


		MeshWallImage.SetPixelByGreyscale(scx, scy, scz, 0, 420);

	}

	const char *SegResultName = MeshWallName.data();
	zxh::SaveImage(&MeshWallImage, SegResultName);

	return 1;
}

