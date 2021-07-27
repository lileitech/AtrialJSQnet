#include "STLreader.h"

STLreader::STLreader(std::string meshfilepath)//构造函数
{
	//---------------读入生成的mesh-----------------------------------------//
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	char * meshFileName = (char *)meshfilepath.c_str();  //把string转成char型
	reader->SetFileName(meshFileName);
	reader->Update();

	//---------------对mesh进行抽取，增大网格间距-----------------------------------------//
	vtkSmartPointer<vtkDecimatePro> decimation =
		vtkSmartPointer<vtkDecimatePro>::New();
	decimation->SetInputData(reader->GetOutput());
	decimation->SetTargetReduction(0.000000001);//有0.000000001的三角面片被移除
	decimation->Update();

	decimated = decimation->GetOutput();

	/*decimated = reader->GetOutput();*/

	/*
	//---------------显示抽取后的mesh-----------------------------------------//
	//输出一些值，来检验mesh
	vtkIdType ptId = 100;
	vtkIdList* clist = vtkIdList::New();

	std::cout << "ptId坐标：" << endl;
	std::cout << decimated->GetPoint(ptId)[0] << "\t";//输入下标，得到相连的周围点的坐标值
	std::cout << decimated->GetPoint(ptId)[1] << "\t";
	std::cout << decimated->GetPoint(ptId)[2] << endl;
	std::cout << endl;

	decimated->GetPointCells(ptId, clist);//输入下标，得到周围点的坐标值
	std::cout << "ptId周围点的坐标：" << endl;
	for (size_t j = 0; j < clist->GetNumberOfIds(); j++)
	{
		std::cout << decimated->GetPoint(clist->GetId(j))[0] << "\t";
		std::cout << decimated->GetPoint(clist->GetId(j))[1] << "\t";
		std::cout << decimated->GetPoint(clist->GetId(j))[2] << endl;
		std::cout << endl;
	}


	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(decimation->GetOutputPort());
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	renderer->AddActor(actor);
	renderer->SetBackground(.3, .6, .3); // Background color green 
	renderWindow->Render();
	renderWindowInteractor->Start();
	*/
}


STLreader::~STLreader()
{
}
