
clc;clear all;
verbose=' -v 0 ' ;
fstdForGenerateProb=2;

foldtrain='C:\LeiLi\2020_MICCAI\LA2020\Data_60\test_data\';
a=dir([foldtrain 'p*']); 
cd (foldtrain); 

for i=1:numel(a)
    CaseName=a(i).name; 
    AutoWHS=[foldtrain  CaseName '\en_seg_msp.nii.gz'];
    ManualLASeg=[foldtrain  CaseName '\atriumSegImgMO.nii.gz'];
    
    %------------------Generate the LA_Mesh_M.stl------------------
    command=['GenerateModelsFromLabels ' foldtrain ' ' CaseName]; system(command); 
    LAmesh=[foldtrain CaseName '\LA_mesh.stl'];          
    newname=strrep(LAmesh, 'LA_mesh', 'LA_mesh_M');  
    copyfile(LAmesh,newname);
    delete(LAmesh)
    
    
    %------------------Generate LA_label_GauiisanBlur_M.nii.gz------------------
	LABlur_M=[foldtrain  CaseName  '\LA_label_GauiisanBlur_M.nii.gz'];  
    system(['zxhimageop -int ' ManualLASeg ' -o ' LABlur_M ' -gau 4 -v 0 ']);   
    command=['GenMeshWall ' foldtrain ' ' CaseName];   system(command); 
    command=['GenMeshWallLabel ' foldtrain ' ' CaseName];system(command); 
    
    %generate manual wall from manual LA
    strWhsLabelFilename=[foldtrain  CaseName '\en_seg_msp_M.nii.gz'];
    savecaseid = CaseName;
    savefold = [foldtrain  CaseName '\']; 
    [imgLab,Sub3D]=zxhGenerateProbForLAwallSeg(strWhsLabelFilename,savecaseid, savefold); 
    
    %generate manual wall scar 
    command=['GenDistProb ' foldtrain ' ' CaseName]; system(command);
    
    %command=['DetectPV ' foldtrain ' ' CaseName];system(command); 

  
     %------------------LearnGC: Generate dist file------------------
    enseg=[foldtrain a(i).name '\LA_MeshWallLabel_M.nii.gz'];%changed into fixed;
    verbose=' -v 0 ' ;
	img1=strrep(enseg, 'LA_MeshWallLabel_M', 'LA_MeshWall_test');
    WallImag=strrep(enseg, 'LA_MeshWallLabel_M', 'LA_MeshWall_421_M');
    ScarImag=strrep(enseg, 'LA_MeshWallLabel_M', 'LA_MeshWall_422_M');
    WallImagDismap=strrep(enseg, 'LA_MeshWallLabel_M', 'LA_MeshWall_421_dismap_M');
    ScarImagDismap=strrep(enseg, 'LA_MeshWallLabel_M', 'LA_MeshWall_422_dismap_M');
    
	command=['zxhimageop -int ' enseg ' -o ' img1 ' -vr 1 1000 -VS 1 ' verbose]; system(command);
	command=['zxhimageop -int ' enseg  ' -o ' WallImag ' -vr 421 421 -VS 421 ' verbose]; system(command); 
    command=['zxhimageop -int ' enseg  ' -o ' ScarImag ' -vr 422 422 -VS 422 ' verbose]; system(command);    
    command=['zxhvolumelabelop ' WallImag ' ' WallImagDismap ' -genmap 0 0 ' verbose]; system(command);      
    command=['zxhimageop -float ' WallImagDismap  ' -float ' img1 ' -o ' WallImagDismap  ' -mul' verbose]; system(command);    
    command=['zxhvolumelabelop ' ScarImag  '  '  ScarImagDismap ' -genmap 0 0 ' verbose]; system(command); 
    command=['zxhimageop -float ' ScarImagDismap  ' -float ' img1 ' -o ' ScarImagDismap  ' -mul' verbose]; system(command); 
    
    delete(img1,WallImag,ScarImag);
   
    %------------------------Generate mesh prob-------------------------
    command=['GenMeshProb ' foldtrain ' ' CaseName];system(command); 
   
end       
     








