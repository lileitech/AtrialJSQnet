
clc;clear all;
verbose=' -v 0 ' ;
fstdForGenerateProb=2;

foldtrain='C:\LeiLi\2020_MICCAI\LAscar2020\test_data\';
a=dir([foldtrain '*_*']); 
cd (foldtrain); 

for i=1:numel(a)

    CaseName=a(i).name; 
    %generate manual wall from manual LA
    strWhsLabelFilename=[foldtrain  CaseName '\en_seg_msp_M.nii.gz'];
    ManualLASeg=[foldtrain  CaseName '\atriumSegImgMO.nii.gz'];
    LA_surfaceName = [foldtrain  CaseName '\LA_MeshWall_M.nii.gz'];
    savecaseid = CaseName;
    savefold = [foldtrain  CaseName '\'];
    
%   command=['GenerateModelsFromLabels ' foldtrain ' ' CaseName]; system(command);  %Generate the LA_Mesh_M.stl 
%   command=['GenMeshWall ' foldtrain ' ' CaseName];   system(command); %generate LA_MeshWall_M.nii.gz 


    %------------------Generate LA_label_GauiisanBlur_M.nii.gz------------------
	%LABlur_M=[foldtrain  CaseName  '\LA_label_GauiisanBlur_M.nii.gz'];  
    %command=['zxhimageop -int ' ManualLASeg ' -vr 1 420 -vs 420 ' verbose]; system(command);
    %system(['zxhimageop -int ' ManualLASeg ' -o ' LABlur_M ' -gau 4 -v 0 ']); 
    %------------------to generate LAwall_gd_new.nii.gz ------------------
%     [imgLab,Sub3D]=zxhGenerateProbForLAwallSeg(strWhsLabelFilename,savecaseid, savefold);    
%     command=['zxhboundary -i ' ManualLASeg ' -o '  LA_surfaceName ' -R 1 420 0 0 -v 0 ']; system(command);
    
    LA_surfaceName = [foldtrain  CaseName '\scarSegImgM_wall.nii.gz'];
    if exist(LA_surfaceName, 'file')
        fprintf('Done');
    else
        %------------------generate LA_MeshWallLabel_M.nii.gz ------------------
        command=['GenMeshWallLabel ' foldtrain ' ' CaseName];system(command);%generate LA_MeshWallLabel_M.nii.gz      
        %------------------generate manual wall scar------------------
        command=['GenDistProb ' foldtrain ' ' CaseName]; system(command); % generate scarSegImgM_wall.nii.gz
    end
end       
     








