function [imgLab,Sub3D]=zxhGenerateProbForLAwallSeg(strWhsLabelFilename,savecaseid, savefold)
% author: Xiahai Zhuang
% date of creation:   2017-02
% current version:  
%	succ=zxhGenerateProbForLAwallSeg(strWhsLabelFilename, strPreSave, fstdForGenerateProb, iMaskDilateMM) 
% Save Files: strPreSave + "Label.nii.gz"; strPreSave +"Sub3D.nii.gz"(mask) strPreSave + "Prob_Label0/1/2/3.nii.gz"; all in 1x1x1 mm
%             Label with new label values for 0 (bk), 420 (LA blood), 429 (LA wall), 199 (other myo wall), 550 (other blood), 500 (LV)
%		* erode+dilate 1.5 mm for LA=420, aorta=820-825, RA=550, PA=850-855; all wall thickness init = 3 mm
%                        average wall thickness: 2.2\pm0.9mm, for MRI init 3 mm could be good
%		* dilated label - ori label = wall, set LA wall, and the other walls and LVmyo (200-250) to 200 
%		* dilate iMaskDilateMM (recommend: 4-6) on label 2 to generate a mask ---> LAwall+20mm for generating ROI of Sub3D
%		* zxhvolumelabelop -genprobf f imgref, f=fstdForGenerateProb (recommend: 2)
%    
	verbose=' -v 0 ' ;
	imgLab = [savefold savecaseid '_Label.nii.gz'];
% 	Sub3D = [savefold savecaseid '_Sub3D.nii.gz']; 
    Sub3D = [savefold 'LAwall_gd_new.nii.gz']; 

	img1 = [savefold savecaseid '__tmp1.nii.gz']; 
	img2 = [savefold savecaseid '__tmp2.nii.gz']; 
	img3 = [savefold savecaseid '__tmp3.nii.gz']; 
	img4 = [savefold savecaseid '__tmp4.nii.gz']; 
    
    %改变spacing，并且做nearest插值，resize,方便后面的dilation或者erosion的尺寸的取值
	%command=['zxhtransform ' strWhsLabelFilename ' -o ' imgLab ' -resave -spacing 1 1 1 -nearest ' verbose]; system(command);
    copyfile( strWhsLabelFilename, imgLab, 'f' ) ;%-----add by lei 2020.06.08
    
    %把RA(550),RV(600),PA(850),AO(820)的vauel改成550
    %other blood ->550
	command=['zxhimageop -int ' imgLab ' -vr 550 1000 -vs 550 ' verbose]; system(command);
    
    %把maskimage复制给img1,img4
	copyfile( imgLab, img1, 'f' ) ;  
	copyfile( imgLab, img4, 'f' ) ;  
	
    
	% set all expect LA+LV to 0, erode 1.5mm from LA-bk, and set to LA wall
    
    %把除了·LA(420),LV(500)·都设置为0
	command=['zxhimageop -int ' img1 ' -vr 0 400 -vr 550 1000 -vs 0 ' verbose]; system(command);
    
    %以0为背景，LA(420)为前景，erosion得到img2
    %command=['zxhimageop -int ' img1 ' -o ' img2 ' -DIs 0.5 0 420 ' verbose]; system(command); %-----add by lei 2020.03.05
	command=['zxhimageop -int ' img1 ' -o ' img3 ' -ERs 2 0 420 ' verbose]; system(command);%-----modify by lei 2020.06.08
    
    %img3= img1-img2,得到的就是刚刚LA  erosion出来的la wall
	command=['zxhimageop -int ' img1 ' -int ' img3 ' -o ' img3 ' -sub ' verbose]; system(command);
    
    %把得到的la wall 作为mask ，借此把la wall 添加进去，并把这部分的 value 设置成429
	command=['zxhimageop -int ' imgLab ' -o ' imgLab ' -vmaskimage ' img3 ' 1 1000  -vs 429 ' verbose]; system(command);
	
	% set LA to 0, erode 1.5mm from otherblood-bk, and set to other wall; set LVmyo to other wall
    
    %把LA去掉
	command=['zxhimageop -int ' img4 ' -vr 420 420 -vs 0 ' verbose]; system(command); 
    
    %根据上面“把RA(550),RV(600),PA(850),AO(820)的vauel改成550”，可以知道，这一步除了LA以外的部分erosion
	command=['zxhimageop -int ' img4 ' -o ' img2 ' -ERs 1.5 0 550 ' verbose]; system(command); 
    
    %同上，这里得到的就是其他的blood的erosion出来的wall
	command=['zxhimageop -int ' img4 ' -int ' img2 ' -o ' img3 ' -sub ' verbose]; system(command);
    
    %把这部分的wall作为mask ，借此把la wall 添加进去，并把这部分的 value 设置成199
	command=['zxhimageop -int ' imgLab ' -o ' imgLab ' -vmaskimage ' img3 ' 1 1000  -vs 199 ' verbose]; system(command);
	 

	 % set LAwall dilate resp 1.5 mm 
	command=['zxhimageop -int ' imgLab ' -DIs 1.5 0 429 ' verbose]; system(command);
	 % set other wall dilate 1.5 mm;  
	command=['zxhimageop -int ' imgLab ' -DIs 1.5 0 199 ' verbose]; system(command);
    
    %把myocardial的 value也设为199
	command=['zxhimageop -int ' imgLab ' -o ' imgLab ' -vr 199 250  -vs 199 ' verbose]; system(command);


    
    %把la wall的value的值 改设 为 100， 生成一个la wall
	command=['zxhimageop -int ' imgLab ' -o ' Sub3D ' -vr 429 429 -VS 100 ' verbose]; system(command);
    %command=['zxhimageop -int ' imgLab ' -o ' Sub3D ' -imageinfosrc2 ' strWhsLabelFilename]; system(command);

	
	delete(img1,img2,img3,img4, imgLab); 
	
end

% zxhvolumelabelop MeanLabel_new.nii.gz  test1.nii.gz -gvoutrange 
% zxhvolumelabelop test1.nii.gz  test2.nii.gz -rmnonwh 

