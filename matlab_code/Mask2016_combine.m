%% combine parts




%% combine with whole image predictions
base_dir   = '../results/PN_prob/';
res_dir    = '../results/favos_baseline/';


addpath('util/');

ratios     = {'64','55','56','65','46','66'};

for i = 1:length(classes)

class_name = classes{i};

mkdir([out_dir1,class_name]);
mkdir([out_dir2,class_name]);


images = dir(fullfile(DAVIS_dir,'JPEGImages/480p/',class_name,'/*.jpg'));
for k = 1:length(images)
    
    ann_img = imread([DAVIS_dir,'Annotations/480p/',class_name,'/00000.png']);
    ann_img = uint8(ann_img>0);
    
    num_obj = 1;
    im_name = images(k).name;
    im_name = im_name(1:end-4);
    

    img      = imread([DAVIS_dir,'JPEGImages/480p/', class_name,'/',im_name,'.jpg']);
    base_img = imread([base_dir,class_name,'/',im_name,'.png']);
    if (size(base_img,1)~=size(img,1)) || (size(base_img,2)~=size(img,2))
        base_img = imresize(base_img,[size(img,1),size(img,2)]);
    end
    base_img = double(base_img);
    base_img = base_img/max(max(base_img));
 
 
    #img_parts = double(img_parts);
    #img_parts = img_parts/max(max(img_parts));
    
    if k==1
       ratio = select_ratios(ratios, base_img, img_part);
    end
    
    img_pick  = img_parts;
 
    res_img   = base_img*str2num(ratio(1))*0.1 + img_pick*str2num(ratio(2))*0.1;


     
     imwrite(res_img,         [res_dir,classes{i},'/',im_name,'.jpg']);
     imwrite((res_img>0.5),   [res_dir,classes{i},'/',im_name,'.png']);
     

end



