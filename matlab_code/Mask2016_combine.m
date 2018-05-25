clear all; close all; clc;
DAVIS_dir = '../data/DAVIS2016/';
classes   = textread([DAVIS_dir,'ImageSets/2016/val.txt'],'%s\n');

base_dir   = '../results/favos_baseline/';
res_dir    = '../results/res_favos/';

addpath('util/');

ratios     = {'64','55','56','65','46','66'};

mkdir(res_dir);


%% combine parts
for i = 1:length(classes)

class_name = classes{i};
mkdir([res_dir,class_name]);

images = dir(fullfile(DAVIS_dir,'JPEGImages/480p/',class_name,'/*.jpg'));
for k = 1:length(images)
    
    ann_img = imread([DAVIS_dir,'Annotations/2017/',class_name,'/00000.png']);
    ann_img = uint8(ann_img>0);
    
    num_obj = 1;
    im_name = images(k).name;
    im_name = im_name(1:end-4);
    
    if exist([out_dir,classes{i},'/',im_name,'.png'])
        continue
    end

    img      = imread([DAVIS_dir,'JPEGImages/480p/', class_name,'/',im_name,'.jpg']);
    base_img = imread([base_dir,class_name,'/',im_name,'.png']);
    if (size(base_img,1)~=size(img,1)) || (size(base_img,2)~=size(img,2))
        base_img = imresize(base_img,[size(img,1),size(img,2)]);
    end
    [H, W, ~] = size(img);
    
    
   mask_img = zeros(size(ann_img));
   mask_num = zeros(size(ann_img));


   mask_img_sim   = zeros(size(ann_img));
   mask_img_score = zeros(size(ann_img));

   roi_seg_file = [res_dir,class_name,'/',im_name,'.mat'];
   load(roi_seg_file);

   roi_sim_file     = [sim_dir,'/',classes{i},'/',im_name,'_sim.mat'];
   roi_sim          = load(roi_sim_file);
   roi_sim.sim(roi_sim.sim>100) = 100;
   roi_sim.sim      = roi_sim.sim/max(max(roi_sim.sim)); 
   [res_sim, pos_sim]     = min(roi_sim.sim);
   res_sim          = 1 - res_sim;
   num_parts     = size(roi_sim.rois, 1);


   roi_pool_file = [part_dir, classes{i},'.mat'];
   roi_pool      = load(roi_pool_file);
   
   box_num  = size(mask,1);

   assert(num_parts == box_num, 'part num == box num');
   assert(length(roi_pool.scores) == size(roi_sim.sim, 1), 'len(part score) == len (feat pool)')

  
     for j = 1:box_num
         
        box      = rois(j,:);
        box      = box(2:end);

        
       [~,~,roi_h,roi_w] = size(mask);
       
        box    = round(box);
        box(3) = box(3)-box(1)+1;
        box(4) = box(4)-box(2)+1;
        ww     = box(3);
        hh     = box(4);

        patch = imcrop(img,box);
        if numel(patch) == 0
            continue
        end      
        
        tmp_mask = mask(j,:,:,:);
        [~,~,roi_h,roi_w] = size(tmp_mask);
        tmp_mask = reshape(tmp_mask,[2, roi_h,roi_w]);
        res_mask = exp(tmp_mask(2,:,:))./(exp(tmp_mask(1,:,:))+exp(tmp_mask(2,:,:)));
        res_mask = reshape(res_mask,[roi_h,roi_w]);

        
       roi_mask = imresize(res_mask,[hh, ww]);
       box      = rois(j,:);
       box      = round(box(2:end));    
       if box(2) == 0
           box(2) = 1;
           box(4) = box(4)+1;
       end
       if box(1) == 0
           box(1) = 1;
           box(3) = box(3)+1;  
       end
       box(3) = min(box(3), W);
       box(4) = min(box(4), H);
       if (hh~=box(4)-box(2)+1) || (ww~=box(3)-box(1)+1)
           roi_mask = roi_mask(1:box(4)-box(2)+1,1:box(3)-box(1)+1);
       end

       mask_img(box(2):box(4),box(1):box(3))       =  mask_img(box(2):box(4),box(1):box(3))  + roi_mask;
       mask_img_sim(box(2):box(4),box(1):box(3))   =  mask_img_sim(box(2):box(4),box(1):box(3))  + roi_mask*res_sim(j);
       mask_img_score(box(2):box(4),box(1):box(3)) =  mask_img_score(box(2):box(4),box(1):box(3))  + roi_mask*res_sim(j)*roi_pool.scores(pos_sim(j));
       mask_num(box(2):box(4),box(1):box(3))       =  mask_num(box(2):box(4),box(1):box(3))  + 1;
     end

     mask_img_num = double(mask_num);
     if max(max(mask_num)) > 255
         mask_img_num  = mask_img_num*1.0/max(max(mask_img_num))*255;
     end
     imwrite(uint8(mask_img_num), [out_dir,classes{i},'/',im_name,'_parts_num.png']);
     
     mask_num(mask_num == 0) = 1;
     mask_img = mask_img./mask_num;
     mask_img = mask_img/max(max(mask_img));
     
     img_part = mask_img;





%% combine with whole image predictions
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
    
    if k==1
       ratio = select_ratios(ratios, base_img, img_part);
    end
    
    img_pick  = img_parts;
 
    res_img   = base_img*str2num(ratio(1))*0.1 + img_pick*str2num(ratio(2))*0.1;


     imwrite(res_img,         [res_dir,classes{i},'/',im_name,'.jpg']);
     imwrite((res_img>0.5),   [res_dir,classes{i},'/',im_name,'.png']);
     
end



