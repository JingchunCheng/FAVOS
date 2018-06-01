function combine_mask(class_name)

DAVIS_dir = '../data/DAVIS2016/';
base_dir   = '../results/favos_baseline/';
part_dir   = '../results-demo/res_part/';
res_dir    = '../results-demo/res_favos/';


assert(exist(base_dir)>0, 'Please download baseline results "results/favos_baseline".');
assert(exist(part_dir)>0, 'Please run part segmentation first.');

if ~exist(res_dir)
   mkdir(res_dir);
end


%% combine parts
mkdir([res_dir,class_name]);

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
    [H, W, ~] = size(img);
    
    
   mask_img = zeros(size(ann_img));
   mask_num = zeros(size(ann_img));


   mask_img_sim   = zeros(size(ann_img));
   mask_img_score = zeros(size(ann_img));

   roi_seg_file = [part_dir, class_name,'/',im_name,'.mat'];
   load(roi_seg_file);

   
   box_num  = size(mask,1);


  
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
       mask_num(box(2):box(4),box(1):box(3))       =  mask_num(box(2):box(4),box(1):box(3))  + 1;
     end

     mask_img_num = double(mask_num);
      
     mask_num(mask_num == 0) = 1;
     mask_img  = mask_img./mask_num;
     mask_img  = mask_img/max(max(mask_img));
     
     img_parts = mask_img;


%% combine with whole image predictions 
    num_obj = 1;
    im_name = images(k).name;
    im_name = im_name(1:end-4);
    
    
    if k == 1
       load('weights.mat');
       ratio = weights(find(strcmp(classes,class_name)>0));
    end

    base_img = imread([base_dir,class_name,'/',im_name,'.png']);
    if (size(base_img,1)~=size(ann_img,1)) || (size(base_img,2)~=size(ann_img,2))
        base_img = imresize(base_img,[size(ann_img,1),size(ann_img,2)]);
    end
    base_img = double(base_img);
    base_img = base_img/max(max(base_img));
   
    img_pick  = img_parts;
 
    res_img   = base_img*ratio + img_pick*(1-ratio);

    % soft output
    % imwrite(res_img,         [res_dir,class_name,'/',im_name,'.jpg']);
    imwrite((res_img>0.5),   [res_dir,class_name,'/',im_name,'.png']);

     
end

quit;

