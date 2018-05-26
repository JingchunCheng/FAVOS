function rate = combine_rate(ann_img, base_img, img_parts)

ratios     = {'64','55','56','65','46','66'};
J_all      = zeros(length(ratios), 1);
for i = 1:length(ratios)
 ratio     = ratios{i};
 base_img  = double(base_img);
 img_parts = double(img_parts);
 res_img   = base_img*str2num(ratio(1))*0.1 + img_parts*str2num(ratio(2))*0.1;
 J_all(i)  = jaccard_single(res_img<128, ann_img);
end

[~,pos] = max(J_all);
rate    = ratios{pos};
disp(rate);


end
