python infer_davis16.py ../models/ROISegNet_2016.caffemodel deploy.prototxt blackswan 1
matlab -nojvm -nodesktop -nodisplay -r "combine_mask('blackswan')"
