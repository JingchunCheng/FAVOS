# Demo script for FAVOS
# Usage: sh test_davis16.sh {GPU-id} {sequence-name}

cd demo && \
python infer_davis16.py ../models/ROISegNet_2016.caffemodel deploy.prototxt $2 $1 && \
matlab -nojvm -nodesktop -nodisplay -r "combine_mask('$2')" && \
cd ../ 
