#!bin/bash
export CUDA_VISIBLE_DEVICES=$3
 
#Give input and output files as arguments !! 
python $1 > $2 2>&1 &