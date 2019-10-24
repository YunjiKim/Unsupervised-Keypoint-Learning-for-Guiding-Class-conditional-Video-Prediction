#!/bin/bash

original_data_dir=${1}
output_data_dir=${2}

mkdir -p ${output_data_dir}
mv ${original_data_dir}/frames ${output_data_dir}/frames
mv ${original_data_dir}/labels ${output_data_dir}/labels
cp assets/penn_split/* ${output_data_dir}/