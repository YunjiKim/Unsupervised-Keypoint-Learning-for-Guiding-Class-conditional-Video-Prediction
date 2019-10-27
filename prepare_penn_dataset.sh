#!/bin/bash

original_data_dir=${1}
output_data_dir=${2}

mkdir -p ${output_data_dir}
mv ${original_data_dir}/frames ${output_data_dir}/frames

cp assets/penn_split/* ${output_data_dir}/
