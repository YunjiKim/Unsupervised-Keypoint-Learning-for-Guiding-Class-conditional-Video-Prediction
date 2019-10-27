# Unsupervised Keypoint Learning <br/> for Guiding Class-Conditional Video Prediction
An official implementation of the paper "Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction", NeurIPS, 2019. [[paper](https://arxiv.org/abs/1910.02027)] [[supp](https://arxiv.org/abs/1910.02027)]

<p align="left">
  <img src='img/model_overview.png' width="860" title="Overview">
</p>


## I. Requirements

- Linux
- NVIDIA GeForce GTX 1080Ti
- Tensorflow 1.12.0
- Python3 (>= 3.5.2)

### Dependencies

You can install packages by running pip install -r requirements.txt. <br/>
Or you can download our prebuilt [docker image](https://cloud.docker.com/repository/registry-1.docker.io/join16/python3-cuda),
by running `docker pull join16/python3-cuda:3.5-cuda9.0-nips2019`. <br/>
If you want, you can build docker image manually,
by running `docker build -t {image_name} .`


#### ※ Dataset
This code is for the Penn Action dataset. The dataset can be downloaded [here](http://dreamdragon.github.io/PennAction/).
After download PennAction.tar.gz, unzip and then run following code to prepare dataset.

```sh
./prepare_penn_dataset.sh {unzipped_original_dataset_dir} {dest_dir}
```

#### ※ Pretrained VGG-Net
For the training, pretrained VGG19 network is needed. It can be downloaded [here](https://github.com/machrisaa/tensorflow-vgg).

## II. Train

###### ※※※ Please adjust the paths for inputs and outputs in the configuration file. ※※※

#### 1. Train the keypoints detector & image translator
```
python train.py --mode detector_translator --config configs/penn.yaml
```

#### 2. Make pseudo-keypoints labels
```
python make_pseudo_labels.py --config configs/penn.yaml --checkpoint {path/to/detector_translator/checkpoint}
```

#### 3. Train the motion generator
```
python train.py --mode motion_generator --config configs/penn.yaml
```


## III. Test
```
python evaluate.py --config configs/penn.yaml \
    --checkpoint_stage1 {path/to/detector_translator/checkpoint} \
    --checkpoint_stage2 {path/to/motion_generator/checkpoint} \
    --save_dir {path/to/save/results}
```


#### Pretrained model
1. [Keypoints Detector & Image Translator](https://www.dropbox.com/s/0ltjzjc2quvq69t/stage1.zip?dl=0)
2. [Motion Generator](https://github.com/pytorch/pytorch)


## IV. Results

###### ※※※ All videos were generated from a single input image. ※※※

#### Penn Action
<p>
   <img src='img/tennis_serve.gif' width=92 />
   <img src='img/tennis_forehand.gif' width=92 />
   <img src='img/pull_up.gif' width=92 />
   <img src='img/jumping_jacks.gif' width=92 />
   <img src='img/golf_swing.gif' width=92 />
   <img src='img/clean_and_jerk.gif' width=92 />
   <img src='img/baseball_swing.gif' width=92 />
   <img src='img/baseball_pitch.gif' width=92 />
   <img src='img/squats.gif' width=92 />
<br>
   <img src='img/classes.png' width=860 />
</p>

#### UvA-NEMO
<p>
   <img src='img/nemo_0.gif' width=92 />
   <img src='img/nemo_1.gif' width=92 />
   <img src='img/nemo_2.gif' width=92 />
   <img src='img/nemo_3.gif' width=92 />
   <img src='img/nemo_4.gif' width=92 />
   <img src='img/nemo_5.gif' width=92 />
   <img src='img/nemo_6.gif' width=92 />
   <img src='img/nemo_7.gif' width=92 />
   <img src='img/nemo_8.gif' width=92 />
</p>

#### MGIF
<p>
   <img src='img/mgif_0.gif' width=92 />
   <img src='img/mgif_1.gif' width=92 />
   <img src='img/mgif_2.gif' width=92 />
   <img src='img/mgif_3.gif' width=92 />
   <img src='img/mgif_4.gif' width=92 />
   <img src='img/mgif_5.gif' width=92 />
   <img src='img/mgif_6.gif' width=92 />
   <img src='img/mgif_7.gif' width=92 />
   <img src='img/mgif_8.gif' width=92 />
</p>

###### ※※※ Qualitative comparison of the results. ※※※

<p align="center">
   <img src='img/tb_blank.png' width=92 />
   &nbsp;&nbsp;&nbsp;&nbsp;
   <img src='img/tb_head_1.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/tb_head_2.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/tb_blank.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/tb_blank.png' width=92 />
   <img src='img/tb_head_3.png' width=92 />
   <img src='img/tb_blank.png' width=92 />
   <img src='img/tb_blank.png' width=92 />
<br>
   <img src='img/penn_text.png' width=92 />
   &nbsp;&nbsp;&nbsp;&nbsp;
   <img src='img/penn_start.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/input_act.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/penn_real.gif' width=92 />
   &nbsp;&nbsp;
   <img src='img/penn_ours.gif' width=92 />
   <img src='img/penn_villegas.gif' width=92 />
   <img src='img/penn_wichers.gif' width=92 />
   <img src='img/penn_li.gif' width=92 />
<br>
   <img src='img/penn_text.png' width=92 />
   &nbsp;&nbsp;&nbsp;&nbsp;
   <img src='img/penn_start_2.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/input_act_2.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/penn_real_2.gif' width=92 />
   &nbsp;&nbsp;
   <img src='img/penn_ours_2.gif' width=92 />
   <img src='img/penn_villegas_2.gif' width=92 />
   <img src='img/penn_wichers_2.gif' width=92 />
   <img src='img/penn_li_2.gif' width=92 />
<br>
   <img src='img/nemo_text.png' width=92 />
   &nbsp;&nbsp;&nbsp;&nbsp;
   <img src='img/nemo_start.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/-.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/nemo_real.gif' width=92 />
   &nbsp;&nbsp;
   <img src='img/nemo_ours.gif' width=92 />
   <img src='img/-.png' width=92 />
   <img src='img/nemo_wichers.gif' width=92 />
   <img src='img/nemo_li.gif' width=92 />
<br>
   <img src='img/mgif_text.png' width=92 />
   &nbsp;&nbsp;&nbsp;&nbsp;
   <img src='img/mgif_start.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/-.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/mgif_real.gif' width=92 />
   &nbsp;&nbsp;
   <img src='img/mgif_ours.gif' width=92 />
   <img src='img/-.png' width=92 />
   <img src='img/mgif_wichers.gif' width=92 />
   <img src='img/mgif_li.gif' width=92 />
<br>
   <img src='img/tb_blank_2.png' width=92 />
   &nbsp;&nbsp;&nbsp;&nbsp;
   <img src='img/tb_blank_2.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/tb_blank_2.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/tb_tail_1.png' width=92 />
   &nbsp;&nbsp;
   <img src='img/tb_tail_2.png' width=92 />
   <img src='img/tb_tail_3.png' width=92 />
   <img src='img/tb_tail_4.png' width=92 />
   <img src='img/tb_tail_5.png' width=92 />
</p>

## V. Related Works
Learning to Generate Long-term Future via Hierarchical Prediction, Villegas et. al., ICML, 2017. [[code](https://github.com/rubenvillegas/icml2017hierchvid)]<br>
Hierarchical Long-term Video Prediction without Supervision, Wichers et. al., ICML, 2018. [[code](https://github.com/brain-research/long-term-video-prediction-without-supervision)]<br>
Flow-Grounded Spatial-Temporal Video Prediction from Still Images, Li et. al., ECCV, 2018. [[code](https://github.com/Yijunmaverick/FlowGrounded-VideoPrediction)]


### ※ Citation
Please cite our paper when you use this code.
```
@inproceedings{yunji_neurips_2019,
  title={Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction},
  author={Kim, Yunji and Nam, Seonghyeon and Cho, In and Kim, Seon Joo},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```
