# Unsupervised Keypoint Learning <br/> for Guiding Class-Conditional Video Prediction
An official implementation of the paper "Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction", NeurIPS 2019, [[paper](https://arxiv.org/abs/1910.02027)] [[supp](https://arxiv.org/abs/1910.02027)]

<p align="left">
  <img src='img/model_overview.png' width="860" title="Overview">
</p>


## Requirements

#### Dependencies
- [PyTorch](https://github.com/pytorch/pytorch) 1.0
- [torchfile](https://github.com/bshillingford/python-torchfile)

This is the pre-built [docker image](https://github.com/pytorch/pytorch) that this code can be run on.

#### Dataset
This code is for the Penn Action dataset. This dataset can be downloaded from [here](http://dreamdragon.github.io/PennAction/).

#### Pretrained VGG-Net
For the training, pretrained VGG19 network is needed. It can be downloaded from [here](https://github.com/machrisaa/tensorflow-vgg).


## Train

#### 1. Train the keypoints detector & image translator
```
python train_kd_it.py configs/penn.yaml
```

#### 2. Make pseudo-keypoints labels
```
python make_labels.py configs/penn.yaml
```

#### 3. Train the motion generator
```
python train_mogen.py configs/penn.yaml
```


## Test
```
python eval.py configs/penn.yaml
```


#### Pretrained model
1. [Keypoints Detector & Image Translator](https://github.com/pytorch/pytorch)
2. [Motion Generator](https://github.com/pytorch/pytorch)


## Results

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
<p align="center">
   <img src='img/table_head.png' width=707 />
<br>
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
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src='img/table_tail.png' width=500 />
</p>

#### Nemo-UvA
<p>
   <img src='img/nemo_ours.gif' width=92 />
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
   <img src='img/mgif_ours.gif' width=92 />
   <img src='img/mgif_1.gif' width=92 />
   <img src='img/mgif_2.gif' width=92 />
   <img src='img/mgif_3.gif' width=92 />
   <img src='img/mgif_4.gif' width=92 />
   <img src='img/mgif_5.gif' width=92 />
   <img src='img/mgif_6.gif' width=92 />
   <img src='img/mgif_7.gif' width=92 />
   <img src='img/mgif_8.gif' width=92 />
</p>


## Related Works
[Villegas et. al.](https://github.com/rubenvillegas/icml2017hierchvid)
[Wichers et. al.](https://github.com/brain-research/long-term-video-prediction-without-supervision)
[Li et. al.](https://github.com/Yijunmaverick/FlowGrounded-VideoPrediction)


## Citation
Please cite our paper when you use this code.
```
@inproceedings{yunji_neurips_2019,
  title={Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction},
  author={Kim, Yunji and Nam, Seonghyeon and Cho, In and Kim, Seon Joo},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```
