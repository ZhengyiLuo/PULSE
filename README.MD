
# Universal Humanoid Motion Representations for Physics-Based Control

Official implementation of ICLR 2023 spotlight paper: "Universal Humanoid Motion Representations for Physics-Based Control". In this work, we develop a humanoi motion latent space that is low dimensional (32), high coverage (99.8% of AMASS motion), can speed up downstream task for hierarchical RL, and can be randomly sampled as a generative model. 

Our proposed Physics-based Universal motion Latent SpacE (PULSE) is akin to a foundation model for control where downstream tasks ranging from simple locomotion, complex terrain traversal, to free-form motion tracking can all reuse this representation

This repo has a large amount of code overlap with [PHC](https://github.com/ZhengyiLuo/PHC); PULSE is more focused on the generative tasks. 

[[paper]](https://arxiv.org/abs/2310.04582) [[website]](https://www.zhengyiluo.com/PULSE/) 

<div float="center">
  <img src="assets/pulse.gif" />
</div>

## News 🚩

[March 19, 2024] Training and Evaluation code released.

## TODOs

- [ ] Add support for smplx/h (fingers!!!).

- [ ] Add suppport for more downstream tasks.

### Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. [Optional if only inference] Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
```

4. Use the following script to download trained models and sample data.

```
bash download_data.sh
```

this wil download amass_isaac_standing_upright_slim.pkl, which is a standing still pose for testing. 

## Evaluation 

### Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| f | focus on humanoid |
| Right click + WASD | change view port |
| Shift + Right click + WASD | change view port fast |
| r | reset episode |
| i | start random sampling PULSE |
| j | apply large force to the humanoid |
| l | record screenshot, press again to stop recording|
| ; | cancel screen shot|
| m | cancel termination based on imitation |

... more shortcut can be found in `phc/env/tasks/base_task.py`

Notes on rendering: I am using pyvirtualdisplay to record the video such that you can see all humanoids at the same time (default function will only capture the first environment). You can disable it using the flag `no_virtual_display=True`. 


### Run PULSE

```
python phc/run_hydra.py env.task=HumanoidImDistillGetup env=env_im_vae exp_name=pulse_vae_iclr robot.real_weight_porpotion_boxes=False learning=im_z_fit env.models=['output/HumanoidIm/phc_3/Humanoid_00258000.pth','output/HumanoidIm/phc_comp_3/Humanoid_00023501.pth'] env.motion_file=sample_data//amass_isaac_standing_upright_slim.pkl test=True env.num_envs=1  headless=False epoch=-1
```

**Press M (disable termination), and press I (start sampling), to see ramdomly sampled motion.**


### Train Downstream Tasks 

(more coming soon!)

Speed:
```
python phc/run_hydra.py env.task=HumanoidSpeedZ env=env_pulse_amp exp_name=pulse_speed robot.real_weight_porpotion_boxes=False learning=pulse_z_task env.models=['output/HumanoidIm/pulse_vae_iclr/Humanoid.pth'] env.motion_file=[insert your data pkl]
```





## Training



### Data Processing AMASS

We train on a subset of the [AMASS](https://amass.is.tue.mpg.de/) dataset.

For processing the AMASS, first, download the AMASS dataset from [AMASS](https://amass.is.tue.mpg.de/). Then, run the following script on the unzipped data:


After downloading AMASS, use the script `python scripts/data_process/convert_amass_data.py`

### Training PULSE

```
python phc/run_hydra.py env.task=HumanoidImDistillGetup env=env_im_vae exp_name=pulse_vae robot.real_weight_porpotion_boxes=False learning=im_z_fit env.models=['output/HumanoidIm/phc_3/Humanoid_00258000.pth','output/HumanoidIm/phc_comp_3/Humanoid_00023501.pth'] env.motion_file=[insert data pkl]
```


## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{
luo2024universal,
title={Universal Humanoid Motion Representations for Physics-Based Control},
author={Zhengyi Luo and Jinkun Cao and Josh Merel and Alexander Winkler and Jing Huang and Kris M. Kitani and Weipeng Xu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=OrOd8PxOO2}
}
```

Also consider citing these prior works that are used in this project:

```

@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}            

@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}     

@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

```

## References
This repository is built on top of the following amazing repositories:
* Main code framework is from: [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* Large amount of code is from: [PHC](https://github.com/ZhengyiLuo/PHC)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)

Please follow the lisence of the above repositories for usage. 





