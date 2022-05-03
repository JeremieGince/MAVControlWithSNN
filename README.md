# MAVControlWithSNN

Disclaimer: **Project currently under development**

## Abstract

In this project, neural dynamics *Leaky-Integrate-and-Fire* (LIF) and 
*Adaptive-Leaky-Integrate-and-Fire* (ALIF) from neuroscience are compared 
with a multilayer perceptron (MLP) from machine learning on a control task. The 
said task is to land a micro air vehicle (MAV) in a simplified environment designed 
with the Unity game engine. Artificial neural networks, all trained by reinforcement 
learning with the *Proximal Policy Optimization* (PPO) algorithm, show that the 
accomplishment of such a task is feasible, but fail to achieve results allowing to move 
to a more complex step. Indeed, it is shown that MLP succeeds in landing the MAV with a 
single degree of freedom using an optical flow camera and an event camera while the ALIF 
and LIF dynamics fail to do so the majority of the time. Finally, a development pipeline 
for accomplishing this goal is presented along with avenues of refinement for the 
continuation of the project.



## Environment
The MAV used in this project is a quadrotor with a event camera attached to the front and a optical flow camera 
attached on the belly.
<p align="center"> <img width="900" height="400" src="https://github.com/JeremieGince/MAVControlWithSNN/blob/main/figures/env_mav.png?raw=true"> </p>

The environment contained a platform where the need to land.
<p align="center"> <img width="900" height="400" src="https://github.com/JeremieGince/MAVControlWithSNN/blob/main/figures/env_floor.png?raw=true"> </p>

In addition to the platform, some markers with random textures are placed randomly in the space.
<p align="center"> <img width="900" height="400" src="https://github.com/JeremieGince/MAVControlWithSNN/blob/main/figures/env_markers_materials.png?raw=true"> </p>

Finally, the environment looks like this:
<p align="center"> <img width="900" height="400" src="https://github.com/JeremieGince/MAVControlWithSNN/blob/main/figures/env_mav_markers.png?raw=true"> </p>



## Sensors
The event camera placed on the front of the MAV is used to detect the variation of intensity in the scene.
<p align="center"> <img width="900" height="400" src="https://github.com/JeremieGince/MAVControlWithSNN/blob/main/figures/MAV_eventCam_showcase.png?raw=true"> </p>



## Neural dynamics
The neural dynamics used in this project are *Leaky-Integrate-and-Fire* (LIF) and *Adaptive-Leaky-Integrate-and-Fire* 
(ALIF). Those dynamics are implemented and come from the following project: 
- [SNNImageClassification](https://github.com/JeremieGince/SNNImageClassification).



## Training
The pipeline for training the neural network is as follows:
<p align="center"> <img width="900" height="400" src="https://github.com/JeremieGince/MAVControlWithSNN/blob/main/figures/training_pipeline.png?raw=true"> </p>



## Results
The results of the training with one degree of freedom are as follows:

| Model       | Inputs          | Mean Rewards | Std Rewards |
|-------------|-----------------|--------------|-------------|
| MLP-128x128 | P               | **1.000**    | 0.000       |
| MLP-128x128 | P+V             | **1.000**    | 0.000       |
| MLP-128x128 | OF              | 0.000        | 0.000       |
| MLP-128x128 | OFH             | **1.000**    | 0.000       |
| MLP-128x128 | EventCam8x8     | **1.000**    | 0.000       |
| MLP-128x128 | OFH+EventCam8x8 | -1.000       | 0.000       |
| LIF-10x5    | OFH             | -0.080       | 0.997       |
| LIF-64x8    | EventCam8x8     | -0.980       | 0.199       |
| LIF-64x8    | OFH+EventCam8x8 | -0.940       | 0.341       |
| ALIF-10x5   | OFH             | -0.300       | 0.954       |
| ALIF-64x8   | EventCam8x8     | -0.140       | 0.990       |
| ALIF-64x8   | OFH+EventCam8x8 | -0.960       | 0.280       |

- P: position of the MAV [x, y, z];
- V: linear velocity of the MAV [Vx, Vy, Vz];
- OF: optical flow represented in scalar;
- OFH: optical flow represented in one hot vector;
- EventCam8x8: event camera of shape 8x8.

 
## Requirements

### Python:

- ```pip install -r requirements.txt```

### Unity:
From package manager:
- ML Agents Version 2.1.0-exp.1

From Assets Store
- OpenCvSharp+Unity


# License
[Apache License 2.0](LICENSE.md)


# Citation
```
@article{Gince_MAVControlWithSNN_2022,
  title={MAV Control With SNN},
  author={Gince, Jérémie},
  year={2022},
  publisher={Université Laval},
  url={https://github.com/JeremieGince/MAVControlWithSNN},
}
```


