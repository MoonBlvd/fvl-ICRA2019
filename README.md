# Egocentric Vision-based Future Vehicle Localization for Intelligent Driving Assistance Systems
*Yu Yao, Mingze Xu, Chiho Choi, David J. Crandall, Ella M. Atkins, and Behzad Dariush*

This repo contains source code of future vehicle localization (FVL)[1], implemented in Keras with tensorflow backend.

:boom: <span style="color:red">ATTENTION</span>: The current repo is a placeholder. The original Keras code is not released yet due to the HRI authority issue. The readers are redirected to a [pytorch implementation](https://github.com/MoonBlvd/tad-IROS2019) of the paper.

![introduction](/data/samples/ad.jpg?raw=true)

Following dependencies (or newer version):
	
	python3.5 or python3.6
	tensorflow-gpu=1.1.0
	keras=1.1.0.
The RNN encoder-decoder model:

![introduction](/data/samples/network.png?raw=true)

To train the model, run

	cd train
	sh run_train.sh
	
Check the command line arguments by ```python train.py --help```

To test the trained model, run
	
	cd test
	sh run_test.sh

We tested our model 

### Test results:
Test results on HEV-I dataset:

| Models              |               Easy Cases              |            Challenging Cases          |                All Cases               |
|------------------------------|:----------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|
| Linear                       |              31.49 / 17.04 / 0.68              |              107.93 / 56.29 / 0.33              |               72.37 / 38.04 / 0.50              |
| ConstAccel                   |              20.82 / 13.86 / 0.74              |               90.33 / 49.06 / 0.35              |               58.00 / 28.05 / 0.53              |
| Conv1D [2] |              18.84 / 12.09 / 0.75              |               37.95 / 20.97 / 0.64              |               29.06 / 16.84 / 0.69              |
| RNN-ED-X                     |              23.57 / 11.96 / 0.74              |               43.15 / 22.24 / 0.60              |               34.04 / 17.46 / 0.67              |
| RNN-ED-XE                    |              22.28 / 11.60 / 0.74              |               42.27 / 22.39 / 0.61              |               32.97 / 17.37 / 0.67              |
| RNN-ED-XO                    |               17.45 / 8.68 / 0.78              |               32.61 / 16.72 / 0.66              |               25.56 / 12.98 / 0.72              |
| RNN-ED-XOE                   | **16.72** / **8.52** / **0.80** | **32.05** / **16.63** / **0.66** | **24.92** / **12.86** / **0.73** |

Test results on KITTI dataset:

|          Models              |       FDE      |       ADE      |      FIOU     |
|------------------------------|:--------------:|:--------------:|:-------------:|
| Linear                       |      78.19     |      38.21     |      0.33     |
| ConstAccel                   |      55.66     |      25.78     |      0.39     |
| Conv1D [2]                   |      44.13     |      24.38     |      0.49     |
| Ours                         |    **37.11**   |    **17.88**   |    **0.53**   |

### Dataset Demo
![Alt Text](data/samples/hevi_demo.gif)

### Prediction Demo
![Alt Text](data/samples/demo.gif)

### Reference
[1] Yao Y, Xu M, Choi C, Crandall DJ, Atkins EM, Dariush B. Egocentric Vision-based Future Vehicle Localization for Intelligent Driving Assistance Systems. arXiv preprint arXiv:1809.07408. 2018 Sep 19.

[2] Yagi T, Mangalam K, Yonetani R, Sato Y. Future person localization in first-person videos. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2018 (pp. 7593-7602).