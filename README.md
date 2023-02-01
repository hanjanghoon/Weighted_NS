External Knowledge Selection with Weighted Negative Sampling in Knowledge-grounded Task-oriented Dialogue Systems <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />
====================================

Implements the model described in the following paper [External Knowledge Selection with Weighted Negative Sampling in Knowledge-grounded Task-oriented Dialogue Systems](https://arxiv.org/abs/2209.02251/) in DSTC10_track2_task2 2022.

```
@misc{https://doi.org/10.48550/arxiv.2209.02251,
  doi = {10.48550/ARXIV.2209.02251},
  url = {https://arxiv.org/abs/2209.02251},
  author = {Han, Janghoon and Shin, Joongbo and Song, Hosung and Jo, Hyunjik and Kim, Gyeonghun and Kim, Yireun and Choi, Stanley Jungkyu},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {External Knowledge Selection with Weighted Negative Sampling in Knowledge-grounded Task-oriented Dialogue Systems},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```


![dstc10](https://user-images.githubusercontent.com/32722198/216006375-aa146a4b-04da-4eff-8cd0-be3f653aa7cc.png)

Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.12.1 cudatoolkit=11.4 
pip install -r requirements.txt
```


Preparing Data and Checkpoints
-------------

### Model Checkpoints. 

- [post-trained checkpoint for generation, fine-tuned checkpoint for generation][1]

### Data for training
- [Synthetic training data][2]

--------

### Automatic Data Construction for training  

```
please refer data_processing/make_dstc10/make_synthetic_dstc10.py for data construction
```

### training and evaluation
```shell
sh paper_(run/train)_(task_name).sh
```

[1]: https://drive.google.com/file/d/15IDkzb-vmk0dYaJtfMzbOCLw0Fj641eK/view?usp=share_link
[2]: https://drive.google.com/file/d/10-GPMLFeV3lzmf2RrAAlB8gDUyD4PWTZ/view?usp=share_link


