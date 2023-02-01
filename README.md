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
This code is reimplemented as a fork of [huggingface/transformers][2].

![alt text](https://user-images.githubusercontent.com/32722198/119294465-79847080-bc8f-11eb-80b1-1dfafb0e1b6c.png)

Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```


Preparing Data and Checkpoints
-------------

### Post-trained and fine-tuned Checkpoints

We provide following post-trained and fine-tuned checkpoints. 

- [fine-grained post-trained checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][3]
- [fine-tuned checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][4]


### Data pkl for Fine-tuning (Response Selection)
We used the following data for post-training and fine-tuning
- [fine-grained post-training dataset and fine-tuning dataset for 3 benchmarks (ubuntu, douban, e-commerce)][5]


Original version for each dataset is availble in [Ubuntu Corpus V1][6], [Douban Corpus][7], and [E-Commerce Corpus][8], respectively.


Fine-grained Post-Training
--------

##### Making Data for post-training and fine-tuning  

```
Data_processing.py
```


### Post-training Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

```shell
python -u FPT/ubuntu_final.py --num_train_epochs 25
python -u FPT/douban_final.py --num_train_epochs 27
python -u FPT/e_commmerce_final.py --num_train_epochs 34
```

### Fine-tuning Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

###### Taining 
```shell
To train the model, set `--is_training`
python -u Fine-Tuning/Response_selection.py --task ubuntu --is_training
python -u Fine-Tuning/Response_selection.py --task douban --is_training
python -u Fine-Tuning/Response_selection.py --task e_commerce --is_training
```
###### Testing
```shell
python -u Fine-Tuning/Response_selection.py --task ubuntu
python -u Fine-Tuning/Response_selection.py --task douban 
python -u Fine-Tuning/Response_selection.py --task e_commerce
```


Training Response Selection Models
--------

### Model Arguments

##### Fine-grained post-training

| task_name  | data_dir                                  |  checkpoint_path                    |
| ---------- | ---------------------                     |  -----------------------------------|
| ubuntu     | ubuntu_data/ubuntu_post_train.pkl         | FPT/PT_checkpoint/ubuntu/bert.pt    |
| douban     | douban_data/douban_post_train.pkl         | FPT/PT_checkpoint/douban/bert.pt    |
| e-commerce | e_commerce_data/e_commerce_post_train.pkl | FPT/PT_checkpoint/e_commerce/bert.pt|

##### Fine-tuning

| task_name     | data_dir                                  |  checkpoint_path                         |
| ----------    | ---------------------                     |  ----------------------------------------|
| ubuntu        | ubuntu_data/ubuntu_dataset_1M.pkl         | Fine-Tuning/FT_checkpoint/ubuntu.0.pt    |
| douban        | douban_data/douban_dataset_1M.pkl         | Fine-Tuning/FT_checkpoint/douban.0.pt    |
| e-commerce    | e_commerce_data/e_commerce_dataset_1M.pkl | Fine-Tuning/FT_checkpoint/e_commerce.0.pt|



Performance
----------

We provide model checkpoints of BERT_FP, which obtained new state-of-the-art, for each dataset.

| Ubuntu         | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [BERT_FP]      | 0.911 | 0.962 | 0.994 |

| Douban         | MAP   | MRR   | P@1   | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| [BERT_FP]      | 0.644 | 0.680 | 0.512 | 0.324 | 0.542 | 0.870 |

| E-Commerce     | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [BERT_FP]      | 0.870 | 0.956 | 0.993 |



[1]: https://github.com/huggingface/transformers
[2]: https://github.com/taesunwhang/BERT-ResSel
[3]: https://drive.google.com/file/d/1-4E0eEjyp7n_F75TEh7OKrpYPK4GLNoE/view?usp=sharing
[4]: https://drive.google.com/file/d/1n2zigNDiIArWtsiV9iUQLwfSBgtNn7ws/view?usp=sharing
[5]: https://drive.google.com/file/d/16Rv8rSRneq7gfPRkpFZseNYfswuoqI4-/view?usp=sharing
[6]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[7]: https://github.com/MarkWuNLP/MultiTurnResponseSelection
[8]: https://github.com/cooelf/DeepUtteranceAggregation

