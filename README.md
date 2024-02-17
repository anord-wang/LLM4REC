# LLM4REC

This ReadMe file contains the Python codes for the [paper](https://arxiv.org/abs/2402.09617).

# 1. Task and Solution
Our task is to use large language models to recommend.
The current methods cannot integrate edge information in graphs into LLMs structurally. Our solution contains two major parts. First, we add an edge measurement in attention calculation. Second, we design a set of prompts for pre-training and fine-tuning. The details can be found in the [paper](https://arxiv.org/abs/2402.09617).

# 2. Dataset
We use the Amazon Review Dataset for experiments. The raw data can be found [here](https://nijianmo.github.io/amazon/index.html).

# 3. Codes Description
There are two parts of the code. The first part is the modified Attention code. The second part is the progress of the proposed method.

## 3.1. Modified Attention Codes
The modified Attention code is in the [folder](modified_transformer/). You can put it in the Transformers lib and the path to those two codes may be like this:

```bash
'/home/local/ASURITE/xwang735/anaconda3/envs/LLM/lib/python3.12/site-packages/transformers/models/gpt2'
```

Or you can just create a new lib containing these codes and name it 'newTransformers'.

## 3.2. Main Codes
There are data preprocessing, pre-training, fine-tuning, and prediction codes in the [src](src/). 

### 3.2.1. Data Pre-processing Codes
First, the data preprocessing codes contain [data_preprocess_amazon.py](src/data_preprocess_amazon.py), [data_preprocessing.py](src/data_preprocessing.py), and [data_pkl.py](src/data_pkl.py).

data_preprocess_amazon.py is used to transform raw data to the format we want. The processed data can be found at this [link]().


### 3.2.2. Useful Components
These codes are in [libs](src/libs/). These codes are built for the dataloader, personalized models, and tokenizer.

### 3.2.3. Codes to Run
These codes are used for pre-training and fine-tuning.


