# LLM4REC

This ReadMe file contains the Python codes for the [paper](https://arxiv.org/abs/2402.09617).

Our task is to use large language models to recommend.
The current methods cannot integrate edge information in graphs into LLMs structurally. Our solution contains two major parts. First, we add an edge measurement in attention calculation. Second, we design a set of prompts for pre-training and fine-tuning. The details can be found in the [paper](https://arxiv.org/abs/2402.09617).

We use the Amazon Review Dataset for experiments. The raw data can be found [here](https://nijianmo.github.io/amazon/index.html).

There are two parts of the code. The first part is the modified Attention code. The second part is the progress of the proposed method.

The modified Attention code is in the [folder](modified_transformer/). You can put it in the Transformers lib and the path to those two codes may be like this:'/home/local/ASURITE/xwang735/anaconda3/envs/LLM/lib/python3.12/site-packages/transformers/models/gpt2'. Or you can just create a new lib containing these codes and name it 'newTransformers'.

There are data preprocessing, pre-training, fine-tuning, and prediction codes in the src folder. 
First, the data preprocessing codes contain .py .py, and .py. The processed data can be found at this [link]().
https://github.com/anord-wang/LLM4REC/tree/main/modified_transformer
