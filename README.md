# LLM4REC

This ReadMe file contains the Python codes for the [paper]: https://arxiv.org/abs/2402.09617.

Our task is to use large language models to recommend.
The current methods cannot integrate edge information in graphs into LLMs structurally. Our solution contains two major parts. First, we add an edge measurement in attention calculation. Second, we design a set of prompts for pre-training and fine-tuning. The details can be found in the paper.

We use the Amazon recommendation data for experiments. The raw data can be found at this [link]: . The processed data can be found at this [link]: .

There are two parts of the code. The first part is the modified Attention code. The second part is the progress of the proposed method.

The modified Attention code is in the [folder]: (src/libs/). You can put it in the Transformers lib or create a new lib containing these codes, and name it 'newTransformers'.

There are data preprocessing, pre-training, fine-tuning, and prediction codes in the src folder. 
First, the data preprocessing codes contain .py .py, and .py. 

