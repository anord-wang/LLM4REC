import re
import os
import sys
import pickle
import fsspec
import argparse
from tqdm import tqdm

import numpy as np

import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.distributed as dist
# from torch.utils.data import Dataset
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from accelerate import Accelerator

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model
from transformers.models.gpt2 import GPT2ModelWithBC
# from transformers import GPT2Tokenizer

sys.path.append("libs")
from libs.tokenizer import TokenizerWithUserItemIDTokensBatch

from libs.data import CollaborativeGPTGeneratorBatch
from libs.data import UserItemContentGPTDatasetBatch

from libs.model import GPT4RecommendationBaseModel
from libs.model import CollaborativeGPTwithItemLMHeadBatch
from libs.model import ContentGPTForUserItemWithLMHeadBatch


def save_local(remote_path, local_path, remote_mode, local_mode):
    '''
        Save the remote file in remote_path
        to the local_path...
    '''
    with fsspec.open(remote_path, remote_mode) as f:
        content = f.read()
    with fsspec.open(local_path, local_mode) as f:
        f.write(content)


def save_remote(local_path, remote_path, local_mode, remote_mode):
    '''
        Save the local file in local_path
        to the remote_path...
    '''
    with fsspec.open(local_path, local_mode) as f:
        content = f.read()
    with fsspec.open(remote_path, remote_mode) as f:
        f.write(content)

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 改路径
# server_root = "hdfs://llm4rec"
server_root = "/home/local/ASURITE/xwang735/LLM4REC/LLM4Rec"
# server_root = "/home/wxy/LLM4Rec"
gpt2_server_root = server_root
# local_root = "tmp"
local_root = '/home/local/ASURITE/xwang735/LLM4REC/sports_model_save'
# local_root = '/home/wxy/LLM4Rec/model_save/toys'
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if not os.path.exists(local_root):
    os.makedirs(local_root, exist_ok=True)

_config = {
    "activation_function": "gelu_new",
    "architectures": [
        "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
        "text-generation": {
            "do_sample": True,
            "max_length": 50
        }
    },
    "vocab_size": 50257
}


def main():
    # Define the accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str,
                        help="specify the dataset for experiment")
    args = parser.parse_args()

    dataset = args.dataset
    lambda_V = float(args.lambda_V)

    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {dataset}")
    accelerator.print(f"lambda_V: {args.lambda_V}")

    # Define the number of GPUs to be used
    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")

    '''
        Get the basic information of the dataset
    '''
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(gpt2_server_root, "dataset", dataset)
    meta_path = os.path.join(data_root, "meta.pkl")

    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)

    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    accelerator.print(f"num_users: {num_users}")
    accelerator.print(f"num_items: {num_items}")
    accelerator.print("-----End Obtaining Dataset Info-----\n")

    '''
        Obtain the tokenizer with user/item tokens
    '''
    accelerator.print("-----Begin Obtaining the Tokenizer-----")
    tokenizer_root = os.path.join(gpt2_server_root, "model", "pretrained", "tokenizer")
    accelerator.print(f"Loading pretrained tokenizer from {tokenizer_root}...")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 内容文件缺失
    remote_vocab_file = os.path.join(tokenizer_root, "vocab_file.json")
    remote_merges_file = os.path.join(tokenizer_root, "merges.txt")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    vocab_file = os.path.join(local_root, "vocab_file.json")
    merges_file = os.path.join(local_root, "merges.txt")

    if accelerator.is_main_process:
        save_local(remote_vocab_file, vocab_file, "r", "w")
        save_local(remote_merges_file, merges_file, "r", "w")
    accelerator.wait_for_everyone()

    tokenizer = TokenizerWithUserItemIDTokensBatch(vocab_file,
                                                   merges_file,
                                                   num_users,
                                                   num_items)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    mapping_graph_bc = []
    # mapping_graph_bc = np.ones((num_users + num_items, num_users + num_items))
    # remote_mapping_graph_bc_path = os.path.join(data_root, "interaction_matrix.npz")
    # local_mapping_graph_bc_path = os.path.join(local_root, "interaction_matrix.npz")
    # if accelerator.is_main_process:
    #     save_local(remote_mapping_graph_bc_path, local_mapping_graph_bc_path, "rb", "wb")
    # accelerator.wait_for_everyone()
    #
    # mapping_graph_bc = load_npz(local_mapping_graph_bc_path)
    # mapping_graph_bc = np.load(local_mapping_graph_bc_path)
    # mapping_graph_bc = mapping_graph_bc['arr_0']
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    '''
        Define the review data generator
    '''
    accelerator.print("-----Begin Obtaining the Review Data Generator-----")
    filepath_list = [os.path.join(data_root, "user_item_texts", "review.pkl"),
                     # os.path.join(data_root, "user_item_texts", "explain.pkl"),
                     os.path.join(data_root, "item_texts", "title.pkl"),
                     os.path.join(data_root, "item_texts", "brand.pkl"),
                     os.path.join(data_root, "item_texts", "categories.pkl"),
                     os.path.join(data_root, "item_texts", "description.pkl"),
                     os.path.join(data_root, "item_texts", "brand_extension.pkl"),
                     os.path.join(data_root, "item_texts", "categories_extension.pkl"),
                     ]
    review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
    accelerator.print(f"Loading data from {review_path}...")
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, filepath_list, mapping_graph_bc)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Review Data Generator-----\n")

    '''
        Now we deal with the user/item interaction data
    '''
    accelerator.print("-----Begin Obtaining the Collaborative Data Generator-----")
    remote_train_mat_path = os.path.join(data_root, "train_matrix.npz")
    accelerator.print(f"Loading data from {remote_train_mat_path}...")
    local_train_mat_path = os.path.join(local_root, "train_matrix.npz")
    if accelerator.is_main_process:
        save_local(remote_train_mat_path, local_train_mat_path, "rb", "wb")
    accelerator.wait_for_everyone()

    train_mat = load_npz(local_train_mat_path)
    collaborative_data_gen = CollaborativeGPTGeneratorBatch(tokenizer, train_mat, mapping_graph_bc)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Collaborative Data Generator-----\n")

    '''
        Extend the config of the original GPT model
    '''
    accelerator.print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items
    accelerator.print("Success!")
    accelerator.print("-----End Setting Up the Config-----\n")

    '''
        Instantiate the pretrained GPT2 model
    '''
    accelerator.print("-----Begin Instantiating the Pretrained GPT Model-----")
    # gpt2model = GPT2Model(config)
    # =================================================================================================================================================================
    accelerator.print("-----Pretrained GPT Model With Graph Edge Information-----")
    gpt2model = GPT2ModelWithBC(config)
    # =================================================================================================================================================================
    pretrained_root = os.path.join(gpt2_server_root, "model", "pretrained")
    accelerator.print(f"Loading pretrained weights from {pretrained_root}...")
    remote_pretrained_weights_path = os.path.join(pretrained_root, "gpt2", "pytorch_model.bin")
    local_pretrained_weights_path = os.path.join(local_root, "gpt2", "pytorch_model.bin")
    if accelerator.is_main_process:
        save_local(remote_pretrained_weights_path, local_pretrained_weights_path, "rb", "wb")
    accelerator.wait_for_everyone()
    gpt2model.load_state_dict(torch.load(local_pretrained_weights_path), strict=False)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pretrained GPT Model-----\n")

    '''
        Instantiate the GPT for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Content GPT Model-----")
    content_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    # gpt2model_content = GPT2ModelWithBC(config)
    # local_pretrained_weights_path_content = os.path.join(server_root, "model", "pretrained", "gpt2", "allgpt_gpt2_content_yelp_1_0.bin")
    # gpt2model_content.load_state_dict(torch.load(local_pretrained_weights_path_content), strict=False)
    # content_base_model = GPT4RecommendationBaseModel(config, gpt2model_content)
    # local_pretrained_user_emb_path_content = os.path.join(server_root, "model", dataset+'_all_gpt', "content", f"user_embeddings_{args.lambda_V}_0.pt")
    # local_pretrained_item_emb_path_content = os.path.join(server_root, "model", dataset+'_all_gpt', "content", f"item_embeddings_{args.lambda_V}_0.pt")
    # content_base_model.user_embeddings.load_state_dict(
    #     torch.load(local_pretrained_user_emb_path_content, map_location=device))
    # content_base_model.item_embeddings.load_state_dict(
    #     torch.load(local_pretrained_item_emb_path_content, map_location=device))
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Content GPT Model-----\n")

    '''
        Freeze the parameters of the pretrained GPT2 for content model
    '''
    # for name, param in content_model.named_parameters():
    #     # we allow only user/item token embeddings to be trained
    #     if ('user_embeddings' not in name) and \
    #             ('item_embeddings' not in name):
    #         param.requires_grad = False

    accelerator.print("-----Trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    accelerator.print("\n-----Non-trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    '''
        Instantiate the GPT for recommendation collaborative model
    '''
    accelerator.print("-----Begin Instantiating the Collaborative GPT Model-----")
    collaborative_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    collaborative_model = CollaborativeGPTwithItemLMHeadBatch(config, collaborative_base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Collaborative GPT Model-----\n")

    '''
        Freeze the parameters of the pretrained GPT2 for collaborative model
    '''
    # for name, param in collaborative_model.named_parameters():
    #     # we allow only user/item token embeddings to be trained
    #     if ('user_embeddings' not in name) and \
    #             ('item_embeddings' not in name):
    #         param.requires_grad = False

    accelerator.print("-----Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))

    accelerator.print("\n-----Non-Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    '''
        Set up the training details
    '''
    accelerator.print("-----Begin Setting Up the Training Details-----")
    learning_rate = 1e-3
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # batch_size = 20
    batch_size_review = 4
    batch_size_collaborative = 32
    num_pretrained_epochs = 3
    # num_pretrained_epochs = 0
    num_epochs = 100
    # num_epochs = 5
    num_workers = 16
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    '''
        Create a data sampler for distributed training
    '''
    accelerator.print("-----Begin Creating the DataLoader-----")
    # Create the review data loader with the custom collate_fn
    review_data_loader = DataLoader(review_data_gen,
                                    batch_size=batch_size_review,
                                    shuffle=True,
                                    collate_fn=review_data_gen.collate_fn,
                                    num_workers=num_workers)

    # Create the collaborative data loader with the custon collate_fn
    collaborative_data_loader = DataLoader(collaborative_data_gen,
                                           batch_size=batch_size_collaborative,
                                           collate_fn=collaborative_data_gen.collate_fn,
                                           num_workers=num_workers)
    accelerator.print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    content_model.train()
    content_model.to(device)

    collaborative_model.train()
    collaborative_model.to(device)

    # Obtain the optimizer
    review_optimizer = optim.Adam(content_model.parameters(),
                                  lr=learning_rate)

    collaborative_optimizer = optim.Adam(collaborative_model.parameters(),
                                         lr=learning_rate)

    # Parallel model, optimizer and data loader with accelerator
    content_model, review_optimizer, review_data_loader = accelerator.prepare(
        content_model, review_optimizer, review_data_loader
    )

    # Parallel model, optimizer and data loader with accelerator
    collaborative_model, collaborative_optimizer, collaborative_data_loader = accelerator.prepare(
        collaborative_model, collaborative_optimizer, collaborative_data_loader
    )

    # Initialize best_loss with infinity
    review_best_loss = float('inf')
    collaborative_best_loss = float('inf')

    # The place to save the content model weights
    content_model_root = os.path.join(server_root, "model", dataset+'_all_gpt', "content")
    accelerator.print(f"Weights will be saved to {content_model_root}!")

    # The place to save the collaborative model weights
    collaborative_model_root = os.path.join(server_root, "model", dataset+'_all_gpt', "collaborative")
    accelerator.print(f"Weights will be saved to {collaborative_model_root}!")

    accelerator.print("-----End Setting Up the Training Details-----\n")

    '''
        Define the pretraining loop for the content GPT
    '''
    accelerator.print("-----Begin Content GPT Pretraining Loop-----")
    for epoch in range(num_pretrained_epochs):
        review_total_loss = 0
        print(f'Epoch {epoch + 1}/{num_pretrained_epochs}')
        progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}",
                            disable=not accelerator.is_local_main_process)

        # Initialize tqdm progress bar
        # progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}", 
        #                     disable=not accelerator.is_local_main_process, ncols=80)
        for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in progress_bar:
            # for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in review_data_loader:
            review_optimizer.zero_grad()

            # Obtain the data
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)
            graph_bc_prompt = graph_bc_prompt.to(device)
            graph_bc_combined = graph_bc_combined.to(device)

            # Forward pass
            outputs = content_model(input_ids_prompt,
                                    input_ids_main,
                                    mapping_graph_bc_prompt=graph_bc_prompt,
                                    mapping_graph_bc_combined=graph_bc_combined,
                                    labels_main=input_ids_main,
                                    attention_mask=attention_mask)
            review_loss = outputs[0]

            # Backward pass and optimization
            accelerator.backward(review_loss)
            review_optimizer.step()

            review_total_loss += review_loss.item()
            progress_bar.set_postfix({"Review Loss": review_loss.item()})

        thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
        gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
        review_average_loss = torch.mean(gathered_review_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")

        gpt_save_path = os.path.join(pretrained_root, "gpt2",
                                     f"allgpt_gpt2_content_{args.dataset}_{args.lambda_V}_{review_average_loss}_{epoch}.bin")
        torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)

        # Check if the current loss is better than the best_loss
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings in the main process 
            user_emb_local_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
            user_emb_remote_path = os.path.join(content_model_root, f"user_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
            # gpt_save_path = os.path.join(pretrained_root, "gpt2", f"gpt2_content_{args.dataset}_{args.lambda_V}_{review_average_loss}_{epoch}.bin")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(content_model).base_model.user_embeddings.state_dict(),
                           user_emb_local_path)
                # torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)
                save_remote(user_emb_local_path, user_emb_remote_path, "rb", "wb")

            # Save item embeddings in the main process
            item_emb_local_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
            item_emb_remote_path = os.path.join(content_model_root, f"item_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(content_model).base_model.item_embeddings.state_dict(),
                           item_emb_local_path)
                save_remote(item_emb_local_path, item_emb_remote_path, "rb", "wb")
    accelerator.print("-----End Content GPT Pretraining Loop-----")

    '''
        Iteratively training the collaborative and content GPT model for recommendations
    '''
    accelerator.print("-----Begin the Iterative Training Loop-----")
    for epoch in range(num_epochs):
        '''
            Optimize the collaborative GPT model
        '''
        print('num_epochs: ', epoch)
        collaborative_total_loss = 0
        regularize_total_loss = 0

        progress_bar = tqdm(collaborative_data_loader, desc=f"Epoch {epoch + 1}",
                            disable=not accelerator.is_local_main_process, ncols=100)
        for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in progress_bar:

            # for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in collaborative_data_loader:
            collaborative_optimizer.zero_grad()

            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)
            graph_bc_prompt = graph_bc_prompt.to(device)
            graph_bc_combined = graph_bc_combined.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                content_embeds = torch.cat(
                    (accelerator.unwrap_model(content_model).base_model.embed(input_ids_prompt),
                     accelerator.unwrap_model(content_model).base_model.embed(input_ids_main)),
                    axis=1
                ).to(device)

            # Forward pass of the collaborative GPT
            outputs = collaborative_model(input_ids_prompt,
                                          input_ids_main,
                                          mapping_graph_bc_prompt=graph_bc_prompt,
                                          mapping_graph_bc_combined=graph_bc_combined,
                                          labels_main=input_ids_main,
                                          attention_mask=attention_mask,
                                          regularize=True,
                                          lambda_V=lambda_V,
                                          content_embeds=content_embeds)
            collaborative_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(collaborative_loss)
            collaborative_optimizer.step()

            collaborative_total_loss += collaborative_loss.item()
            regularize_total_loss += regularize_loss.item()

            # progress_bar.set_postfix({"Collaborative Loss": collaborative_loss.item(),
            #                           "Regularize Loss": regularize_loss.item()})

        # Gather the collaborative LM loss from different device
        thread_collaborative_average_loss = torch.tensor(
            [collaborative_total_loss / len(collaborative_data_loader)]).to(device)
        gathered_collaborative_average_loss = accelerator.gather(thread_collaborative_average_loss)
        collaborative_average_loss = torch.mean(gathered_collaborative_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Collaborative Loss: {collaborative_average_loss:.4f}")

        # Gather the regularize loss from difference device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(collaborative_data_loader)]).to(
            device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        if collaborative_average_loss < collaborative_best_loss:
            collaborative_best_loss = collaborative_average_loss

            # Save user embeddings in the main process
            user_emb_local_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}_{collaborative_average_loss}_{epoch}.pt")
            user_emb_remote_path = os.path.join(collaborative_model_root, f"user_embeddings_{args.lambda_V}_{collaborative_average_loss}_{epoch}.pt")
            gpt_save_path = os.path.join(pretrained_root, "gpt2",
                                         f"allgpt_gpt2_collaborative_{args.dataset}_{args.lambda_V}_{collaborative_average_loss}_{epoch}.bin")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(collaborative_model). \
                           base_model.user_embeddings.state_dict(),
                           user_emb_local_path)
                save_remote(user_emb_local_path, user_emb_remote_path, "rb", "wb")
                torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)

            # Save item embeddings in the main process
            item_emb_local_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}_{collaborative_average_loss}_{epoch}.pt")
            item_emb_remote_path = os.path.join(collaborative_model_root, f"item_embeddings_{args.lambda_V}_{collaborative_average_loss}_{epoch}.pt")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(collaborative_model). \
                           base_model.item_embeddings.state_dict(),
                           item_emb_local_path)
                save_remote(item_emb_local_path, item_emb_remote_path, "rb", "wb")

        '''
            Optimize the content GPT model
        '''

        if (epoch + 1) % 50 == 0:
            review_total_loss = 0
            regularize_total_loss = 0

            progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}",
                                disable=not accelerator.is_local_main_process, ncols=100)
            # for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in review_data_loader:
            for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in progress_bar:
                review_optimizer.zero_grad()

                input_ids_prompt = input_ids_prompt.to(device)
                input_ids_main = input_ids_main.to(device)
                attention_mask = attention_mask.to(device)
                graph_bc_prompt = graph_bc_prompt.to(device)
                graph_bc_combined = graph_bc_combined.to(device)

                accelerator.wait_for_everyone()
                with torch.no_grad():
                    collaborative_embeds = accelerator.unwrap_model(collaborative_model). \
                        base_model.embed(input_ids_prompt).to(device)

                # Forward pass of the content GPT
                outputs = content_model(input_ids_prompt,
                                        input_ids_main,
                                        mapping_graph_bc_prompt=graph_bc_prompt,
                                        mapping_graph_bc_combined=graph_bc_combined,
                                        labels_main=input_ids_main,
                                        attention_mask=attention_mask,
                                        regularize=True,
                                        lambda_V=lambda_V,
                                        collaborative_embeds=collaborative_embeds)
                review_loss = outputs[0]
                regularize_loss = outputs[1]

                # Backward pass and optimization
                accelerator.backward(review_loss)
                review_optimizer.step()

                review_total_loss += review_loss.item()
                regularize_total_loss += regularize_loss.item()
                # progress_bar.set_postfix({"Review Loss": review_loss.item(),
                #                           "Regularize Loss": regularize_loss.item()})

            # Gather the content LM loss from different device
            thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
            gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
            review_average_loss = torch.mean(gathered_review_average_loss)
            accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")

            # Gather the regularize loss from different device
            thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(review_data_loader)]).to(device)
            gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
            regularize_average_loss = torch.mean(gathered_regularize_average_loss)
            accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

            # Check if the current loss is better than the best_loss
            accelerator.wait_for_everyone()
            if review_average_loss < review_best_loss:
                review_best_loss = review_average_loss

                # Save user embeddings in the main process
                user_emb_local_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
                user_emb_remote_path = os.path.join(content_model_root, f"user_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
                gpt_save_path = os.path.join(pretrained_root, "gpt2", f"allgpt_gpt2_content_{args.dataset}_{args.lambda_V}_{review_average_loss}_{epoch}.bin")
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(content_model).base_model.user_embeddings.state_dict(),
                               user_emb_local_path)
                    save_remote(user_emb_local_path, user_emb_remote_path, "rb", "wb")
                    torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)

                # Save item embeddings in the main process
                item_emb_local_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
                item_emb_remote_path = os.path.join(content_model_root, f"item_embeddings_{args.lambda_V}_{review_average_loss}_{epoch}.pt")
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(content_model).base_model.item_embeddings.state_dict(),
                               item_emb_local_path)
                    save_remote(item_emb_local_path, item_emb_remote_path, "rb", "wb")


if __name__ == "__main__":
    main()
