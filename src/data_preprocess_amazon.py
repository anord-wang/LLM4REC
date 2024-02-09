#!/usr/bin/env python
# coding: utf-8

# # This Notebook Processes the Public Amazon Dataset

# In[27]:


import os
import gzip
import json
import pickle
import random
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.sparse import lil_matrix, save_npz


# In[28]:


'''
    We follow the Multi-VAE paper
    and fix the random seed as 98765
'''
seed = 98765
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# ## 1. Define the Utility Functions

# In[29]:


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def save_text(data, filename):
    with open(filename, "w") as f:
        f.write(data)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

# def parse(path):
#     g = gzip.open(path, 'r')
#     for l in g:
#         yield eval(l)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
        
def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def dict_to_txt(dict_data, layer=1, recursive=True):
    txt = "{\n"
    for k,v in dict_data.items():
        txt += " "*2*layer + "{}:{}\n".format(
            dict_to_txt(k, layer=layer+1, recursive=recursive) if isinstance(k, dict) else k, 
            dict_to_txt(v, layer=layer+1, recursive=recursive) if isinstance(v, dict) else v)
    txt += " "*(layer-1)*2 + "}"
    return txt


# ## 2. Select the Target Dataset

# In[30]:


short_data_name = 'food'
if not os.path.exists(short_data_name):
    os.mkdir(short_data_name)
    
if short_data_name == 'beauty':
    full_data_name = 'Beauty'
elif short_data_name == 'toys':
    full_data_name = 'Toys_and_Games'
elif short_data_name == 'sports':
    full_data_name = 'Sports_and_Outdoors'
elif short_data_name == 'scientific':
    full_data_name = 'Industrial_and_Scientific'
elif short_data_name == 'arts':
    full_data_name = 'Arts_Crafts_and_Sewing'
elif short_data_name == 'instruments':
    full_data_name = 'Musical_Instruments'
elif short_data_name == 'office':
    full_data_name = 'Office_Products'
elif short_data_name == 'pantry':
    full_data_name = 'Prime_Pantry'
elif short_data_name == 'luxury':
    full_data_name = 'Luxury_Beauty'
elif short_data_name == 'music':
    full_data_name = 'Digital_Music'
elif short_data_name == 'garden':
    full_data_name = 'Patio_Lawn_and_Garden'
elif short_data_name == 'food':
    full_data_name = 'Grocery_and_Gourmet_Food'

else:
    raise NotImplementedError


# ## 3. Define Functions to Extract Interaction Information

# In[31]:


def Amazon(dataset_name, rating_score):
    '''
        reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
        asin - (Amazon Standard Identification Number) 
             -  ID of the product, e.g. 0000013714
        reviewerName - name of the reviewer
        helpful - helpfulness rating of the review, e.g. 2/3
            --"helpful": [2, 3],
        reviewText - text of the review
            --"reviewText": "I bought this for my husband who plays the piano. ..."
        overall - rating of the product
            --"overall": 5.0,
        summary - summary of the review
            --"summary": "Heavenly Highway Hymns",
        unixReviewTime - time of the review (unix time)
            --"unixReviewTime": 1252800000,
        reviewTime - time of the review (raw)
            --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # data_file = './raw_data/reviews_' + dataset_name + '.json.gz'
    data_file = './raw_data/' + dataset_name + '.json.gz'
    for record in parse(data_file):
        if float(record['overall']) <= rating_score:
            continue
        user = record['reviewerID']
        item = record['asin']
        time = record['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas

def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1]) 
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True 


# filter the K-core
def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items

def id_map(user_items): # user_items dict
    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 0
    item_id = 0
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = user_id
            id2user[user_id] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item[item_id] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


# ## 4. Define Functions to Extract Content Information

# In[32]:


def Amazon_meta(dataset_name, data_maps):
    '''
        asin - ID of the product, e.g. 0000031852
            --"asin": "0000031852",
        title - name of the product
            --"title": "Girls Ballet Tutu Zebra Hot Pink",
        description
        price - price in US dollars (at time of crawl)
            --"price": 3.17,
        imUrl - url of the product image (str)
            --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
        related - related products (also bought, also viewed, bought together, buy after viewing)
            --"related":{
                "also_bought": ["B00JHONN1S"],
                "also_viewed": ["B002BZX8Z6"],
                "bought_together": ["B002BZX8Z6"]
            },
        salesRank - sales rank information
            --"salesRank": {"Toys & Games": 211836}
        brand - brand name
            --"brand": "Coxlures",
        categories - list of categories the product belongs to
            --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    meta_file = './raw_data/meta_' + dataset_name + '.json.gz'
    item_asins = list(data_maps['item2id'].keys())
    for info in parse(meta_file):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas

# categories and brand is all attribute
def get_attr_Amazon(meta_infos, datamaps, attr_core):
    # First, calculate the number of different attributes
    attrs = defaultdict(int)
    for vid, info in tqdm(meta_infos.items()):
        # print(info)
        # for cates in info['categories']:
        for cates in info['category']:
            for cate in cates[1:]:
                attrs[cate] +=1
        try:
            attrs[info['brand']] += 1
        except:
            pass

    # We only save attributes that appear more than attr_core times
    print(f'before delete, attribute num:{len(attrs)}')
    new_meta = {}
    for vid, info in tqdm(meta_infos.items()):
        new_meta[vid] = []
        try:
            if attrs[info['brand']] >= attr_core:
                new_meta[vid].append(info['brand'])
        except:
            pass
        # for cates in info['categories']:
        for cates in info['category']:
            for cate in cates[1:]:
                if attrs[cate] >= attr_core:
                    new_meta[vid].append(cate)
    
    # Save the attribute data
    attr2id = {}
    id2attr = {}
    attrid2num = defaultdict(int)
    attr_id = 1
    items2attrs = {}
    attr_lens = []

    for vid, attrs in new_meta.items():
        item_id = datamaps['item2id'][vid]
        items2attrs[item_id] = []
        for attr in attrs:
            if attr not in attr2id:
                attr2id[attr] = attr_id
                id2attr[attr_id] = attr
                attr_id += 1
            attrid2num[attr2id[attr]] += 1
            items2attrs[item_id].append(attr2id[attr])
        attr_lens.append(len(items2attrs[item_id]))
        
    print(f'before delete, attribute num:{len(attr2id)}')
    print(f'attributes len, Min:{np.min(attr_lens)}, Max:{np.max(attr_lens)}, Avg.:{np.mean(attr_lens):.4f}')
    # update datamap
    datamaps['attr2id'] = attr2id
    datamaps['id2attr'] = id2attr
    datamaps['attrid2num'] = attrid2num
    return len(attr2id), np.mean(attr_lens), datamaps, items2attrs


# ## 5. Processing the Dataset

# In[33]:


def main(data_name, acronym, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp'}
    rating_score = 3.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attr_core = 0

    if data_type == 'Yelp':
        pass
        # date_max = '2019-12-31 00:00:00'
        # date_min = '2019-01-01 00:00:00'
        # datas = Yelp(date_min, date_max, rating_score)
    else:
        datas = Amazon(data_name+'_5', rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    print('Begin extracting meta infos...')

    meta_infos = Amazon_meta(data_name, data_maps)
    attr_num, avg_attr, datamaps, item2attrs = get_attr_Amazon(meta_infos, data_maps, attr_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attr_num)}&'
          f'{avg_attr:.1f} \\')

    return meta_infos, user_items, item2attrs, datamaps


# In[34]:


meta_infos, user_items, item2attrs, datamaps = main(full_data_name, short_data_name, data_type='Amazon')


# ### 5.1. Processing the Interaction Data and Save as Sparse Matrices
# 
# For each user, we use 80% of the historical interactions as the training data, 10% as the validation data, and another 10% as the testing data.   
# Please note that at least one item is used for validation, and at least one another item is used for testing.

# In[35]:


def split_user_item_interactions(interactions):
    # Determine the total number of users and items
    num_users = len(interactions)
    num_items = max([max(items) for items in interactions.values()])+1

    # Create empty dictionaries for training, validation, and testing interactions
    train_interactions = {}
    val_interactions = {}
    test_interactions = {}

    # Iterate over each user
    for user_id, item_list in interactions.items():
        # Shuffle the list of item interactions for each user
        np.random.shuffle(item_list)

        # Calculate the number of interactions for training, validation, and testing
        # We keep at least one validation/testing item for each user
        num_val = max(1, int(len(item_list) * 0.1))
        num_test = num_val
        num_train = int(len(item_list)) - num_val - num_test

        # Split the shuffled item interactions list
        train_items = item_list[:num_train]
        val_items = item_list[num_train:num_train + num_val]
        test_items = item_list[num_train + num_val:num_train + num_val + num_test]

        # Assign the interactions to the corresponding datasets
        train_interactions[user_id] = train_items
        val_interactions[user_id] = val_items
        test_interactions[user_id] = test_items

    # Convert dictionaries into sparse matrices
    train_matrix = dict_to_sparse_matrix(train_interactions, num_users, num_items)
    val_matrix = dict_to_sparse_matrix(val_interactions, num_users, num_items)
    test_matrix = dict_to_sparse_matrix(test_interactions, num_users, num_items)

    return train_matrix, val_matrix, test_matrix


def dict_to_sparse_matrix(dictionary, num_rows, num_cols):
    matrix = lil_matrix((num_rows, num_cols), dtype=np.float32)
    for row, items in dictionary.items():
        for col in items:
            matrix[row, col] = 1.0  
    return matrix.tocsr()


# In[36]:


train_matrix, val_matrix, test_matrix = split_user_item_interactions(user_items)
data_root = os.path.join("dataset", short_data_name)
if not os.path.exists(data_root):
    os.makedirs(data_root)
save_npz(os.path.join(data_root, 'train_matrix.npz'), train_matrix)
save_npz(os.path.join(data_root, 'val_matrix.npz'), val_matrix)
save_npz(os.path.join(data_root, 'test_matrix.npz'), test_matrix)

num_users, num_items = train_matrix.shape
meta_data = {"num_users":num_users, "num_items":num_items}
meta_txt = f"num_users:{num_users}\nnum_items:{num_items}"

meta_path = os.path.join(data_root, "meta.pkl")
meta_txt_path = os.path.join(data_root, "meta.txt")

with open(meta_path, "wb") as file:
    pickle.dump(meta_data, file)
        
with open(meta_txt_path, "w") as file:
    file.write(meta_txt)


# ### 5.2. Processing the Item-Specific Textual Data

# In[37]:


# The interested texts associated with the items
item_texts = ["title", "brand", "categories", "description"]
item_texts = {item_text:[] for item_text in item_texts}

# Process items and append to respective lists
for asin, properties in sorted(meta_infos.items(), key=lambda x: datamaps["item2id"].get(x[0], 0)):
    if asin in datamaps["item2id"]:

        item_id = datamaps["item2id"][asin]
        
        # Obtain the title of item_id
        title = properties.get("title")
        if title:
            item_texts["title"].append([f"The title of item_{item_id} is:", f" {title}"])
        
        # Obtain the brand of item_id
        brand = properties.get("brand")
        if brand:
            item_texts["brand"].append([f"The brand of item_{item_id} is:", f" {brand}"])
        
        # Obtain the categories of item_id
        categories = properties.get("categories")
        if categories:
            categories_text = ", ".join(categories[0])
            item_texts["categories"].append([f"The categories of item_{item_id} are:", f" {categories_text}"])

        # Obtain the description of item_id
        description = properties.get("description")
        if description:
            item_texts["description"].append([f"The description of item_{item_id} is:", f" {description}"])

# Save output lists to files
item_text_root = os.path.join(data_root, "item_texts")
if not os.path.exists(item_text_root):
    os.makedirs(item_text_root)

# Save output lists to files
for name, item_text in item_texts.items():
    # For human to read
    text_filepath = os.path.join(item_text_root, f"{name}.txt")
    # For machine to read
    pkl_filepath = os.path.join(item_text_root, f"{name}.pkl")
    
    with open(text_filepath, "w") as file:
        file.write("\n".join(
            ["".join([prompt, main]) for prompt, main in item_text]
        ))
        
    with open(pkl_filepath, "wb") as file:
         pickle.dump(item_text, file)


# ### 5.3. Processing the Textual Data Associated with a User/Item Pair
# 
# #### 5.3.1 Processing the Review Data of user_i to item_j

# In[38]:


review_data = []
# for review in parse("./raw_data/reviews_{}_5.json.gz".format(full_data_name)):
for review in parse("./raw_data/{}_5.json.gz".format(full_data_name)):
    review_data.append(review)

# Convert review list to dictionary
review_dict = {}
for review in review_data:
    reviewer_id = review['reviewerID']
    # print(reviewer_id)
    asin = review['asin']
    # print(asin)
    if 'reviewText' not in review:
        print('no review')
        print(review.keys())
        review['reviewText'] = 'No Review'
    review_dict[(reviewer_id, asin)] = review['reviewText']

# Traverse the sparse matrix and retrieve review texts efficiently
reviews = []
for user_id, item_id in zip(train_matrix.nonzero()[0], train_matrix.nonzero()[1]):
    reviewer_id = datamaps["id2user"].get(user_id)
    asin = datamaps["id2item"].get(item_id)
    if reviewer_id and asin:
        review_text = review_dict.get((reviewer_id, asin))
        if review_text:
            reviews.append([f"user_{user_id} wrote the following review for item_{item_id}:", f" {review_text}"])
            
# Save output lists to files
user_item_text_root = os.path.join(data_root, "user_item_texts")
if not os.path.exists(user_item_text_root):
    os.makedirs(user_item_text_root)

# Save output lists to files
text_filepath = os.path.join(user_item_text_root, "review.txt")
pkl_filepath = os.path.join(user_item_text_root, "review.pkl")

with open(text_filepath, "w") as file:
    file.write("\n".join(
        ["".join([prompt, main]) for prompt, main in reviews]
    ))

with open(pkl_filepath, "wb") as file:
     pickle.dump(reviews, file)


# #### 5.3.2 Processing the Explanation Data of user_i to item_j

# In[39]:


# # explain_data = load_pickle('./raw_data/reviews_{}.pickle'.format(full_data_name))
# explain_data = load_pickle('./raw_data/reviews_{}.pickle'.format(full_data_name))
#
# # Convert explain list to dictionary
# explain_dict = {}
# for explain in explain_data:
#     reviewer_id = explain['user']
#     asin = explain['item']
#     explain_dict[(reviewer_id, asin)] = explain['text']
#
# # Traverse the sparse matrix and retrieve explain texts efficiently
# explains = []
# for user_id, item_id in zip(train_matrix.nonzero()[0], train_matrix.nonzero()[1]):
#     reviewer_id = datamaps["id2user"].get(user_id)
#     asin = datamaps["id2item"].get(item_id)
#     if reviewer_id and asin:
#         explain_text = explain_dict.get((reviewer_id, asin))
#         if explain_text:
#             explains.append([f"user_{user_id} explains the reason for purchasing item_{item_id}:", f" {explain_text}"])
#
# # Save output lists to files
# text_filepath = os.path.join(user_item_text_root, "explain.txt")
# pkl_filepath = os.path.join(user_item_text_root, "explain.pkl")
#
# with open(text_filepath, "w") as file:
#     file.write("\n".join(
#         ["".join([prompt, main]) for prompt, main in explains]
#     ))
#
# with open(pkl_filepath, "wb") as file:
#      pickle.dump(explains, file)

