import os
import pickle
import re

# 设定 brand.pkl 和 categories.pkl 文件的路径
# dataset = 'beauty'
# dataset = 'sports'
# dataset = 'toys'
# dataset = 'yelp'
# dataset = 'scientific'
# dataset = 'arts'
# dataset = 'instruments'
# dataset = 'office'
# dataset = 'pantry'
# dataset = 'music'
# dataset = 'garden'
dataset = 'food'

server_root = "/home/local/ASURITE/xwang735/LLM4REC/LLM4Rec"
data_root = os.path.join(server_root, "dataset", dataset)
item_text_root = os.path.join(data_root, "item_texts")
brand_pkl_path = os.path.join(item_text_root, "brand.pkl")
categories_pkl_path = os.path.join(item_text_root, "categories.pkl")

# 从 brand.pkl 文件中加载数据
with open(brand_pkl_path, 'rb') as file:
    brand_data = pickle.load(file)

# 从 categories.pkl 文件中加载数据
with open(categories_pkl_path, 'rb') as file:
    categories_data = pickle.load(file)

# 处理加载的数据
# 对于 brand_data，可能需要根据你的具体需求来处理数据
# 例如，如果你想以品牌为基础来创建一个新的列表
brand_based_texts = {}
for text in brand_data:
    brand = text[1].strip()  # 假设品牌名称在每个条目的第二个元素
    user = text[0]
    match = re.search(r'item_\d+', text[0])
    if match:
        extracted_item_id = match.group()
        if brand not in brand_based_texts:
            brand_based_texts[brand] = []
        brand_based_texts[brand].append(extracted_item_id)

# 对于 categories_data，你可能需要将每个类别分开处理
category_based_texts = {}
for text in categories_data:
    categories = text[1].split(", ")
    # print('categories', categories)
    user = text[0]
    match = re.search(r'item_\d+', text[0])
    if match:
        extracted_item_id = match.group()
        for category in categories:
            if category.lower() != "beauty":  # 排除特定类别
                if category not in category_based_texts:
                    category_based_texts[category] = []
                category_based_texts[category].append(extracted_item_id)

# 保存处理后的数据到新的文件中
new_item_text_root = os.path.join(data_root, "new_item_texts")
if not os.path.exists(new_item_text_root):
    os.makedirs(new_item_text_root)

item_texts = ["brand_extension", "categories_extension"]
item_texts = {item_text: [] for item_text in item_texts}

# 保存 brand_based_texts 和 category_based_texts
for category, texts in [("brand", brand_based_texts), ("categories", category_based_texts)]:
    # print('category', category)
    # print('texts', texts)
    if category == "brand":
        for brand, item_ids in texts.items():
            # print(f"Category: {brand}")
            combined_text = ", ".join(item_ids)
            item_texts["brand_extension"].append([f"These item {combined_text} has the same brand:", f" {brand}"])
            # print(f"These item has the same brand {brand}:", f"{combined_text}")
    if category == "categories":
        for categories, item_ids in texts.items():
            # print(f"Category: {categories}")
            combined_text = ", ".join(item_ids)
            item_texts["categories_extension"].append([f"These items {combined_text} are all in the category:", f" {categories}"])
            # print(f"These items are all in the {categories} category:", f"{combined_text}")

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
