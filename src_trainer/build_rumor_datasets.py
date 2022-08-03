"""Read, split and save the  dataset for our model"""

import csv
from email.policy import default
import os
import sys
import emoji as emoji
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import re
import string
import os.path
import re
import jieba
from transformers import RobertaTokenizer, BertTokenizer
import argparse
import json
import pickle

DATASET_PATH_MAP = {
    'Twitter': '../data/tweet/mediaeval2015/',
    'Weibo' : '../data/weibo/',
}


def get_tokenizer(name):
    if name == 'Weibo': return BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    elif name == 'Twitter': return RobertaTokenizer.from_pretrained("roberta-base")

def return_first_image(row):
    return row['image_id'].split(',')[0].strip()

def max_length(lines):
    return max([len(s) for s in lines])

def clean_str_sst(string): 
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()

def stopwordslist(filepath):
    stopwords = {}
    for line in open(filepath, 'r').readlines():
        line = bytes(line, "utf-8").strip()  # unicode
        stopwords[line] = 1
    return stopwords

# read image from data path & transforms for image modal
def read_image(file_list):
    image_list = {}
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                # im = 1
                image_list[filename.split('/')[-1].split(".")[0]] = im
            except:
                pass
                # print(filename)
    # print("image length " + str(len(image_list)))
    return image_list

#   read_post for weibo data
def read_post(pre_path,flag):
    stop_words_path = os.path.join(pre_path ,'stop_words.txt')
    stop_words = stopwordslist(stop_words_path)
    file_list = [pre_path + "tweets/test_nonrumor.txt", pre_path + "tweets/test_rumor.txt", \
                    pre_path + "tweets/train_nonrumor.txt", pre_path + "tweets/train_rumor.txt"]
    if flag == "train":
        id = pickle.load(open(pre_path + "train_id.pickle", 'rb'))
    elif flag == "validate":
        id = pickle.load(open(pre_path + "validate_id.pickle", 'rb')) #load 反序列化
    elif flag == "test":
        id = pickle.load(open(pre_path + "test_id.pickle", 'rb'))

    post_content = []
    data = []
    column = ['post_id', 'image_id', 'original_post', 'post_text', 'label']
    for k, f in enumerate(file_list):

        f = open(f, 'rb')
        if (k + 1) % 2 == 1:
            label = 0  ### real is 0
        else:
            label = 1  ####fake is 1

        twitter_id = 0
        line_data = []
        for i, l in enumerate(f.readlines()):
    

            if (i + 1) % 3 == 1:
                line_data = []
                twitter_id = l.decode("utf-8").split('|')[0]
                line_data.append(twitter_id) #处理编号、来源

            if (i + 1) % 3 == 2:
                line_data.append(l.lower()) #处理链接

            if (i + 1) % 3 == 0:
                l = clean_str_sst(str(l, "utf-8"))  # unicode 处理文本内容

                seg_list = jieba.cut_for_search(l)
                new_seg_list = []
                for word in seg_list:
                    if word not in stop_words:
                        new_seg_list.append(word)  #文本内容重新整理成句子

                clean_l = " ".join(new_seg_list) #添加空格
                if len(clean_l) > 10 and line_data[0] in id:
                    post_content.append(l)
                    line_data.append(l)
                    line_data.append(clean_l)
                    line_data.append(label)
                    data.append(line_data)
        f.close()

    data_df = pd.DataFrame(np.array(data), columns=column)
    #write_txt(top_data)

    return post_content, data_df

#  struct image - text pair
def paired(flag, post, image):  
    ordered_image = []
    ordered_post = []
    label = []
    post_id = []
    image_id_list = []

    image_id = ""
    for i, id in enumerate(post['post_id']):
        for image_id in post.iloc[i]['image_id'].split('|'):
            image_id = image_id.split("/")[-1].split(".")[0]
            if image_id in image:
                break

        if image_id in image:
            image_name = image_id
            image_id_list.append(image_name)
            ordered_image.append(image[image_name])
            ordered_post.append(post.iloc[i]['post_text'])
            post_id.append(id)

            label.append(post.iloc[i]['label'])
    label = np.array(label, dtype=int)

    data = {"post_text": np.array(ordered_post),
            "image": ordered_image, "label": np.array(label),
            "post_id": np.array(post_id),
            "image_id": image_id_list}

    # print("{} data size is ".format(flag) + str(len(data["post_text"])))

    return data

def convert_text_to_token(tokenizer, sentence, seq_length):
    tokens = tokenizer.tokenize(sentence) # 句子转换成token
    tokens = ["[CLS]"] + tokens + ["[SEP]"] # token前后分别加上[CLS]和[SEP]

    # 生成 input_id, seg_id, att_mask
    ids1 = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * len(ids1)
    masks = [1] * len(ids1)
    # 句子长度统一化处理：截断或补全至seq_length
    if len(ids1) < seq_length: #补全
        ids = ids1 + [0] * (seq_length - len(ids1)) #[0]是因为词表中PAD的索引是0
        types = types + [1] * (seq_length - len(ids1))  # [1]表明该部分为PAD
        masks = masks + [0] * (seq_length - len(ids1)) # PAD部分，attention mask置为[0]
    else: # 截断
        ids = ids1[:seq_length]
        types = types[:seq_length]
        masks = masks[:seq_length]
    assert len(ids) == len(types) == len(masks)
    return ids, types, masks

def data_add_id_mask(tokenizer,data, max_l):
    ids = []
    types_pool = []
    masks_pool = []
    # count = 0

    for each in data['post_text']:
        cur_ids, cur_type, cur_mask = convert_text_to_token(tokenizer, each, seq_length=max_l)
        ids.append(cur_ids)
        types_pool.append(cur_type)
        masks_pool.append(cur_mask)

        # count += 1
        # if count % 1000 == 0:
        #     print('已处理{}条'.format(count))
            

    data['post_text'] = ids
    data['mask'] = masks_pool
    data['types'] = types_pool

    return data


def load_dataset(path, mode, flag):
    """Loads dataset into memory from  file"""
    if mode == 'Weibo':
        # path = ../data/weibo/
        images_list = [path +'nonrumor_images/', path + 'rumor_images/']
        images_dict = read_image(images_list)
        post_content, post = read_post(path,flag)

    
        data = paired(flag, post, images_dict)

    elif mode == 'Twitter':
        text_path = os.path.join(path,'tweets.txt')
        image_dir = os.path.join(path, flag + '_images/')

        
        data_df = pd.read_csv(text_path,sep = '\t')
        data_df['first_imageid'] = data_df.apply(lambda row: return_first_image(row), axis=1)

        # del not_available images & text pair
        images_dataset = [i for i in data_df['first_imageid'].tolist()]
        images_folder = [i.split('.')[0].strip() for i in os.listdir(image_dir)]
        images_not_available = set(images_dataset) - set(images_folder)
        data_df = data_df[~data_df['first_imageid'].isin(images_not_available)]


        # encode for label
        Y = data_df['label'].tolist()
        Y = [1 if i == 'fake' else 0 for i in Y]

        
        images_list = [image_dir]
        images_dict = read_image(images_list)

        data_df['label'] = Y
        data_df = data_df.drop(['userId', 'username', 'timestamp'], axis=1)


        data = paired(flag, post=data_df, image=images_dict)

    return data


def load_data(args):
    tokenizer = get_tokenizer(args.datasets)



    if args.datasets == 'Twitter':
        train_df_path = os.path.join(DATASET_PATH_MAP[args.datasets],'train/')
        msg = "{} file not found. Make sure train dataset exits ".format(train_df_path)
        assert os.path.exists(train_df_path), msg

        test_df_path = os.path.join(DATASET_PATH_MAP[args.datasets],'test/')
        msg = "{} file not found. Make sure test dataset exits ".format(test_df_path)
        assert os.path.exists(test_df_path), msg

    else: # 'Weibo'
        train_df_path = DATASET_PATH_MAP[args.datasets]
        msg = "{} file not found. Make sure train dataset exits ".format(train_df_path)
        assert os.path.exists(train_df_path), msg

        test_df_path = DATASET_PATH_MAP[args.datasets]
        msg = "{} file not found. Make sure test dataset exits ".format(test_df_path)
        assert os.path.exists(test_df_path), msg
    if args.mode == 'test':
        test_dataset = load_dataset(test_df_path, args.datasets, flag = 'test')
        test_dataset = data_add_id_mask(tokenizer,test_dataset,max_l = args.max_seqlen)
        return None,test_dataset
    # load & preprocess 
    train_dataset = load_dataset(train_df_path, args.datasets, flag = 'train')
    test_dataset = load_dataset(test_df_path, args.datasets, flag = 'test')

    # add mask for text modal
    train_dataset = data_add_id_mask(tokenizer,train_dataset,max_l = args.max_seqlen)
    test_dataset = data_add_id_mask(tokenizer,test_dataset,max_l = args.max_seqlen)   

    return train_dataset, test_dataset



   