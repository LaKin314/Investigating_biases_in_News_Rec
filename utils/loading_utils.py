import os
import sys

if "../recommenders" not in sys.path:
    sys.path.insert(0,"../recommenders")

import numpy as np
import random
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from recommenders.datasets.mind import download_mind,read_clickhistory,extract_mind

def download_mind_large(download = False):
    MIND_type = 'large'
    mind_types = ['demo', 'small', 'large']
    modes = ['train', 'valid']

    if download:
        train_path,valid_path = download_mind(size='large', dest_path='Dataset_large')
        print(extract_mind(train_path,
        valid_path,
        train_folder="train",
        valid_folder="valid",
        clean_zip_file=True))

    valid_clickhistory_large = read_clickhistory(path=f'MINDlarge_train.zip/valid',
                                        filename='behaviors.tsv')

    valid_clickhistory_small = read_clickhistory(path=f'Dataset_small/valid',
                                        filename='behaviors.tsv')
    train_clickhistory_small = read_clickhistory(path=f'Dataset_small/train',
                                        filename='behaviors.tsv')

    return valid_clickhistory_large,valid_clickhistory_small,train_clickhistory_small

def from_idx_to_UID(model,idx):
    """Takes an index from the user dictionary and returns the according User ID

    Args:
        model (_type_): Recommender model e.g. NRMS
        idx (int): Index (given by batch_data['user_index_batch'])

    Returns:
        str: User ID
    """
    val = list(model.test_iterator.uid2index.keys())[list(model.test_iterator.uid2index.values()).index(idx)]
    print(val)
    return val

def from_idx_to_words(model,idxs):
    rslt = ""
    for i in idxs:
        if i != 0:
            val = list(model.test_iterator.word_dict.keys())[list(model.test_iterator.word_dict.values()).index(i)]
            rslt = rslt + " " + val

    print(rslt)
    return rslt

def setup_full_mind():
    valid_clickhistory_large,valid_clickhistory_small,train_clickhistory_small = download_mind_large()

    valid_sessions_large, valid_histories_large = valid_clickhistory_large
    print('Number of valid large histories: ', len(valid_histories_large.keys()))

    # Small valid
    valid_sessions_small, valid_histories_small = valid_clickhistory_small
    print('Number of valid small histories: ', len(valid_histories_small.keys()))

    # Small train
    train_sessions_small, train_histories_small = train_clickhistory_small
    print('Number of train small histories: ', len(train_histories_small.keys()))


    train_small_users = set(train_histories_small)
    # print(len(train_small_users))
    valid_small_users = set(valid_histories_small)
    # print(len(valid_small_users))
    valid_large_users = set(valid_histories_large)
    # print(len(valid_large_users))

    candidate_users_no_overlap = valid_large_users.difference(valid_small_users)
    candidate_users_overlap = candidate_users_no_overlap.intersection(train_small_users)

    candidate_users_no_overlap_valid_or_train = candidate_users_no_overlap.difference(train_small_users)
    # print(len(candidate_users_no_overlap))
    # print(len(candidate_users_overlap))

    # Select 40k that did not appear in the training/validation set
    # Select 10k that did appear in the training but not in the validation phase (this is to have no overlap between validation and test sets)

    sampled_candidates_no_overlap = random.sample(list(candidate_users_no_overlap_valid_or_train), 40000)
    sampled_candidates_overlap = random.sample(list(candidate_users_overlap), 10000)

    # print(sampled_candidates_no_overlap[0])
    print(len(sampled_candidates_no_overlap))
    print(len(sampled_candidates_overlap))

    users = []
    users.extend(sampled_candidates_no_overlap)
    users.extend(sampled_candidates_overlap)
    print('Final amount of users: ', len(users))

    samples_users_set = set(users)
    train_users = set((train_histories_small))

    overlap = len(samples_users_set.intersection(train_users))
    print('Overlap of the 50k train users with the 50k sampled test users is: ', overlap)

    with open(os.path.join(f'MINDlarge_train.zip/valid',
                                    'behaviors.tsv')) as f:
        lines = f.readlines()
    sessions = []

    counter = 1

    with open(os.path.join(f'Dataset_small/test/',
                                        'behaviors.tsv'), 'w') as f:
        for i in tqdm(range(len(lines))):
            idx, userid, imp_time, click, imps = lines[i].strip().split("\t")
            if userid in users:
                line = f'{counter}\t{userid}\t{imp_time}\t{click}\t{imps}\n'
                f.write(line)
                counter += 1 

def get_dataframe(set_type : str, behav_file = True):
    """Get the pandas dataframe for behaviors or news file for train,test or valid set 

    Args:
        set_type (str): Should be 'train', 'valid' or 'test'
        behav_file (bool, optional): True if behavior dataframe should be returned. False for news file. Defaults to True.
    """
    if behav_file:
        # Path hardcoded 
        result_df = pd.read_csv(f"Dataset_small/{set_type}/behaviors.tsv",delimiter='\t')
        result_df.columns = ['Impression ID', 'User ID', 'Time', 'History' , 'Impressions']
        return result_df
    else:
        if set_type == "test":
            return None
        # Path hardcoded 
        result_df = pd.read_csv(f"MINDlarge_train.zip/{set_type}/news.tsv",delimiter='\t')
        result_df.columns=['News ID',
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "Title Entities",
            "Abstract Entities "]
        return result_df

def get_test_news():
    large_valid = pd.read_csv("MINDlarge_train.zip/valid/news.tsv",sep='\t',header=None)
    large_valid.columns=['News ID',
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities "]

    large_train = pd.read_csv("MINDlarge_train.zip/train/news.tsv",sep='\t',header=None)
    large_train.columns=['News ID',
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities "]

    small_train = pd.read_csv("Dataset_small/train/news.tsv",sep='\t',header=None)
    small_train.columns=['News ID',
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities "]

    small_valid = pd.read_csv("Dataset_small/valid/news.tsv",sep='\t',header=None)
    small_valid.columns=['News ID',
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities "]

    vlarge = pd.concat([large_train,large_valid,small_train,small_valid]).drop_duplicates()

    # Remove tabs
    vlarge['Abstract'] = vlarge['Abstract'].str.replace('\t',' ')
    vlarge['Title'] = vlarge['Title'].str.replace('\t',' ')

    filepath = Path('Dataset_small/test/news.tsv')
    vlarge.to_csv(filepath,sep='\t',index=False,header=False)

    return filepath

def behaviors_with_historysize(n : int,filepath = "Dataset_small/Users_with_n_behaviors",size_of_file = 156965,seed = 42):
    """Creates a new behaviors file with behaviors with a history of at least n 

    Args:
        n (int): Minimal history size 
        filepath (str, optional): Saving destination (without filename -> filepath/behaviors.tsv). Defaults to "Dataset_small/Users_with_n_behaviors".
        size_of_file (int, optional): Should be the same of the normal (small) file. Defaults to 156965.
        seed (int, optional): Seed for reproduction since entries are sampled. Defaults to 42.

    Returns:
        str: File path
    """

    if os.path.exists(f"{filepath}/behaviors_{n}.tsv"):
        return f"{filepath}/behaviors_{n}.tsv"
    # !Hardcoded Paths
    if not os.path.exists("MINDlarge_train.zip"):
        download_mind_large(True)
    df_full = pd.read_csv("MINDlarge_train.zip/train/behaviors.tsv",sep="\t",names=['Impression ID', 'User ID', 'Time', 'History' , 'Impressions'])
    df_valid_small = pd.read_csv("MINDlarge_train.zip/train/behaviors.tsv",sep="\t",names=['Impression ID', 'User ID', 'Time', 'History' , 'Impressions'])

    df_full['History'] = df_full['History'].astype('str')
    df_app = df_full[df_full['History'].apply(lambda x : len(x.split(' '))>= n )]

    print(len(df_app))

    # size_of_file = 156965 is the number of behaviors used in the normal small behaviors dataset
    if len(df_app) > size_of_file:
        df_app = df_app.sample(size_of_file,random_state=seed)
    else:
        print(f"In behaviors_with_historysize(): NOTE! The changed behaviors file consists only of {len(df_app)} entries instead of {size_of_file}")
    df_app.to_csv(f"{filepath}/behaviors_{n}.tsv",sep='\t',index=False,header=False)

    return f"{filepath}/behaviors_{n}.tsv"

def main():
    extract_mind("Dataset_large/MINDlarge_train.zip",
        "Dataset_large/MINDlarge_dev.zip",
        train_folder="train",
        valid_folder="valid",
        clean_zip_file=True)


if __name__=="__main__":
    setup_full_mind()