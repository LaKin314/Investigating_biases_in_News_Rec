from transformers import pipeline
import os
import sys
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np



# News
path_train_news_small = "Dataset_small/train/news.tsv"
path_train_news_large = "MINDlarge_train.zip/train/news.tsv"

path_val_news_small = "Dataset_small/valid/news.tsv"
path_val_news_large = "MINDlarge_train.zip/valid/news.tsv"

# Behaviors
path_train_behaviors_small = "Dataset_small/train/behaviors.tsv"
path_train_behaviors_large = "MINDlarge_train.zip/train/behaviors.tsv"

path_val_behaviors_small = "Dataset_small/valid/behaviors.tsv"
path_val_behaviors_large = "MINDlarge_train.zip/valid/behaviors.tsv"


cols_news = ['News ID',
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities "]

cols_behaviors = ['Impression ID', 
    'User ID',
    'Time',
    'History' ,
    'Impressions']

def create_sentiments_in_news(news_data = "Dataset_small/train/news.tsv", save_csv = "Dataset_small/train/"):
    """Creates a new dataframe which yields another column 'Sentiment' which indicates the sentiment of the title

    Args:
        news_data (str, optional): Path to the news data which should be used. Defaults to train data "Dataset_small/train/news.tsv".
        save_csv (str, optional): Path to the save directory. If None the dataframe is not saved. Defaults to "Dataset_small/train/".

    Returns:
        pd.Dataframe: A new dataframe with column 'Sentiment'
    """


    news_dataframe = pd.read_csv(news_data,delimiter='\t',index_col=False)
    news_dataframe.columns =  ['News ID',
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities "]

    # Setting up the pipeline for sentiment analysis
    pipe = pipeline("text-classification")

    # Sentiment will be set to 'neutral' if model is less than the threshold sure
    uncertainty_threshold = 0.5

    # Adding sentiment to dataframe
    with tqdm(total=len(news_dataframe)) as pbar:
        for row in news_dataframe.to_dict("records"):
            sentiment = pipe(row['Title'])

            if (sentiment[0]['label'] == 'POSITIVE' and sentiment[0]['score'] > uncertainty_threshold):
                news_dataframe.loc[news_dataframe['News ID'] == row['News ID'],'Sentiment'] = 1
            elif (sentiment[0]['label'] == 'NEGATIVE' and sentiment[0]['score'] > uncertainty_threshold):
                news_dataframe.loc[news_dataframe['News ID'] == row['News ID'],'Sentiment'] = -1
            else:
                news_dataframe.loc[news_dataframe['News ID'] == row['News ID'],'Sentiment'] =  0
            pbar.update(1)
    
    if save_csv is not None:
        news_dataframe.to_csv(f"{save_csv}/test_with_sentiment.tsv",sep="\t")

    return news_dataframe

def hist_to_cat_count(row,small_val_news):
    if row['History'] is not np.nan:
        hist = row['History'].split(' ')
    else:
        return
    for entry in hist:
        row[small_val_news[small_val_news['News ID'] == entry]['Category']] += 1

def hist_to_cat():

    small_val_news = pd.read_csv(path_val_news_small,delimiter='\t')
    small_val_news.columns = cols_news

    small_val_behaviors = pd.read_csv(path_val_behaviors_small,delimiter='\t')
    small_val_behaviors.columns = cols_behaviors

    categories = set(small_val_news['Category'])

    df_hist_cats = small_val_behaviors.drop_duplicates(subset=['User ID']).copy()
    df_hist_cats.reset_index()

    for cat in categories:
            df_hist_cats[cat] = 0

    with tqdm(total=len(df_hist_cats)) as pbar:
        for idx in df_hist_cats.index:
            row = df_hist_cats.loc[idx]

            if row['History'] is not np.nan:
                hist = row['History'].split(' ')
            else:
                continue
            for entry in hist:
                try:
                    df_hist_cats.loc[idx,[small_val_news[small_val_news['News ID'] == entry]['Category'].item()]] += 1
                except:
                    print(f"There was a problem with the idx: {idx} and {small_val_news[small_val_news['News ID'] == entry]['Category']}")
            pbar.update(1)

    df_hist_cats.to_csv("hist_to_cats.tsv",sep='\t')

def categories_count_to_dict():

    # News
    path_train_news_small = "Dataset_small/train/news.tsv"

    # Behaviors
    path_train_behaviors_small = "Dataset_small/train/behaviors.tsv"


    small_train_news = pd.read_csv(path_train_news_small,delimiter='\t')
    small_train_news.columns = cols_news

    small_train_behaviors = pd.read_csv(path_train_behaviors_small,delimiter='\t')
    small_train_behaviors.columns = cols_behaviors

    categories = set(small_train_news['Category']) # Each category in the dataset

    category_count = {}

    for cat in categories:
        category_count[cat] = len(small_train_news.loc[small_train_news['Category'] == cat])

    category_count_shown = {}
    category_count_clicked = {}
    for cat in categories:
        category_count_shown[cat] = 0
        category_count_clicked[cat] = 0
    with tqdm(total=len(small_train_behaviors)) as pbar:
        for impr in small_train_behaviors['Impressions'].to_dict().values():
            splitted = impr.split(' ')
            for s in splitted:
                news = s.split('-')
                category_count_shown[small_train_news[small_train_news['News ID'] == news[0]]['Category'].item()] += 1
                # category_count_clicked[small_train_news[small_train_news['News ID'] == news[0]]['Category'].item()] += int(news[1])
            pbar.update(1)
    if True:
        with open('Snapshots/category_count_shown.pkl', 'wb') as f:
            pickle.dump(category_count_shown, f)
        # with open('Snapshots/category_count_clicked.pkl', 'wb') as f:
        #     pickle.dump(category_count_clicked, f)

    


def get_impression_and_behavior_sentiments():
    # Train sentiment
    # Like in the original paper, the overall sentiment is measured by the mean sentiment of each user


    cols_news = ['News ID',
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities "]

    cols_behaviors = ['Impression ID', 
    'User ID',
    'Time',
    'History' ,
    'Impressions']
    # News
    path_train_news_small = "Dataset_small/train/news.tsv"

    # Behaviors
    path_train_behaviors_small = "Dataset_small/train/behaviors.tsv"



    small_train_news = pd.read_csv(path_train_news_small,delimiter='\t')
    small_train_news.columns = cols_news

    small_train_behaviors = pd.read_csv(path_train_behaviors_small,delimiter='\t')
    small_train_behaviors.columns = cols_behaviors

    # The dictionary will have the form {'User': {'Behaviors' :  X, 'Impressions' : Y}}
    # X = mean sentiment of the news titles
    # Y = mean(mean sentiment of the impressions)
    results_for_users = {}

    # Created sentiments via bias_utils.py : create_sentiments_in_news()
    path_sentimented_news = "Dataset_small/train/test_with_sentiment.tsv"
    sentimented_news = pd.read_csv(path_sentimented_news,sep="\t",index_col=0)

    # Each user should be evaluated only once
    user_behaviors = small_train_behaviors.sort_values(['User ID'])

    current_user = ""
    user_behaviors.reset_index()
    with tqdm(total=len(user_behaviors)) as pbar:
        for row in user_behaviors.to_dict('records'):
            tmp_user = row['User ID']
            if row['User ID'] != current_user:
                if current_user in results_for_users.keys(): # 
                    results_for_users[current_user]['Impressions'] = np.mean(results_for_users[current_user]['Impressions'])

                current_user = row['User ID']
                # Behavior sentiment
                mean_sum = 0
                if row['History'] is not np.nan:
                    history = row['History'].split(' ')
                else:
                    history = []
                his_size = len(history)
                for entry in history:
                    if not sentimented_news.loc[sentimented_news['News ID'] == entry,'Sentiment'].empty:
                        sentiment = sentimented_news.loc[sentimented_news['News ID'] == entry,'Sentiment'].item()
                    else:
                        his_size += -1
                        continue
                    if sentiment != 0:
                        mean_sum += sentiment
                    else:
                        his_size += -1  # If the model is uncertain about sentiment, the entry will be removed
                results_for_users[current_user] = {}
                if his_size == 0:
                    results_for_users[current_user]['Behaviors'] = 0
                else:
                    results_for_users[current_user]['Behaviors'] = mean_sum/his_size
                results_for_users[current_user]['Impressions'] = []

                # Impression sentiment
                if row['Impressions'] is not np.nan:    # Check if there is an impression (This should be always the case)
                    impr = row['Impressions'].split(' ')
                else:
                    print(f"From line 141: {row['User ID']}")
                    impr = []
                mean_sum = 0
                impr_size = len(impr)
                for entry in impr:
                    if not sentimented_news.loc[sentimented_news['News ID'] == entry.split('-')[0],'Sentiment'].empty:
                        sentiment = sentimented_news.loc[sentimented_news['News ID'] == entry.split('-')[0],'Sentiment'].item()
                    else:
                        impr_size += -1
                        continue
                    if sentiment != 0:
                        mean_sum += sentiment
                    else:
                        impr_size += -1  # If the model is uncertain about sentiment, the entry will be removed
                if impr_size != 0:
                    results_for_users[current_user]['Impressions'].append(mean_sum/impr_size)
                else:
                    results_for_users[current_user]['Impressions'].append(0)
            else:
                # Impression sentiment
                impr = row['Impressions'].split(' ')
                mean_sum = 0
                impr_size = len(impr)
                for entry in impr:
                    if not sentimented_news.loc[sentimented_news['News ID'] == entry.split('-')[0],'Sentiment'].empty:
                        sentiment = sentimented_news.loc[sentimented_news['News ID'] == entry.split('-')[0],'Sentiment'].item()
                    else:
                        impr_size += -1
                        continue
                    if sentiment != 0:
                        mean_sum += sentiment
                    else:
                        impr_size += -1  # If the model is uncertain about sentiment, the entry will be removed
                if impr_size != 0:
                    results_for_users[current_user]['Impressions'].append(mean_sum/impr_size)
                else:
                    results_for_users[current_user]['Impressions'].append(0)
            pbar.update(1)
        results_for_users[tmp_user]['Impressions'] = np.mean(results_for_users[tmp_user]['Impressions'])



    with open('Snapshots/saved_dictionary.pkl', 'wb') as f:
        pickle.dump(results_for_users,f)
def main():
    # HERE THE TEST SHOULD BE SET
    hist_to_cat()


if __name__ == "__main__":
    main()