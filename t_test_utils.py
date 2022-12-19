import pandas as pd
import numpy as np
import os
import sys

import tensorflow as tf
from recommenders_tests import load_small
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import prepare_hparams

from scipy.stats import ttest_rel,ttest_ind
from scipy.stats import t


def get_loaded_model_with_seed(model_type : str, weight_paths : list, seed = 42, batch_size = 32, epochs = 5):
    """ Load model with its saved weights for each path in weight_path according to model type

    Args:
        model_type (str): Should be in [NRMS,LSTUR]
        weight_paths (list): Relative path to "/home/langenhagen/Masterthesis/Masterarbeit/Save_models/"
        seed (int, optional): Seed to be set in the models. Defaults to 42.
        batch_size (int, optional): Batch size of the model when trained. Defaults to 32.
        epochs (int, optional): Epochs when trained. Defaults to 5.

    Returns:
        list: List of models 
    """
    loaded_models = []
    # Get all relevant files for the models
    train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,userDict_file,wordDict_file,vertDict_file,subvertDict_file = load_small()
    if model_type == "NRMS":
        yaml_file = os.path.join("/home/langenhagen/Masterthesis/Masterarbeit/Dataset_small", "utils", r'nrms.yaml')
        hparams= prepare_hparams(yaml_file, 
                            wordEmb_file=wordEmb_file,
                            wordDict_file=wordDict_file, 
                            userDict_file=userDict_file,
                            vertDict_file=vertDict_file, 
                            subvertDict_file=subvertDict_file,
                            batch_size=32,
                            epochs=5)
        for path in weight_paths:
            model = NRMSModel(hparams,MINDIterator,seed)
            model.model.load_weights(f"/home/langenhagen/Masterthesis/Masterarbeit/Save_models/{path}/")
            loaded_models.append(model)
    
    return loaded_models

def paired_t_test(X_,Y_):
    diff = X_ - Y_

    squared_diff = np.square(diff)

   
    
    sum_squared_diff = np.sum(squared_diff)
    
    sum_diff = np.sum(diff)
   
    n = len(X_)

    denom = (sum_diff/n)
    nom = sum_squared_diff - (sum_diff*sum_diff)/2

    nom = nom / ((n-1) * n)
    t =  denom / np.sqrt(nom)

    return t

# https://medium.com/analytics-vidhya/using-the-corrected-paired-students-t-test-for-comparing-the-performance-of-machine-learning-dc6529eaa97f
def ttest_mod(RFC_score, SVM_score):
    #Compute the difference between the results
    diff = [y - x for y, x in zip(RFC_score, SVM_score)]
    #Comopute the mean of differences
    d_bar = np.mean(diff)
    #compute the variance of differences
    sigma2 = np.var(diff,ddof=1)
    #compute the number of data points used for training 
    n1 = len(RFC_score)
    #compute the number of data points used for testing 
    n2 = len(RFC_score)
    #compute the total number of data points
    n = len(RFC_score)
    #compute the modified variance
    sigma2_mod = sigma2 * (1/n + n2/n1)
    #compute the t_static
    t_static =  d_bar / np.sqrt(sigma2_mod)
    from scipy.stats import t
    #Compute p-value and plot the results 
    Pvalue = ((1 - t.cdf(t_static, n-1))*2)
    return t_static, Pvalue

def has_same_dist(p, alpha = 0.05):
    if p > alpha:
        print(f"Probably same distribution (p = {round(p,4)})")
        return True
    elif p <= alpha:
        print("Probably different distribution -> Results are significant (p = {round(p,4)})")
        return False
    else:
        print("Error in values (for instance exact same distributions)")
        return False

def ttest(modelA, modelB, num_of_samples = 1000,\
        test_behaviors_file = "/home/langenhagen/Masterthesis/Masterarbeit/Dataset_small/test/behaviors.tsv",\
        test_news_file = "/home/langenhagen/Masterthesis/Masterarbeit/Dataset_small/test/news.tsv"):
    """ Tests significance difference between two models via t-test

    Args:
        modelA (Recommenders model): First model
        modelB (Recommenders model): Second model
        num_of_samples (int, optional): Number of samples to evaluate. Defaults to 1000.

    Returns:
        list: [T-statistics : double/list, P-values : double/list]
    """


    behav_df = pd.read_csv(test_behaviors_file,delimiter="\t",header=None)

    modelA.test_iterator.init_news(test_news_file)
    modelB.test_iterator.init_news(test_news_file)

    modelA.support_quick_scoring = False
    modelB.support_quick_scoring = False

    results_A = []
    results_B = []


    for i in range(1000):
        behav_df.iloc[i:i+1].to_csv("/home/langenhagen/Masterthesis/Masterarbeit/tmp/tmp_behav.tsv", sep="\t", index=False, header=False)
        new_behav = "/home/langenhagen/Masterthesis/Masterarbeit/tmp/tmp_behav.tsv"

        modelA.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_A.append(modelA.run_eval(test_news_file,new_behav))

        modelB.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_B.append(modelB.run_eval(test_news_file,new_behav))

    df_A = pd.DataFrame(results_A)
    df_B = pd.DataFrame(results_B)

    results = ttest_ind(df_A,df_B)

    return results

def print_results_ttest(results, alpha = 0.05):

    used_metrics = ["Group AUC", "Mean MRR", "nDCG@5", "nDCG@10"]

    if isinstance(results[1],float):
        has_same_dist(results[1], alpha = 0.05)
    else:
        for idx, metric in enumerate(used_metrics):
            print(used_metrics)
            print("---------")
            has_same_dist(results[1][idx])
            print("---------")

def paired_ttest_with_anchor(models: list, anchor_model_idx):
    num_of_models = len(models)
    result_list = []

    for idx, model in enumerate(models):
        if idx != anchor_model_idx:
            results = ttest(modelA=models[anchor_model_idx],modelB=model)
            print_results_ttest(results)
            result_list.append(results)
        else:
            result_list.append(None)

    return result_list

def ttest_proto():

    train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,userDict_file,wordDict_file,vertDict_file,subvertDict_file = load_small()
    yaml_file = os.path.join("/home/langenhagen/Masterthesis/Masterarbeit/Dataset_small", "utils", r'nrms.yaml')

    test_behaviors_file="/home/langenhagen/Masterthesis/Masterarbeit/Dataset_small/test/behaviors.tsv"

    hparams= prepare_hparams(yaml_file, 
                            wordEmb_file=wordEmb_file,
                            wordDict_file=wordDict_file, 
                            userDict_file=userDict_file,
                            vertDict_file=vertDict_file, 
                            subvertDict_file=subvertDict_file,
                            batch_size=32,
                            epochs=1)

    nrms_1 = NRMSModel(hparams,MINDIterator,42)
    nrms_10 = NRMSModel(hparams,MINDIterator,42)
    nrms_25 = NRMSModel(hparams,MINDIterator,42)
    nrms_50 = NRMSModel(hparams,MINDIterator,42)
    nrms_100 = NRMSModel(hparams,MINDIterator,42)
    nrms_150 = NRMSModel(hparams,MINDIterator,42)


    nrms_1.model.load_weights("/home/langenhagen/Masterthesis/Masterarbeit/Save_models/NRMS_weights_hist_1/").expect_partial()
    nrms_10.model.load_weights("/home/langenhagen/Masterthesis/Masterarbeit/Save_models/NRMS_weights_hist_10/").expect_partial()
    nrms_25.model.load_weights("/home/langenhagen/Masterthesis/Masterarbeit/Save_models/NRMS_weights_hist_25/").expect_partial()
    nrms_50.model.load_weights("/home/langenhagen/Masterthesis/Masterarbeit/Save_models/NRMS_weights_hist_50/").expect_partial()
    nrms_100.model.load_weights("/home/langenhagen/Masterthesis/Masterarbeit/Save_models/NRMS_weights_hist_100/").expect_partial()
    nrms_150.model.load_weights("/home/langenhagen/Masterthesis/Masterarbeit/Save_models/NRMS_weights_hist_150/").expect_partial()
    
    
    test_behaviors_file = "/home/langenhagen/Masterthesis/Masterarbeit/Dataset_small/test/behaviors.tsv"
    test_news_file = "/home/langenhagen/Masterthesis/Masterarbeit/Dataset_small/test/news.tsv"

    behav_df = pd.read_csv(test_behaviors_file,delimiter="\t",header=None)


    nrms_1.test_iterator.init_news(test_news_file)
    nrms_10.test_iterator.init_news(test_news_file)
    nrms_25.test_iterator.init_news(test_news_file)
    nrms_50.test_iterator.init_news(test_news_file)
    nrms_100.test_iterator.init_news(test_news_file)
    nrms_150.test_iterator.init_news(test_news_file)
 

    nrms_1.support_quick_scoring = False
    nrms_10.support_quick_scoring = False
    nrms_25.support_quick_scoring = False
    nrms_50.support_quick_scoring = False
    nrms_100.support_quick_scoring = False
    nrms_150.support_quick_scoring = False

    results_100 = []
    results_25 = []
    results_1 = []
    results_10 = []
    results_150 = []
    results_50 = []


    for i in range(1000):
        behav_df.iloc[i:i+1].to_csv("/home/langenhagen/Masterthesis/Masterarbeit/tmp/tmp_behav.tsv", sep="\t", index=False, header=False)
        new_behav = "/home/langenhagen/Masterthesis/Masterarbeit/tmp/tmp_behav.tsv"

        nrms_1.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_1.append(nrms_1.run_eval(test_news_file,new_behav))

        nrms_10.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_10.append(nrms_10.run_eval(test_news_file,new_behav))

        nrms_25.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_25.append(nrms_25.run_eval(test_news_file,new_behav))

        nrms_50.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_50.append(nrms_50.run_eval(test_news_file,new_behav))

        nrms_100.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_100.append(nrms_100.run_eval(test_news_file,new_behav))

        nrms_150.test_iterator.init_behaviors(behaviors_file = new_behav)
        results_150.append(nrms_150.run_eval(test_news_file,new_behav))

    df_1 = pd.DataFrame(results_1)
    df_10 = pd.DataFrame(results_10)
    df_25 = pd.DataFrame(results_25)
    df_50 = pd.DataFrame(results_50)
    df_100 = pd.DataFrame(results_100)
    df_150 = pd.DataFrame(results_150)

    print_results_ttest(ttest_ind(df_1,df_50))
    print_results_ttest(ttest_ind(df_10,df_50))
    print_results_ttest(ttest_ind(df_25,df_50))
    print_results_ttest(ttest_ind(df_100,df_50))
    print_results_ttest(ttest_ind(df_150,df_50))


if __name__ == "__main__":

    different_hists = [1, 10, 25, 50, 100, 150] 
    model_paths = []

    for i in range(len(different_hists)):
        model_paths.append(f"NRMS_weights_hist_{different_hists[i]}")
    
    models = get_loaded_model_with_seed("NRMS",model_paths)

    # Compare with hist_size of 50 (idx = 3)
    paired_ttest_with_anchor(models,3)