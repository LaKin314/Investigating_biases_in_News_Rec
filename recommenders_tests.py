import os
import sys

from utils.loading_utils import behaviors_with_historysize

if "../recommenders" not in sys.path:
    sys.path.insert(0,"../recommenders")

import numpy as np
import tensorflow as tf
import pandas as pd

import wandb
from recommenders.models.deeprec.deeprec_utils import \
    download_deeprec_resources
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import (get_mind_data_set,
                                               prepare_hparams)

# Import for the models
#  Newsrec models
from recommenders.models.newsrec.models.lstur import LSTURModel
from recommenders.models.newsrec.models.naml import NAMLModel
from recommenders.models.newsrec.models.npa import NPAModel
from recommenders.models.newsrec.models.nrms import NRMSModel

##  Deeprec models
from recommenders.models.deeprec.models.dkn import DKN
from recommenders.models.deeprec.io.dkn_iterator import DKNTextIterator



# ! Before importing tensorflow set devices (-1 for NO GPU Acceleration)

from scipy.special import softmax
from sklearn.metrics._ranking import roc_auc_score
from sklearn.metrics import accuracy_score

from scipy.stats import t


global shrink_size,data_path
data_path = "Dataset_small"
shrink_size = 20

def load_small():

    train_news_file = os.path.join(data_path, 'train', r'news.tsv')
    train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
    valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
    valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
    wordEmb_file = os.path.join(data_path, "utils", "embedding_all.npy")
    userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(data_path, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")

    # Use yaml as needed
    # yaml_file = os.path.join(data_path, "utils", r'naml.yaml')

    # Use demo code here
    mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set("small")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
        
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(mind_url, \
                                os.path.join(data_path, 'valid'), mind_dev_dataset)
    # if not os.path.exists(yaml_file):
    #     download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
    #                             os.path.join(data_path, 'utils'), mind_utils)
    
    return train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,userDict_file,wordDict_file,vertDict_file,subvertDict_file




def apply_shrink(x):
    if x is np.nan:
        return ""
    split_x = x.split(" ")
    if len(split_x) >= shrink_size:
        return ' '.join(split_x[:shrink_size])

def train_epochs(epochs : int):
    yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')
    train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,userDict_file,wordDict_file,vertDict_file,subvertDict_file = load_small()
    
    hparams= prepare_hparams(yaml_file, 
                                wordEmb_file=wordEmb_file,
                                wordDict_file=wordDict_file, 
                                userDict_file=userDict_file,
                                vertDict_file=vertDict_file, 
                                subvertDict_file=subvertDict_file,
                                batch_size=32,
                                epochs=epochs)
    nrms = NRMSModel(hparams, MINDIterator, 42)

    # Fit model and evaluate after every epoch
    _, results = nrms.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file,track_wand=False,results_as_list = True)

    return nrms

def get_scores_ttest_recommenders(model,test_news,test_behav,save_name=None):
    test_behaviors_file = test_behav
    test_news_file = test_news

    behav_df = pd.read_csv(test_behaviors_file,delimiter="\t",header=None)

    model.test_iterator.init_news(test_news_file)

    model.support_quick_scoring = False

    results = []

    iterations = min(10000,len(behav_df))

    for i in range(iterations):
        behav_df.iloc[i:i+1].to_csv("tmp/tmp_behav.tsv", sep="\t", index=False, header=False)
        new_behav = "tmp/tmp_behav.tsv"

        model.test_iterator.init_behaviors(behaviors_file = new_behav)
        results.append(model.run_eval(test_news_file,new_behav))

    df_results = pd.DataFrame(results)
    if save_name is not None:
        df_results.to_csv(f'Metric_scores/{save_name}')

def get_scores_ttest(model,test_news,test_behav,save_name=None):
    """Saves ROC AUC and accuracy scores on the the test set

    Args:
        model (_type_): Model to evaluate
        test_news (_type_): Test news file path
        test_behav (_type_): Test behaviors file path
        save_name (_type_, optional): When set the file will be saved under the given name. Defaults to None.

    Returns:
        _type_: Dataframe with results
    """

    results_fast = model.run_fast_eval(test_news,test_behav)

    y_pred = results_fast[2]
    y_true = results_fast[1]
    impression = results_fast[0]

    num_of_impr = len(y_pred)
    
    scores = {}

    for impr in range(num_of_impr):
        scores[impression[impr]] = {'ROC' : roc_auc_score(y_true[impr],y_pred[impr]) , 'Accuracy' : accuracy_score(y_true[impr],softmax(y_pred[impr]) > 0.5)}

    # Save Scores to csv
    score_df = pd.DataFrame(scores).T
    if save_name is not None:
        score_df.to_csv(f'Metric_scores/{save_name}',header=None)
        

    return score_df


def shrink_history_to_n(n : int, csv_file : str):
    """Generate new csv file with shrinked histories (limit size to n)

    Args:
        n (int): Maximum history size
        csv_file (string): Path to csv

    Returns:
        string: Return path to new shrinked behaviors file
    """

    
    # Return only path if file already exists
    if os.path.exists(f"Dataset_small/Changed_histories/behaviors_hist_{n}.tsv"):
        return os.path.join("Dataset_small","Changed_histories",f"behaviors_hist_{n}.tsv")

    global shrink_size
    shrink_size = n

    # Read in csv to pandas dataframe
    df = pd.read_csv(csv_file,sep="\t",names=['Impression ID', 'User ID', 'Time', 'History' , 'Impressions'])
    copy_data = df.copy(deep=True)

    # Apply the shrink top history and save file
    copy_data["History"] = copy_data["History"].apply(apply_shrink)
    copy_data.to_csv(f"Dataset_small/Changed_histories/behaviors_hist_{shrink_size}.tsv",sep="\t",header=False,index=False)

    return os.path.join("Dataset_small","Changed_histories",f"behaviors_hist_{shrink_size}.tsv")

def avg_metrics(metrics : list):
    """ Gets a list of lists of dictionaries and average the results of every i'th dictionary (which should hold the metrics)

    Args:
        metrics (list): List of lists of dictionaries

    Returns:
        dict: List of dictionaries (averaged)
    """

    assert type(metrics[0]) == list
    assert type(metrics[0][0]) == dict

    num_of_epochs = len(metrics[0])

    result = []
    for i in range(num_of_epochs):
        result_dict = {}
        for m in metrics:
            for k in m[i].keys():
                if not k in result_dict.keys():
                    result_dict[k] = 0
                result_dict[k] += m[i][k]
        
        for k in result_dict.keys():
            result_dict[k] = result_dict[k] / len(metrics)
        result.append(result_dict)

    return result
    
def run_different_histories_nrms(relevant_files : list, steps : list, project_name : str, config : dict, epochs = 5, save_model = False, track_wand = True, save_ttest_scores = False, seed_list = [42,43,44], train_on_same_size = False, train_on_large_enough = False, track_only_avg = True):
    """Tests how model performs on different history sizes

    Args:
        relevant_files (list): List of needed files for training/validation and seeting up hyperparams
        steps (list): History sizes to take into account 
        project_name (str): Name of the current project
        config (dict): Metadata for the project
        epochs (int, optional): Epochs to train. Defaults to 5.
        save_model (bool, optional): Save model weights to file. Defaults to False.
        track_wand (bool, optional): Track performance using weights&biases . Defaults to True.
        train_on_same_size (bool, optional): If true the training behaviors consists only users with a history length according to the tests. Defaults to False.
        train_on_large_enough (bool, optional): Like train_on_same_size but uses the same trainset for every training. Defaults to False.
        track_only_avg (bool, optional): If true the algorithm only tracking the average of metrics and not every single (seeded) step
    """

    # Name relevant files
    train_news_file, train_behaviors_file, valid_news_file\
        , valid_behaviors_file, wordEmb_file, userDict_file\
        , wordDict_file, vertDict_file\
        , subvertDict_file, yaml_file = relevant_files


    
    if track_wand:
        wandb.login()

    batch_size = 32
    # conf = {'Epochs':epochs,'Batch size':batch_size}

    seeds = seed_list


    # !Paths are hardcoded (change later) 
    test_news = "Dataset_small/test/news.tsv"
    test_behaviors = "Dataset_small/test/behaviors.tsv"

    for i in steps:
        # Use only users with a history bigger than i for training
        if train_on_same_size:
            # ! Hardcoded Paths
            train_news_file = "Dataset_small/test/news.tsv"
            train_behaviors_file = behaviors_with_historysize(i)
        # Use only users with a history bigger than 100 for training
        if train_on_large_enough:
            train_news_file = "Dataset_small/test/news.tsv"
            train_behaviors_file = behaviors_with_historysize(100)

        hparams = prepare_hparams(yaml_file, 
                                wordEmb_file=wordEmb_file,
                                wordDict_file=wordDict_file, 
                                userDict_file=userDict_file,
                                vertDict_file=vertDict_file, 
                                subvertDict_file=subvertDict_file,
                                batch_size=batch_size,
                                his_size = i,
                                epochs=epochs)

        # Modified Histories -> Shrink size to n
        changed_behaviors = shrink_history_to_n(i, train_behaviors_file)

        # This List gets lists of results for each epoch on each seed
        seeded_results = []
        for idx,seed in enumerate(seeds):
            if track_wand:
                wandb.init(project = project_name, name = f"|History| = {i} (SEED: {seed})",config=config, reinit = True)

            # Initiate new model
            nrms = NRMSModel(hparams, MINDIterator, seed)

            # Evaluation without training
            first_eval = nrms.run_eval(valid_news_file, valid_behaviors_file)
            if track_wand and not track_only_avg:
                wandb.log(first_eval)

            # Fit model and evaluate after every epoch
            _, results = nrms.fit(train_news_file, changed_behaviors, valid_news_file, valid_behaviors_file,track_wand=track_wand,results_as_list = True)

            seeded_results.append(results)

            # Get accuracy and ROC AUC Score on test files and save them to file
            if save_ttest_scores:
                get_scores_ttest_recommenders(nrms,test_news, test_behaviors, save_name = f'NRMS_SEED_{seed}_HIST_{i}')

            # Final evaluation on test set
            nrms.test_iterator.init_news(test_news)
            nrms.test_iterator.init_behaviors(behaviors_file = test_behaviors)
            test_eval = nrms.run_eval(test_news,test_behaviors)
     
            
            # Test runs get different names on wandb
            test_eval = {k + '_test' : v for k,v in test_eval.items()}
            if track_wand:
                wandb.log(test_eval)

            results.insert(0, first_eval)
            results.append(test_eval)
            
            
            # Save model if needed
            if save_model and (idx == 0):
                nrms.model.save_weights(f"Save_models/NRMS_weights_hist_{i}/")
            if track_wand:
                wandb.finish()
        
        # Average results over every seed
        if track_wand:
            if len(seeds) > 1:
                wandb.init(project = project_name, name = f"|History| = {i} (Average)",config=config, reinit=True)
                avg_results = avg_metrics(seeded_results)

                for sr in avg_results:
                    wandb.log(sr)
                
                wandb.finish()
                print(f"Average result : {avg_results}")
        print(f"Results for each seed : {seeded_results}")


def run_different_histories_nrms_single_step( train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,wordDict_file,\
userDict_file,vertDict_file,subvertDict_file,yaml_file,step,epochs = 5):
    wandb.login()

    batch_size = 32


    
    wandb.init(project="Different_histories_NRMS_with_hparam_hist_change(CAN BE DELETED)",name=f"|History| = {step} (tests)",reinit=True)

    hparams= prepare_hparams(yaml_file, 
                        wordEmb_file=wordEmb_file,
                        wordDict_file=wordDict_file, 
                        userDict_file=userDict_file,
                        vertDict_file=vertDict_file, 
                        subvertDict_file=subvertDict_file,
                        batch_size=batch_size,
                        history_size= step,
                        epochs=epochs)
    nrms = NRMSModel(hparams,MINDIterator,42)

    wandb.log(nrms.run_eval(valid_news_file,valid_behaviors_file))
    csv_file = pd.read_csv(train_behaviors_file,sep="\t",names=['Impression ID', 'User ID', 'Time', 'History' , 'Impressions'])
    nrms.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, valid_news_file)
    wandb.log(nrms.run_eval(valid_news_file,"Dataset_small/test/behaviors.tsv"))
    wandb.finish()

def run_different_hists(project_name : str ,config : dict, splitted_test : int, epochs = 5, save_model=True, track_wand=True, save_ttest_scores=False, train_on_same_size  = False, train_on_large_enough = False):

    # steps = [1, 10, 25, 50*, 100, 150]
    if splitted_test == 1:
        steps = [1, 10]
    if splitted_test == 2:
        steps = [25, 50] #!50 is optimal
    if splitted_test == 3:
        steps = [100]

    # Setup yaml  and everything needed for nrms
    yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')
    train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,userDict_file,wordDict_file,vertDict_file,subvertDict_file = load_small()
    
    relevant_files = [train_news_file   # News file for training
        ,train_behaviors_file           # Behaviors/Histories for training
        ,valid_news_file                # News file for validation
        ,valid_behaviors_file           # Behaviors/Histories for validation
        ,wordEmb_file                   # Pretrained word embeddings
        ,userDict_file                  # Lookup table/Dictionary user -> user ID
        ,wordDict_file                  # Lookup table/Dictionary Word -> word ID
        ,vertDict_file                  # Lookup table/Dictionary Category -> category ID
        ,subvertDict_file               # Lookup table/Dictionary Sub-Category -> Sub-Category ID
        ,yaml_file                      # YAML for hyperparams 
    ]


    # Run for 5 epochs, save model weights and track results via wandb
    run_different_histories_nrms(relevant_files, steps,project_name, config, epochs=epochs, save_model=save_model, track_wand=track_wand, save_ttest_scores=save_ttest_scores, train_on_same_size = train_on_same_size, train_on_large_enough=train_on_large_enough)



def tests():
    test_behaviors_file="Dataset_small/test/behaviors.tsv"

    p_data = pd.read_csv(test_behaviors_file,sep="\t",names=['Impression ID', 'User ID', 'Time', 'History' , 'Impressions'])
    print(p_data.head())

def tests_1707():
    train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,userDict_file,wordDict_file,vertDict_file,subvertDict_file = load_small()
    yaml_file = os.path.join("Dataset_small", "utils", r'nrms.yaml')

    test_behaviors_file="Dataset_small/test/behaviors.tsv"

    hparams= prepare_hparams(yaml_file, 
                            wordEmb_file=wordEmb_file,
                            wordDict_file=wordDict_file, 
                            userDict_file=userDict_file,
                            vertDict_file=vertDict_file, 
                            subvertDict_file=subvertDict_file,
                            batch_size=32,
                            epochs=2)

    nrms = NRMSModel(hparams,MINDIterator,42)
    wandb.login()
    wandb.init(project="DELETE",name=f"|History| = (tests)",reinit=True)

    nrms, results = nrms.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, valid_news_file,track_wand=True,results_as_list = True)
    print(results)

def test_single_run():
    steps = [10]


    # Setup yaml  and everything needed for nrms
    yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')
    train_news_file,train_behaviors_file,valid_news_file,valid_behaviors_file,wordEmb_file,userDict_file,wordDict_file,vertDict_file,subvertDict_file = load_small()
    
    relevant_files = [train_news_file   # News file for training
        ,train_behaviors_file           # Behaviors/Histories for training
        ,valid_news_file                # News file for validation
        ,valid_behaviors_file           # Behaviors/Histories for validation
        ,wordEmb_file                   # Pretrained word embeddings
        ,userDict_file                  # Lookup table/Dictionary user -> user ID
        ,wordDict_file                  # Lookup table/Dictionary Word -> word ID
        ,vertDict_file                  # Lookup table/Dictionary Category -> category ID
        ,subvertDict_file               # Lookup table/Dictionary Sub-Category -> Sub-Category ID
        ,yaml_file                      # YAML for hyperparams 
    ]

    # Run for 5 epochs, save model weights and track results via wandb
    run_different_histories_nrms(relevant_files, steps,"TEST (DELETE THIS)", epochs=2, save_model=False, track_wand=False,seed_list=[42])

def main():

    # --- Run each test "run_different_hists(1)", "run_different_hists(2)", "run_different_hists(3)" on different GPUs ---

    # Set details of the run
    # -----
    model_type = "NRMS"
    project_name = "Different_histories_on_NRMS (Train with users on ! size >= 100 ! and changed his_size)"
    description = """This run uses the shrinked histories with the train set of the size=100 history.
    Additionally the his_size is set which has an impact to the size of the attention.
    This results should be compared to the last results to see if the addded his_size parameter has any effect on the learned attention and performance.
    The model is saved to get access to the attention weights"""
    shrinked_history = True
    epochs = 5
    save_model = True
    track_wand = True
    save_ttest_scores = False

    train_on_same_size = False
    train_on_large_enough = True

    steps = [1,10,25,50,100]

    # ----

    config = {"Model": model_type, "Epochs": epochs, "steps": steps,"Shrinked history" : shrinked_history, "trained on same size" : train_on_same_size, "Trained on same size with same train set" : train_on_large_enough, "Description" : description}


    # run_different_hists(project_name, config, 1, epochs=epochs, save_model=save_model, track_wand=track_wand, save_ttest_scores=save_ttest_scores,train_on_same_size=train_on_same_size,train_on_large_enough=train_on_large_enough)
    # run_different_hists(project_name, config, 2,epochs=epochs, save_model=save_model, track_wand=track_wand, save_ttest_scores=save_ttest_scores,train_on_same_size=train_on_same_size,train_on_large_enough=train_on_large_enough)
    run_different_hists(project_name, config, 3,epochs=epochs, save_model=save_model, track_wand=track_wand, save_ttest_scores=save_ttest_scores,train_on_same_size=train_on_same_size,train_on_large_enough=train_on_large_enough)

    
    # ---

    # Tests
    # test_single_run()
    


if __name__ == "__main__":
    main()
