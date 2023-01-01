# Investigating Biases in News Recommendation
## Abstract
With increasing automation and the continuous development of machine learning, modern algorithms are now used in almost all areas to improve and simplify workflows. Recommendation Systems (RS) are one group of these algorithms. They enable automated suggestions of items based on the interests of the user. In this work, we will focus on the investigation of biases in recommendation systems for news. News Recommendation Systems (NRS) provide a way to suggest targeted news to users according to their needs. As news is the primary source of information, it is imperative that it is presented fairly and free from bias. Thus, in addition to good recommendations for the user, novelty and diversity should be ensured by the NRS. For this purpose, several experiments are conducted with the MIND dataset, which has collected 1,000,000 users’ data on MSN. This work gives an overview of the different biases in the feedback loop of news recommendation systems. In particular, data and model biases are examined and related to other user biases. The research should enable a template for bias modeling. 


## File structure
.
└── Edit me to generate/
    ├── a/
    │   └── nice/
    │       └── tree/
    │           ├── diagram!
    │           └── :)
    └── Use indentation/
        ├── to indicate/
        │   ├── file
        │   ├── and
        │   ├── folder
        │   └── nesting.
        └── You can even/
            └── use/
                ├── markdown
                └── bullets!

## Installation

### Setup the repository
This repository can be cloned using <https://github.com/LaKin314/Investigating_biases_in_News_Rec.git>. Before changing the directory to this repository, please also clone <https://github.com/LaKin314/recommenders.git>.

Since we ran test on some models using [Recommenders]:<https://github.com/Microsoft/Recommenders> by Microsoft, we forked the original library to modify the code for evaluation purpose. The forked repository can be found [here]:<https://github.com/LaKin314/recommenders>.

Proceed with changing the directory after cloning.

```sh
git clone https://github.com/LaKin314/Investigating_biases_in_News_Rec.git
git clone https://github.com/LaKin314/recommenders.git
cd Investigating_biases_in_News_Rec
```

### Setup Anaconda
The experiments are done with Tensorflow and Python. To ensure the same dependencies, we used a Anaconda. So please ensure that you have [Anaconda]:<https://docs.anaconda.com/anaconda/install/index.html> set-up. 
To create the conda-environment use the command:

```sh
conda env create -f environment.yaml
```

This creates an environment called "investigate_newsrec_bias". To use the environment use the command:

```sh
conda activate investigate_newsrec_bias
```

The [environment.yaml]:<https://github.com/LaKin314/Investigating_biases_in_News_Rec/blob/main/environment.yaml> contains all dependencies needed for the tests.

### Download relevant files
Due to small storage space in Github, additional big files can be downloaded via the [scripts]:<https://github.com/LaKin314/Investigating_biases_in_News_Rec/tree/main/utility_scripts>.

| Skript | Description| Link |
| ------ | ------ | ------ |
| [utility_scripts/download_all.sh]:<https://github.com/LaKin314/Investigating_biases_in_News_Rec/blob/main/utility_scripts/download_all.sh> | Load all files | ------ |
| [utility_scripts/download_small.sh]:<https://github.com/LaKin314/Investigating_biases_in_News_Rec/blob/main/utility_scripts/download_data_small.sh>  | Load small MIND and every modified version used| <https://seafile.cloud.uni-hannover.de/f/d1752d0ec90148ddb1bb/?dl=1> |
| [utility_scripts/download_large.sh]:<https://github.com/LaKin314/Investigating_biases_in_News_Rec/blob/main/utility_scripts/download_data_large.sh>  | Load large MIND | <https://seafile.cloud.uni-hannover.de/f/94ac34a318f2449180df/?dl=1
> |
| [utility_scripts/download_results.sh]:<https://github.com/LaKin314/Investigating_biases_in_News_Rec/blob/main/utility_scripts/download_data_results.sh>  | Load results of the tests | <https://seafile.cloud.uni-hannover.de/f/20fe4c6c2b874fd79106/?dl=1> |
| [utility_scripts/download_models.sh]:<https://github.com/LaKin314/Investigating_biases_in_News_Rec/blob/main/utility_scripts/download_data_models.sh> | Load saved models of NRMS | <https://seafile.cloud.uni-hannover.de/f/a7329732a61c4a7383a4/?dl=1> |

You may want to download all used files when first time using the code. However, it is also possible to download/create these files in the process. 

```sh
bash utility_scripts/download_all.sh
```
If you want to download only specific files use the other scripts.

