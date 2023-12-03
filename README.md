# Movie Recommendations

### Creator
Sofi Shulepina

B21-DS-02

s.zaitseva@innopolis.university

## Task Deskription

Create a Movie Recommender System that will suggest to a user some movies unseen before by him/her.

## Repository structure

```
Movie_Recommendations
├── README.md # The top-level README
│
├── data
│   ├── interim  # Intermediate data that has been transformed, access by a link
│   └── raw      # The original, immutable data, access by a link
│
├── models       
│   └── model.pt # Trained model weight
│
├── notebooks    #  Jupyter notebooks
│   ├── 1_data_exploration_vizualization_preprocessing.ipynb  # Download and prepare data
│   └── 2_model_train_test_visualize.ipynb  # Model building, training and evaluating       
│
├── references.md  # Used resources
│
├── reports      
│   └── final_report.pdf # Full description of the solution
├── requirements.txt # The requirements file for reproducing the analysis environment
│
└── benchmark                 # Code for model evaluation
    │                 
    ├── data            # Dataset to be used for evaluation, access by a link
    └── evaluate.py # Script for model evaluation
```


### Requirements 

Install requirements before start of the work

```python
pip install requirements.txt
```

## Data Description

Implementation based on MovieLens 100K dataset consisting user ratings to movies, user and movie information.

Initial data may be found by link: https://grouplens.org/datasets/movielens/100k/

### Download data

#### MovieLens 100k - initial data

```python
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
mv ml-100k/ua.base Movie_Recommendations/data/raw
mv ml-100k/ua.test Movie_Recommendations/data/raw
mv ml-100k/u.user Movie_Recommendations/data/raw
mv ml-100k/u.genre Movie_Recommendations/data/raw
mv ml-100k/u.item Movie_Recommendations/data/raw
```

#### Preprocessed datasets
Step-by-step preprocessing is shown in notebooks/1_data_exploration_vizualization_preprocessing.ipynb

```python
cd Movie_Recommendations/data/interim
gdown https://drive.google.com/uc?id=1VLrWudGAGcQDRHiKDx-T72XfAmU5FUdY
gdown https://drive.google.com/uc?id=19GAIyyOM1TYTWmMa99XFHCOpwcaFLKGB
gdown https://drive.google.com/uc?id=1n5CWF21lXCy9Sx8Mh5-eGe3PEKBfz0dL
gdown https://drive.google.com/uc?id=1g3SHTa_lNFtP8YtM7jrSGdfeau4RfrsL
```

#### Dataset for evaluation
Preprocessed by the same steps from notebooks/1_data_exploration_vizualization_preprocessing.ipynb

```python
cd Movie_Recommendations/benchmark/data
gdown https://drive.google.com/uc?id=1UYSvHSjFj__xB0CJrskIdfW0bSZXwut9
```

### Run model

Let's evaluate the model and look at recomendations. Please, make sure that all datasets are downloaded in the nedeed directories before running the evaluation process.

```
python benchmark/evaluate.py
```


### References

All used materials are listed in file references.md
