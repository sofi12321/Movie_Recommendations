# Movie Recommendations

### Creator
Sofi Shulepina

B21-DS-02

s.zaitseva@innopolis.university

## Task Deskription

Create a Movie Recommender System that will suggest to a user some movies unseen before by him/her.

## Data Description

Implementation based on MovieLens 100K dataset consisting user ratings to movies, user and movie information.

Initial data may be found by link: https://grouplens.org/datasets/movielens/100k/


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

### Run model

Let's evaluate the model and look at recomendations

```
python benchmark/evaluate.py
```


### References

All used materials are listed in file references.md
