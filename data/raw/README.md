The dataset can be found by this link:
https://grouplens.org/datasets/movielens/100k/

As a train set, I used ua.base, as a validation set - ua.test.

```python
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
mv ml-100k/ua.base Movie_Recommendations/data/raw
mv ml-100k/ua.test Movie_Recommendations/data/raw
```
