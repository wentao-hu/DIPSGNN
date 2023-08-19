# DIPSGNN
This is our implementation for "Towards Differential Privacy in Sequential Recommendation: A Noisy Graph Neural Network Approach""

## How to run.
1. Download dataset:
- [ML-1M](https://grouplens.org/datasets/movielens/)
- [Yelp](https://www.yelp.com/dataset)
- [Tmall](https://tianchi.aliyun.com/dataset/42)

2. Preprocess dataset:
```
cd datasets
python preprocess.py
```
3. run DIPSGNN with default hyperparameters:
```
python main_dipsgnn.py --dataset "ml-1m" --clip_norm 0.5 --epsilon1 20 --epsilon 4 --delta 2.5e-07 --step 1
```

## Requirements
```
python: 3.8
torch: 1.12.1
```

## Code Reference
[Session-based Recommendation with Graph Neural Networks](https://github.com/CRIPAC-DIG/SR-GNN)
