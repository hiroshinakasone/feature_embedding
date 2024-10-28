# Feature Embedding
A sample project to convert feature vectors into embeddings using rating data of Movie Lens.

## Setup
```shell
mkdir data
curl https://files.grouplens.org/datasets/movielens/ml-latest-small.zip > data/ml-latest-small.zip
unzip -d data data/ml-latest-small.zip

pip install matplotlib tqdm
```

## Installation
```shell
pip install .
```

## Run
```shell
python train.py
```

## Result
![plot](plot-sample.png)