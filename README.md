# Feature Embedding

## Setup
```shell
mkdir data
curl https://files.grouplens.org/datasets/movielens/ml-latest-small.zip > data/ml-latest-small.zip
unzip -d data data/ml-latest-small.zip

pip install .
pip install matplotlib
```

## Run
```shell
python train.py
```

## Result
![plot](plot-sample.png)