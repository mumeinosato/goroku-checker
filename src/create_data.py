'''
dataset/raw/positive.txtとnegative.txtをランダムにシャッフルしてtrain.csvを作成する
train.csvは以下のような形式で作成する
sentence,label
'''

import pandas as pd

def create_train_csv():
    with open('dataset/raw/positive.txt', 'r', encoding='utf-8') as f:
        positive = f.read().splitlines()
    with open('dataset/raw/negative.txt', 'r', encoding='utf-8') as f:
        negative = f.read().splitlines()

    data = []
    for sentence in positive:
        data.append({'sentence': sentence, 'label': 1})
    for sentence in negative:
        data.append({'sentence': sentence, 'label': 0})

    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # シャッフル
    df.to_csv('dataset/train.csv', index=False, encoding='utf-8')