import pandas as pd
import logging
from src.tokenizer import Tokenizer
from model import Model

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

dataset_path = 'dataset/train.csv'

t = Tokenizer()

#語彙を構築
train_data = pd.read_csv(dataset_path)
train_data = train_data['sentence'].tolist()
t.build_vocab(train_data)

vocab_size = len(t.word2idx) + 1
model = Model(vocab_size)

#テスト
test_sentence = "うせやろ？"
encoded = t.encode(test_sentence)
decoded = t.decode(encoded)

assert test_sentence[:t.max_len] == decoded, "Not match after encoding and decoding"