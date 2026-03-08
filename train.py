import pandas as pd
import logging
from src.tokenizer import tokenizer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

dataset_path = 'dataset/train.csv'

t = tokenizer()

#語彙を構築
train_data = pd.read_csv(dataset_path)
train_data = train_data['sentence'].tolist()
t.build_vocab(train_data)

logging.debug(f"語彙サイズ: {len(t.vocab)}")
logging.debug(f"word2idx のサンプル: {list(t.word2idx.items())[:10]}")
logging.debug(f"idx2word のサンプル: {list(t.idx2word.items())[:10]}")


test_sentence = "うせやろ？"
encoded = t.encode(test_sentence)
logging.debug(f"エンコードされた結果: {encoded}")

decoded = t.decode(encoded)
logging.debug(f"デコードされた結果: {decoded}")

assert test_sentence[:t.max_len] == decoded, "Not match after encoding and decoding"