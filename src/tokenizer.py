import json

class Tokenizer:
    def __init__(self, config=None):
        if config is None:
            self.max_len = 128
        else:
            self.max_len = config.get('max_len', 128)

    def tokenize(self, sentence):
        tokens = list(sentence)
        return tokens[:self.max_len]

    def build_vocab(self, dataset):
        self.vocab = set()
        for sentence in dataset:
            tokens = self.tokenize(sentence)
            for token in tokens:
                self.vocab.add(token)
        self.word2idx = {word: idx + 1 for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word2idx[0] = '<PAD>'
        
    def encode(self, sentence):
        tokens = self.tokenize(sentence)
        idxes = [self.word2idx.get(token, 0) for token in tokens]
        idxes = idxes + [0] * (self.max_len - len(idxes))
        return idxes

    def decode(self, indices):
        return ''.join([self.idx2word.get(idx, '') for idx in indices])
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)