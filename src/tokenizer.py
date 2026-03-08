class Tokenizer:
    def __init__(self, max_len=128):
        self.max_len = max_len

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
    