import numpy as np
from nltk import stem
from nltk import word_tokenize

snowball = stem.snowball.EnglishStemmer()

class Sequence(object):

    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def encode(self, encoding):
        return encoding.encode_sequence(self)

    def decode(self, encoding):
        return encoding.decode_sequence(self)

    def iter(self):
        return iter(self.seq)

    def __str__(self):
        return str(self.seq)

class SingletonSequence(Sequence):

    def __init__(self, obj):
        self.seq = [obj]

class NumberSequence(Sequence):

    def __init__(self, seq):
        self.seq = np.array(seq)

    def concatenate(self, num_seq):
        return NumberSequence(np.concatenate([self.seq, num_seq.seq]))

class WordSequence(Sequence):

    def __init__(self, seq):
        self.seq = seq

    @staticmethod
    def from_string(string):
        return WordSequence([w.lower() for w in word_tokenize(string)])

class CharacterSequence(Sequence):

    def __init__(self, seq):
        self.seq = seq

    def concatenate(self, char_seq):
        return CharacterSequence(self.seq + char_seq.seq)

    @staticmethod
    def from_string(string):
        return CharacterSequence(list(string.lower()))

    def __str__(self):
        return ''.join(self.seq)
