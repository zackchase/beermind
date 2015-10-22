from sequence import NumberSequence, WordSequence

class Encoding(object):

    in_sequence_type = None
    out_sequence_type = None

    def build_encoding(self, sequences):
        raise NotImplementedError

    def encode(self, obj):
        raise NotImplementedError

    def decode(self, obj):
        raise NotImplementedError

    def encode_sequence(self, sequence):
        return self.out_sequence_type(
            [self.encode(s) for s in sequence.iter()]
        )

    def decode_sequence(self, sequence):
        return self.in_sequence_type(
            [self.decode(s) for s in sequence.iter()]
        )

class OneHotEncoding(Encoding):

    out_sequence_type = NumberSequence

    def __init__(self):
        self.forward_mapping = {}
        self.backward_mapping = []
        self.index = 0
        self.in_sequence_type = None

    def build_encoding(self, sequences):
        for sequence in sequences:
            self.in_sequence_type = sequence.__class__
            for word in sequence.iter():
                if word not in self.forward_mapping:
                    self.forward_mapping[word] = self.index
                    self.backward_mapping.append(word)
                    self.index += 1

    def encode(self, word):
        if word not in self.forward_mapping:
            assert word in self.forward_mapping, "'%s' not in one-hot mapping" % word
        return self.forward_mapping[word]

    def decode(self, index):
        return self.backward_mapping[index]
