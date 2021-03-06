import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from dataset.sequence import NumberSequence, CharacterSequence, SingletonSequence
from dataset.batch import WindowedBatcher

import dataset
from deepx.optimize import SGD, RMSProp, Momentum
from deepx.sequence import CharacterRNN

reviews, beers = dataset.beer.load_data('data/beer')

text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]
review_users = [SingletonSequence(review.user) for review in reviews]
beer_styles = [SingletonSequence(review.beer.style) for review in reviews]
beer_ratings = [NumberSequence([[float(r) / 5.0 for r in review.ratings]]) for review in reviews]

text_encoding = dataset.OneHotEncoding(include_start_token=True,
                                       include_stop_token=True)
text_encoding.build_encoding(text_sequences)

style_encoding = dataset.OneHotEncoding(include_start_token=False,
                                        include_stop_token=False)
style_encoding.build_encoding(beer_styles)
identity_encoding = dataset.IdentityEncoding(5)

review_num_seqs = [c.encode(text_encoding) for c in text_sequences]
review_styles = [r.stack(c.encode(style_encoding).replicate(len(r))) for c, r in zip(beer_styles, review_num_seqs)]
review_ratings = [r.stack(c.replicate(len(r))) for c, r in zip(beer_ratings, review_num_seqs)]

# num_seq = NumberSequence(np.concatenate([c.seq for c in review_styles]))
num_seq = NumberSequence(np.concatenate([c.seq for c in review_num_seqs]))
beer_seq = NumberSequence(np.concatenate([c.replicate(len(r)).seq for c, r in zip(beer_ratings, review_num_seqs)]))
# batcher = WindowedBatcher(num_seq, [text_encoding, style_encoding], sequence_length=200, batch_size=500)
batcher = WindowedBatcher([num_seq, beer_seq], [text_encoding, identity_encoding], sequence_length=200, batch_size=500)
# batcher = WindowedBatcher(num_seq, [text_encoding], sequence_length=200, batch_size=500)

D = text_encoding.index

# charrnn = CharacterRNN('2pac', len(text_encoding) + len(style_encoding), len(text_encoding), n_layers=2, n_hidden=512)
# charrnn = CharacterRNN('2pac', len(text_encoding), len(text_encoding), n_layers=2, n_hidden=1024)
charrnn = CharacterRNN('2pac', len(text_encoding) + len(identity_encoding), len(text_encoding), n_layers=2, n_hidden=512)
# charrnn.compile_method('generate')

# sgd = SGD(charrnn)
# rmsprop = RMSProp(charrnn)
# mom = Momentum(charrnn)

def train(optimizer, n_iterations, *args):
    state = None
    for i in xrange(n_iterations):
        X, y = batcher.next_batch()
        if state is None:
            state = np.zeros((X.shape[1], charrnn.n_layers, charrnn.n_hidden))
        error, state = optimizer.train(X, state, y, *args)
        state = state[-1]
        print "Iteration %u (%u):" % (i + 1, batcher.batch_index - 1), error

def generate(length, temperature=1.0):
    results = charrnn.generate(np.eye(D)[text_encoding.encode('<STR>')], length, temperature)
    seq = NumberSequence(results.argmax(axis=1))
    return seq.decode(text_encoding)

def generate_with_concat(beer_name, length, temperature=1.0):
    results = charrnn.generate_with_concat(np.eye(D)[text_encoding.encode('<STR>')], np.eye(len(style_encoding))[style_encoding.encode('American Double / Imperial IPA')], length, temperature)
    seq = NumberSequence(results.argmax(axis=1))
    return seq.decode(text_encoding)

