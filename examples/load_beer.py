import theano.tensor as T
import logging
logging.basicConfig(level=logging.DEBUG)
from dataset.sequence import *
from dataset.batch import WindowedBatcher

import dataset
from deepx.optimize import SGD, RMSProp, Momentum
from deepx.sequence import CharacterRNN

reviews, beers = dataset.beer.load_data('data/beer')

text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]
review_users = [SingletonSequence(review.user) for review in reviews]
review_beers = [SingletonSequence(review.beer.id) for review in reviews]

text_encoding = dataset.OneHotEncoding(include_start_token=True)
text_encoding.build_encoding(text_sequences)

num_seq = NumberSequence(np.concatenate([c.encode(text_encoding).seq for c in text_sequences]).ravel())
batcher = WindowedBatcher(num_seq, text_encoding, sequence_length=50, batch_size=500)

D = text_encoding.index

charrnn = CharacterRNN('2pac', text_encoding, n_layers=2, n_hidden=512)
charrnn.load_parameters('wat.pkl')
charrnn.compile_method('generate')

sgd = SGD(charrnn)
rmsprop = RMSProp(charrnn)
mom = Momentum(charrnn)

def train(optimizer, n_iterations, *args):
    state = None
    for i in xrange(n_iterations):
        X, y = batcher.next_batch()
        if state is None:
            state = np.zeros((X.shape[1], charrnn.n_layers, charrnn.n_hidden))
        error, state = optimizer.train(X, state, y, *args)
        state = state[-1]
        print "Iteration %u:" % (i + 1), error

def generate(length, temperature=1.0):
    results = charrnn.generate(np.eye(D)[0], length, temperature)
    seq = NumberSequence(results.argmax(axis=1))
    return seq.decode(text_encoding)

