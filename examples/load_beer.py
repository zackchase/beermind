import theano.tensor as T
import logging
logging.basicConfig(level=logging.DEBUG)
from dataset.sequence import *
from dataset.batch import WindowedBatcher

import dataset
from deepx.optimize import SGD, RMSProp
from deepx.sequence import CharacterRNN

reviews, beers = dataset.beer.load_data('data/beer')

review_text = ' '.join([review.text for review in reviews])
review_users = [SingletonSequence(review.user) for review in reviews]
review_beers = [SingletonSequence(review.beer.id) for review in reviews]

char_seq = CharacterSequence.from_string(review_text)
text_encoding = dataset.OneHotEncoding()
text_encoding.build_encoding([char_seq])

num_seq = char_seq.encode(text_encoding)
batcher = WindowedBatcher(num_seq, text_encoding, sequence_length=50, batch_size=500)

charrnn = CharacterRNN('2pac', text_encoding, n_layers=2, n_hidden=512)

optimizer = SGD(charrnn, [T.tensor3('X'), T.tensor3('state'), T.tensor3('y')])

state = None
for _ in xrange(100):
    X, y = batcher.next_batch()
    if state is None:
        state = np.zeros((X.shape[1], charrnn.n_layers, charrnn.n_hidden))
    error, state = optimizer.optimize(X, state, y)
    state = state[-1]
    print error
