from nltk import word_tokenize
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from dataset.sequence import NumberSequence, CharacterSequence, SingletonSequence
from dataset.batch import WindowedBatcher
from nltk.stem.porter import PorterStemmer
import cPickle as pickle

import dataset
from deepx.optimize import SGD, RMSProp, Momentum
from deepx.sequence import CharacterRNN
from deepx.train import Trainer

with open('data/beer/beer_core.pkl', 'rb') as fp:
    reviews, beers = pickle.load(fp)

user_counts = {}
for review in reviews:
    if review.user:
        if review.user not in user_counts:
            user_counts[review.user] = 0
        user_counts[review.user] += 1

top_users = sorted(user_counts.items(), key=lambda x: -x[1])

text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]
beer_users = [SingletonSequence(review.user) for review in reviews]

text_encoding = dataset.OneHotEncoding(include_start_token=True,
                                       include_stop_token=True)
text_encoding.build_encoding(text_sequences)

user_encoding = dataset.OneHotEncoding(include_start_token=False,
                                        include_stop_token=False)
user_encoding.build_encoding(beer_users)

review_num_seqs = [c.encode(text_encoding) for c in text_sequences]
review_users = [r.stack(c.encode(user_encoding).replicate(len(r))) for c, r in zip(beer_users, review_num_seqs)]

# num_seq = NumberSequence(np.concatenate([c.seq for c in review_styles]))
num_seq = NumberSequence(np.concatenate([c.seq for c in review_num_seqs]))
user_seq = NumberSequence(np.concatenate([c.encode(user_encoding).replicate(len(r)).seq for c, r in zip(beer_users, review_num_seqs)]))
# batcher = WindowedBatcher(num_seq, [text_encoding, user_encoding], sequence_length=200, batch_size=500)
batcher = WindowedBatcher([num_seq, user_seq], [text_encoding, user_encoding], sequence_length=200, batch_size=256)
# batcher = WindowedBatcher(num_seq, [text_encoding], sequence_length=200, batch_size=500)

D = text_encoding.index

# charrnn = CharacterRNN('2pac', len(text_encoding) + len(user_encoding), len(text_encoding), n_layers=2, n_hidden=512)
# charrnn = CharacterRNN('2pac', len(text_encoding), len(text_encoding), n_layers=2, n_hidden=1024)
charrnn = CharacterRNN('2pac', len(text_encoding) + len(user_encoding), len(text_encoding), n_layers=2, n_hidden=1024)
# charrnn.compile_method('generate_with_concat')

# sgd = SGD(charrnn)
rmsprop = RMSProp(charrnn)
# mom = Momentum(charrnn)

trainer = Trainer(rmsprop, batcher)

def generate(beer_user, length, temperature=1.0):
    results = charrnn.generate_with_concat(np.eye(D)[text_encoding.encode('<STR>')], np.eye(len(user_encoding))[user_encoding.encode(beer_user)], length, temperature)
    seq = NumberSequence(results.argmax(axis=1))
    return seq.decode(text_encoding)
