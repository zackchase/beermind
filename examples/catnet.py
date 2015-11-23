import seaborn as sns
sns.set_style('whitegrid')
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from dataset.sequence import NumberSequence, CharacterSequence, SingletonSequence
from dataset.batch import WindowedBatcher
import cPickle as pickle

import dataset
from deepx.optimize import RMSProp
from deepx.sequence import CharacterRNN
from deepx.train import Trainer

with open('data/beer/beer_top.pkl', 'rb') as fp:
    reviews, beers = pickle.load(fp)

text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]

text_encoding = dataset.OneHotEncoding(include_start_token=True,
                                       include_stop_token=True)
text_encoding.build_encoding(text_sequences)

beer_cats = [SingletonSequence(review.beer.style) for review in reviews]

cat_encoding = dataset.OneHotEncoding(include_start_token=False,
                                        include_stop_token=False)
cat_encoding.build_encoding(beer_cats)

with open('data/beer/beer_top-train.pkl', 'rb') as fp:
    reviews, beers = pickle.load(fp)

text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]
beer_cats = [SingletonSequence(review.beer.style) for review in reviews]

review_num_seqs = [c.encode(text_encoding) for c in text_sequences]

num_seq = NumberSequence(np.concatenate([c.seq for c in review_num_seqs]))
beer_seq = NumberSequence(np.concatenate([c.encode(cat_encoding).replicate(len(r)).seq for c, r in
                                          zip(beer_cats, review_num_seqs)]))
batcher = WindowedBatcher([num_seq, beer_seq], [text_encoding, cat_encoding],
                          sequence_length=200, batch_size=256)

catnet = CharacterRNN('2pac', len(text_encoding) + len(cat_encoding),
                       len(text_encoding), n_layers=2, n_hidden=1024)
catnet.compile_method("generate_with_concat")

def load_charnet():
    catnet.load_parameters('models/charnet-top_2-1024-2.pkl')
    layer = catnet.lstm.input_layer

    weights = {
        'W_ix': layer.get_parameter_value("W_ix"),
        'W_ox': layer.get_parameter_value("W_ox"),
        'W_fx': layer.get_parameter_value("W_fx"),
        'W_gx': layer.get_parameter_value("W_gx"),
    }

    for w, value in weights.items():
        layer.set_parameter_value(w, np.vstack([value,
                                                np.zeros((len(cat_encoding), catnet.n_hidden)).astype(
                                                    np.float32
                                                )]))

rmsprop = RMSProp(catnet)

trainer = Trainer(rmsprop, batcher, learning_curve="out/stylenet/learning_curve.png")

def generate(beer_cat, length, temperature=1.0):
    results = catnet.generate_with_concat(np.eye(len(text_encoding))[text_encoding.encode('<STR>')],
                                          np.eye(len(cat_encoding))[cat_encoding.encode(beer_cat)],
                                          length, temperature)
    seq = NumberSequence(results.argmax(axis=1))
    return seq.decode(text_encoding)
