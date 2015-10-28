from nltk import word_tokenize
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from dataset.sequence import NumberSequence, CharacterSequence, SingletonSequence
from dataset.batch import WindowedBatcher
from nltk.stem.porter import PorterStemmer

import dataset
from deepx.optimize import SGD, RMSProp, Momentum
from deepx.sequence import CharacterRNN

reviews, beers = dataset.beer.load_data('data/beer')

text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]
beer_styles = [SingletonSequence(review.beer.style) for review in reviews]

text_encoding = dataset.OneHotEncoding(include_start_token=True,
                                       include_stop_token=True)
text_encoding.build_encoding(text_sequences)

style_encoding = dataset.OneHotEncoding(include_start_token=False,
                                        include_stop_token=False)
style_encoding.build_encoding(beer_styles)

review_num_seqs = [c.encode(text_encoding) for c in text_sequences]
review_styles = [r.stack(c.encode(style_encoding).replicate(len(r))) for c, r in zip(beer_styles, review_num_seqs)]

# num_seq = NumberSequence(np.concatenate([c.seq for c in review_styles]))
num_seq = NumberSequence(np.concatenate([c.seq for c in review_num_seqs]))
style_seq = NumberSequence(np.concatenate([c.encode(style_encoding).replicate(len(r)).seq for c, r in zip(beer_styles, review_num_seqs)]))
# batcher = WindowedBatcher(num_seq, [text_encoding, style_encoding], sequence_length=200, batch_size=500)
batcher = WindowedBatcher([num_seq, style_seq], [text_encoding, style_encoding], sequence_length=200, batch_size=500)
# batcher = WindowedBatcher(num_seq, [text_encoding], sequence_length=200, batch_size=500)

D = text_encoding.index

# charrnn = CharacterRNN('2pac', len(text_encoding) + len(style_encoding), len(text_encoding), n_layers=2, n_hidden=512)
# charrnn = CharacterRNN('2pac', len(text_encoding), len(text_encoding), n_layers=2, n_hidden=1024)
charrnn = CharacterRNN('2pac', len(text_encoding) + len(style_encoding), len(text_encoding), n_layers=2, n_hidden=512)
charrnn.load_parameters('beer_model-2-512.pkl')
charrnn.compile_method('generate_with_concat')

# sgd = SGD(charrnn)
rmsprop = RMSProp(charrnn)
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

def generate(beer_name, length, temperature=1.0):
    results = charrnn.generate_with_concat(np.eye(D)[text_encoding.encode('<STR>')], np.eye(len(style_encoding))[style_encoding.encode('American Double / Imperial IPA')], length, temperature)
    seq = NumberSequence(results.argmax(axis=1))
    return seq.decode(text_encoding)

# review_length = 10000
# print "Building IPA review..."
# ipa_review = generate('American IPA', review_length, temperature=0.5).seq
# print "Building Stout review..."
# stout_review = generate('American Double / Imperial Stout', review_length, temperature=0.5).seq

stemmer = PorterStemmer()

def build_histogram(review):
    review = review.lower()
    review = review.replace("<STR>", " ")
    review = review.replace("<EOS>", " ")

    freqs = {}
    for word in word_tokenize(review):
        word = stemmer.stem(word)
        if word not in freqs:
            freqs[word] = 0
        freqs[word] += 1
    total = sum(freqs.values())
    for word in freqs:
        freqs[word] /= float(total)
    return freqs
# ipa_hist = build_histogram(ipa_review)
# stout_hist = build_histogram(stout_review)
