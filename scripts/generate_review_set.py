import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from tqdm import tqdm
from dataset.sequence import CharacterSequence, SingletonSequence, NumberSequence
import cPickle as pickle

import dataset
from deepx.sequence import CharacterRNN

REAL_FILE = "real.txt"
GEN_FILE = "gen.txt"

def generate(beer_cat, length, num_examples=100, temperature=0.6):
    results = catnet.generate_examples(np.eye(len(text_encoding))[text_encoding.encode('<STR>')],
                                          np.eye(len(cat_encoding))[cat_encoding.encode(beer_cat)],
                                          length,
                                          num_examples,
                                          temperature).argmax(axis=2)
    reviews = []
    for i in xrange(results.shape[1]):
        reviews.append(str(NumberSequence(results[:, i]).decode(text_encoding)).split('<EOS>')[0])
    return reviews

if __name__  == "__main__":
    with open('data/beer/beer_top.pkl') as fp:
        reviews, beers = pickle.load(fp)

    text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]

    text_encoding = dataset.OneHotEncoding(include_start_token=True,
                                        include_stop_token=True)
    text_encoding.build_encoding(text_sequences)

    beer_cats = [SingletonSequence(review.beer.style) for review in reviews]

    cat_encoding = dataset.OneHotEncoding(include_start_token=False,
                                            include_stop_token=False)
    cat_encoding.build_encoding(beer_cats)

    catnet = CharacterRNN('2pac', len(text_encoding) + len(cat_encoding),
                       len(text_encoding), n_layers=2, n_hidden=1024)
    catnet.load_parameters('models/catnet_2-1024-2.pkl')
    catnet.compile_method('generate_examples')

    with open(REAL_FILE, 'w') as fp:
        for review in reviews:
            print >>fp, "%u: %s" % (cat_encoding.encode(review.beer.style), review.text)
    with open(GEN_FILE, 'w') as fp:
        for i, beer in enumerate(cat_encoding.backward_mapping):
            logging.info("Generating %s reviews", beer)
            for _ in tqdm(xrange(300)):
                gen_reviews = generate(beer, 2000)
                for review in gen_reviews:
                    print >>fp, "%u: %s" % (i, review)
