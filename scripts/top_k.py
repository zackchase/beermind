import cPickle as pickle
import networkx as nx
import dataset
from collections import Counter

import logging
logging.basicConfig(level=logging.DEBUG)

reviews, beers = dataset.beer.load_data('data/beer', file='beer.json', cache_file='beer_full.pkl')

def get_review_name(review):
    return 'review_%s' % review.user

def get_beer_name(beer):
    return 'beer_%u' % beer.id

c = Counter()
for review in reviews:
    c[review.beer.style] += 1

k = 20

top_beers = set([
    'American IPA',
    'Russian Imperial Stout',
    'American Porter',
    'Fruit / Vegetable Beer',
    'American Adjunct Lager'
])


core_reviews = {}
for review in reviews:
    if review.beer.style in top_beers:
        if review.beer.style not in core_reviews:
            core_reviews[review.beer.style] = []
        core_reviews[review.beer.style].append(review)

import random

final_reviews = []


for beer_type, reviews in core_reviews.iteritems():
    random.seed(1337)
    random.shuffle(reviews)
    reviews = reviews[:30000]
    final_reviews.extend(reviews)

random.seed(1337)
random.shuffle(final_reviews)
test_length = int(0.1 * len(final_reviews))

train, test = final_reviews[:-test_length], final_reviews[-test_length:]

with open('data/beer/beer_top.pkl', 'wb') as fp:
    pickle.dump((final_reviews, beers), fp)
with open('data/beer/beer_top-test.pkl', 'wb') as fp:
    pickle.dump((test, beers), fp)
with open('data/beer/beer_top-train.pkl', 'wb') as fp:
    pickle.dump((train, beers), fp)
