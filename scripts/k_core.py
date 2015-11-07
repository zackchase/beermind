import cPickle as pickle
import networkx as nx
import dataset

import logging
logging.basicConfig(level=logging.DEBUG)

reviews, beers = dataset.beer.load_data('data/beer', file='beer.json', cache_file='beer_full.pkl')

def get_review_name(review):
    return 'review_%s' % review.user

def get_beer_name(beer):
    return 'beer_%u' % beer.id

graph = nx.Graph()
review_map = {}
for review in reviews:
    graph.add_edge(get_review_name(review), get_beer_name(review.beer))
    review_map[(get_review_name(review), get_beer_name(review.beer))] = review


def get_core_reviews(k):
    core = nx.algorithms.core.k_core(graph, k=k)
    core_reviews = []
    for edge in core.edges_iter():
        (a, b) = edge
        if edge not in review_map:
            core_reviews.append(review_map[(b, a)])
        else:
            core_reviews.append(review_map[edge])
    return core_reviews

core_reviews = get_core_reviews(190)

import random
random.seed(1337)
random.shuffle(core_reviews)

test_length = int(0.1 * len(core_reviews))

train, test = core_reviews[:-test_length], core_reviews[-test_length:]

with open('data/beer/beer_core-test.pkl', 'wb') as fp:
    pickle.dump((test, beers), fp)
with open('data/beer/beer_core-train.pkl', 'wb') as fp:
    pickle.dump((train, beers), fp)
