from scipy.stats import binom_test
import numpy as np
import logging
import cPickle as pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from argparse import ArgumentParser
from path import Path

from deepx.sequence import CharacterRNN

from dataset.sequence import CharacterSequence
from dataset.encoding import OneHotEncoding, IdentityEncoding

ratings_values = np.arange(0.2, 1.001, 0.001)

ratings_matrix = np.array(
    [[a] for a in ratings_values
    ]
)

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--model_dir', default='models/')
    argparser.add_argument('--data_dir', default='data/beer/')
    argparser.add_argument('--out_dir', default='out/')

    return argparser.parse_args()

def load_ratnet(data_dir, model_dir):
    logging.info("Loading data from cache...")
    with open(data_dir / 'beer_core-train.pkl', 'rb') as fp:
        reviews, beers = pickle.load(fp)

    random.seed(1337)
    random.shuffle(reviews)

    logging.info("Loading text sequences...")
    text_sequences = [CharacterSequence.from_string(review.text) for review in reviews]

    text_encoding = OneHotEncoding(include_start_token=True,
                                   include_stop_token=True)
    text_encoding.build_encoding(text_sequences)

    identity_encoding = IdentityEncoding(1)

    logging.info("Loading model...")
    ratnet = CharacterRNN('ratnet', len(text_encoding) + len(identity_encoding), len(text_encoding), n_layers=2, n_hidden=1024)
    ratnet.load_parameters(model_dir / 'ratnet1_2-1024.pkl')
    ratnet.compile_method('generate_with_concat')
    ratnet.compile_method('log_probability')
    return ratnet, text_encoding

def main(args):
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    out_dir = Path(args.out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    out_dir = out_dir / "ratnet1"

    if not out_dir.exists():
        out_dir.mkdir()
    logging.basicConfig(level=logging.INFO, filename=out_dir / 'experiment.txt', filemode='w')

    ratnet, encoding = load_ratnet(data_dir, model_dir)

    run_sentiment_analysis(ratnet, encoding, data_dir, out_dir)

def log_prob_sum(ratnet, encoding, review):
    num_seq = np.vstack([encoding.convert_representation(b) for b
                         in CharacterSequence(review).encode(encoding).seq])
    length = num_seq.shape[0]
    num_seq = np.tile(num_seq, (ratings_matrix.shape[0], 1, 1))
    rating_seq = np.tile(ratings_matrix, (length, 1, 1)).swapaxes(0, 1)
    num_seq = np.dstack([num_seq, rating_seq]).swapaxes(0, 1)
    X, y = num_seq[:-1, :], num_seq[1:, :]
    y = y[:, :, :-1].swapaxes(0, 1)
    probs = ratnet.log_probability(X).swapaxes(0, 1)
    probs = probs[:, xrange(length - 1), y.argmax(axis=2)[0]].cumsum(axis=1)
    return probs

def run_sentiment_analysis(ratnet, encoding, data_dir, out_dir):
    logging.info("Running experiment: sentiment")

    out_dir = out_dir / "sentiment"
    data_dir = data_dir / "experiment_files"

    if not out_dir.exists():
        out_dir.mkdir()

    positive_file = data_dir / 'positive50.txt'
    negative_file = data_dir / 'negative50.txt'

    with open(positive_file) as fp:
        positive_phrases = fp.read().strip().split('\n')
    with open(negative_file) as fp:
        negative_phrases = fp.read().strip().split('\n')

    final_ratings = []
    for phrase_list in [positive_phrases, negative_phrases]:
        ratings = []
        for phrase in phrase_list:
            logging.info("Analysing phrase: %s", phrase)
            log_probs = log_prob_sum(ratnet, encoding, phrase).T[:-1]
            means, medians, modes = [], [], []
            for l in xrange(log_probs.shape[0]):
                probs = log_probs[l]
                norm_probs = probs / probs.sum()
                means.append(np.dot(norm_probs, ratings_values) * 5)
                medians.append(ratings_values[np.arange(len(norm_probs))[np.cumsum(norm_probs) >= 0.5][0]] * 5)
                modes.append(ratings_values[norm_probs.argmax()] * 5)

            plt.figure()
            plt.plot(ratings_values * 5, probs)
            plt.xlabel("Rating")
            plt.ylabel("Log Likelihood")
            plt.savefig(out_dir / "endgraph_%s.png" % phrase)
            ratings.append(modes[-1])

            logging.info("Final rating: %f", modes[-1])

            plt.figure()
            plt.plot(modes)
            plt.xticks(xrange(len(phrase)), list(phrase))
            plt.ylabel("Rating")
            plt.ylim([0.5, 5.5])
            plt.savefig(out_dir / "phrasegraph_%s-mode.png" % phrase)

        final_ratings.append(ratings)
    positive_ratings, negative_ratings = final_ratings
    positive_ratings, negative_ratings = np.array(positive_ratings) > 3, np.array(negative_ratings) <= 3

    logging.info("True Positives: %u", sum(positive_ratings))
    logging.info("True Negatives: %u", sum(negative_ratings))

    logging.info("False Positives: %u", 50 - sum(positive_ratings))
    logging.info("False Negatives: %u", 50 - sum(negative_ratings))

    logging.info("p-value: " + str(binom_test(sum(positive_ratings) + sum(negative_ratings), 100)))

    logging.shutdown()

if __name__ == "__main__":
    main(parse_args())
