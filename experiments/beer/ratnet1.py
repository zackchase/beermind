from tqdm import tqdm
import sklearn
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

from dataset.sequence import CharacterSequence, NumberSequence
from dataset.encoding import OneHotEncoding, IdentityEncoding

ratings_values = np.arange(-1, 1.1, 0.1)

ratings_matrix = np.array(
    [[a] for a in ratings_values
    ]
)

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--model_dir', default='models/')
    argparser.add_argument('--data_dir', default='data/beer/')
    argparser.add_argument('--out_dir', default='out/')

    argparser.add_argument('--sentiment', action='store_true')
    argparser.add_argument('--rating', action='store_true')

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

def log_prob(ratnet, encoding, review):
    num_seq = np.vstack([encoding.convert_representation(b) for b
                         in CharacterSequence(review).encode(encoding).seq])
    length = num_seq.shape[0]
    num_seq = np.tile(num_seq, (ratings_matrix.shape[0], 1, 1))
    rating_seq = np.tile(ratings_matrix, (length, 1, 1)).swapaxes(0, 1)
    num_seq = np.dstack([num_seq, rating_seq]).swapaxes(0, 1)
    X, y = num_seq[:-1, :], num_seq[1:, :]
    y = y[:, :, :-1].swapaxes(0, 1)
    probs = ratnet.log_probability(X).swapaxes(0, 1)
    probs = probs[:, xrange(length - 1), y.argmax(axis=2)[0]].sum(axis=1)
    return probs

def estimate_rating(ratnet, encoding, review):
    probs = log_prob(ratnet, encoding, review)
    rating = ratings_matrix[probs.argsort()[::-1]][0]
    return rating

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

def transform_rating(r):
    return 2 * r + 3

def run_sentiment_analysis(ratnet, encoding, data_dir, out_dir):
    logging.info("Running experiment: sentiment")

    out_dir = out_dir / "sentiment"
    data_dir = data_dir / "experiment_files"

    if not out_dir.exists():
        out_dir.mkdir()

    positive_file = data_dir / 'positive50.txt'
    negative_file = data_dir / 'negative50.txt'
    random_file = data_dir / 'phrases.txt'

    with open(positive_file) as fp:
        positive_phrases = fp.read().strip().split('\n')
    with open(negative_file) as fp:
        negative_phrases = fp.read().strip().split('\n')
    with open(random_file) as fp:
        random_phrases = fp.read().strip().split('\n')

    final_ratings = []
    for phrase_list in [positive_phrases, negative_phrases, random_phrases]:
        ratings = []
        for phrase in phrase_list:
            logging.info("Analysing phrase: %s", phrase)
            log_probs = log_prob_sum(ratnet, encoding, phrase).T[:-1]
            means, medians, modes = [], [], []
            for l in xrange(log_probs.shape[0]):
                probs = log_probs[l]
                norm_probs = probs / probs.sum()
                means.append(np.dot(norm_probs, ratings_values))
                medians.append(ratings_values[np.arange(len(norm_probs))[np.cumsum(norm_probs) >= 0.5][0]])
                modes.append(ratings_values[norm_probs.argmax()])

            medians = map(transform_rating, medians)
            modes = map(transform_rating, modes)
            means = map(transform_rating, means)

            plt.figure()
            plt.plot(transform_rating(ratings_values), probs)
            plt.xlabel("Rating")
            plt.xlim([1, 5])
            plt.ylabel("Log Likelihood")
            plt.savefig(out_dir / "endgraph_%s.png" % phrase, bbox_inches='tight')
            ratings.append(modes[-1])

            logging.info("Final rating: %f", modes[-1])

            plt.figure()
            plt.plot(modes)
            plt.xticks(xrange(len(phrase)), list(phrase))
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.tick_params(axis='both', which='minor', labelsize=8)
            plt.xlim([0, len(phrase) - 1])
            plt.ylabel("Most Likely Rating")
            plt.ylim([0.5, 5.5])
            plt.savefig(out_dir / "phrasegraph_%s-mode.png" % phrase, bbox_inches='tight')

            plt.close()

        final_ratings.append(ratings)
    positive_ratings, negative_ratings = final_ratings[:2]

    y_true = [1] * 53 + [0] * 53

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, positive_ratings + negative_ratings)
    plt.figure()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot(fpr, tpr)
    plt.savefig(out_dir / "roc.png", bbox_inches='tight')
    plt.close()

    auc = sklearn.metrics.roc_auc_score(y_true, positive_ratings + negative_ratings)

    positive_ratings, negative_ratings = np.array(positive_ratings) > 3, np.array(negative_ratings) <= 3

    logging.info("True Positives: %u", sum(positive_ratings))
    logging.info("True Negatives: %u", sum(negative_ratings))

    logging.info("False Positives: %u", 53 - sum(positive_ratings))
    logging.info("False Negatives: %u", 53 - sum(negative_ratings))

    logging.info("p-value: " + str(binom_test(sum(positive_ratings) + sum(negative_ratings), 100)))
    logging.info("AUC: " + str(auc))


def run_rating_benchmark(ratnet, encoding, data_dir, out_dir):
    logging.info("Running experiment: rating_benchmark")

    out_dir = out_dir / "rating_benchmark"

    if not out_dir.exists():
        out_dir.mkdir()

    with open(data_dir / 'beer_core-test.pkl', 'rb') as fp:
        test_reviews, test_beers = pickle.load(fp)

    scores = []
    predictions = []
    truths = []

    filtered_reviews = {}
    indices = {}
    for i, review in tqdm(enumerate(test_reviews)):
        if review.rating_overall > 2 and review.rating_overall < 4:
            continue
        positive = review.rating_overall >= 4
        if positive not in filtered_reviews:
            indices[positive] = []
            filtered_reviews[positive] = []
        filtered_reviews[positive].append(review)
        indices[positive].append(i)

    idx = []
    reviews = []
    for vals in filtered_reviews.values():
        reviews.extend(vals[:500])
    for vals in indices.values():
        idx.extend(vals[:500])

    with open('indices-core.pkl', 'wb') as fp:
        pickle.dump(idx, fp)

    random.seed(1337)
    random.shuffle(reviews)

    for review in tqdm(reviews):
        truths.append(int(review.rating_overall >= 4))
        rating = transform_rating(estimate_rating(ratnet, encoding, review.text))
        predictions.append(int(rating >= 3.1))
        scores.append(rating)
        if truths[-1] != predictions[-1]:
            logging.info("Mistake[%u, %u]: %s (%f)", truths[-1], predictions[-1], review.text, scores[-1])
        else:
            logging.info("Correct[%u, %u]", truths[-1], predictions[-1])

    with open('ratings.pkl', 'wb') as fp:
        pickle.dump(scores, fp)
    with open('truths.pkl', 'wb') as fp:
        pickle.dump(truths, fp)

    fpr, tpr, _ = sklearn.metrics.roc_curve(truths, scores)
    plt.figure()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot(fpr, tpr)
    plt.savefig(out_dir / "roc.png", bbox_inches='tight')
    plt.close()

    auc = sklearn.metrics.roc_auc_score(truths, scores)
    logging.info("Ran on %u test reviews.", len(predictions))
    logging.info("Number of positives: %u", sum(truths))
    logging.info("Number of negatives: %u", len(truths) - sum(truths))

    logging.info("AUC: " + str(auc))

    confusion_mat = sklearn.metrics.confusion_matrix(truths, predictions)
    abbrevs = ["Positive", "Negative"]
    plt.figure()
    sns.heatmap(confusion_mat, xticklabels=abbrevs,
                               yticklabels=abbrevs,
                               cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    logging.info("Confusion Matrix: %s", str(confusion_mat))
    plt.savefig(out_dir / "confusion_matrix.png", bbox_inches='tight')
    plt.close()

def generate(ratnet, rating, text_encoding, length, temperature=1.0):
    results = ratnet.generate_with_concat(np.eye(len(text_encoding))[text_encoding.encode('<STR>')],
                                            [rating],
                                            length, temperature)
    seq = NumberSequence(results.argmax(axis=1))
    return seq.decode(text_encoding)

if __name__ == "__main__":
    args = parse_args()
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    out_dir = Path(args.out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    out_dir = out_dir / "ratnet1-2"

    if not out_dir.exists():
        out_dir.mkdir()
    logging.basicConfig(level=logging.INFO, filename=out_dir / 'experiment.txt', filemode='w')

    ratnet, encoding = load_ratnet(data_dir, model_dir)


    gen = lambda r, l, t: generate(ratnet, r, encoding, l, t)
    if args.sentiment:
        run_sentiment_analysis(ratnet, encoding, data_dir, out_dir)

    if args.rating:
        run_rating_benchmark(ratnet, encoding, data_dir, out_dir)

    logging.shutdown()
