import logging
import numpy as np
from tqdm import tqdm
import cPickle as pickle

from word2vec import train, load_model
from tokenize import tokenize_sentence, tokenize_word
from vocab import Vocabulary

class CornellMoviesDataset(object):

    def __init__(self, data_dir, word_size=100):
        self.data_dir = data_dir
        self.word_size = word_size
        self.max_seq_length = float('-inf')

        self.movies = {}
        logging.debug("Loading movie title data...")
        with open(self.data_dir / 'movie_titles_metadata.txt') as fp:
            for line in tqdm(fp):
                movie = Movie.from_line(line)
                self.movies[movie.id] = movie

        logging.debug("Loading movie lines...")
        self.lines = {}
        with open(self.data_dir / 'movie_lines.txt') as fp:
            for line in tqdm(fp):
                line = Line.from_line(line, self.movies)
                self.lines[line.id] = line
                if line.length > self.max_seq_length:
                    self.max_seq_length = line.length

        logging.debug("Loading movie conversations...")
        self.conversations = []
        with open(self.data_dir / 'movie_conversations.txt') as fp:
            for line in tqdm(fp):
                conversation = Conversation.from_line(line, self.lines)
                conversation.max_seq_length = self.max_seq_length
                self.conversations.append(conversation)

    def train_word2vec(self, model_name, force=False):
        sentences = []
        if (self.data_dir / model_name).exists() and not force:
            return Vocabulary(load_model(self.data_dir / model_name), self.word_size)
        for conversation in tqdm(self.conversations):
            for line in conversation.lines:
                text = line.raw_text.lower() + " eor"
                sentences.extend([tokenize_word(s) for s in tokenize_sentence(text)])
        w2v = train(self.data_dir, sentences, model_name, min_count=1, size=self.word_size)
        assert 'eor' in w2v.vocab, "No EOR token added to vocabulary"
        return Vocabulary(w2v, self.word_size)

    def save(self, loc):
        with open(loc, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(loc):
        with open(loc, 'rb') as fp:
            obj = pickle.load(fp)
        return obj

class Movie(object):

    def __init__(self, id, title, year, rating, genres):
        self.id = id
        self.title = title
        self.year = year
        self.rating = rating
        self.genres = genres

    @staticmethod
    def from_line(line):
        line = [m.strip() for m in line.strip().split('+++$+++')]
        assert len(line) == 6, "Improperly formatted movie: %s" % str(line)
        movie = Movie(line[0], line[1], int(line[2][:4]), float(line[3]), line[5])
        return movie

    def __repr__(self):
        return "Movie<%s (%u)>" % (self.title, self.year)

class Line(object):

    def __init__(self, id, movie, character, text):
        self.id = id
        self.movie = movie
        self.character = character
        self.raw_text = text.decode('utf-8', 'ignore')
        self.tokens = tokenize_word(self.raw_text.lower())[:1]
        self.length = len(self.tokens)

    @staticmethod
    def from_line(line, movies):
        line = [m.strip() for m in line.strip().split('+++$+++')]
        assert len(line) == 5, "Improperly formatted line: %s" % str(line)
        line = Line(line[0], movies[line[2]], line[3], line[4])
        return line

    def as_matrix(self, vocab, max_seq_length):
        tokens = self.tokens
        mat = np.zeros((max_seq_length, vocab.w2v.layer1_size))
        mask = np.zeros(max_seq_length)
        mask[:self.length] = 1
        for i, token in enumerate(tokens):
            mat[i] = vocab.w2v[token]
        return mat, mask

    def as_softmax(self, vocab, max_seq_length):
        tokens = self.tokens
        mat = np.zeros((max_seq_length, vocab.vocab_size))
        mask = np.zeros(max_seq_length)
        for i, token in enumerate(tokens):
            mat[i, vocab.forward_map[token]] = 1
            mask[i] = 1
        return mat, mask

class Conversation(object):

    def __init__(self, lines, movie):
        self.lines = lines
        self.movie = movie
        self.length = len(self.lines)
        self.characters = set([l.character for l in self.lines])
        self.max_seq_length = None

    @staticmethod
    def from_line(line, lines):
        line = [m.strip() for m in line.strip().split('+++$+++')]
        assert len(line) == 4, "Improperly formatted conversation: %s" % str(line)
        lines = [lines[l] for l in eval(line[3])]
        return Conversation(lines, lines[0].movie)

    def __str__(self):
        return "\n".join(
            ["%s: %s" % (l.character, l.raw_text) for l in self.lines]
        )

    def as_sequence(self, vocab):
        assert self.max_seq_length is not None, "Set max_seq_length please"
        mat = np.zeros((self.length, self.max_seq_length, vocab.w2v.layer1_size))
        mask = np.zeros((self.length, self.max_seq_length))
        for i, line in enumerate(self.lines):
            mat[i], mask[i] = line.as_matrix(vocab, self.max_seq_length)
        return mat, mask
