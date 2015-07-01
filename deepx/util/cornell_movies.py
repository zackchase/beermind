import numpy as np
from tqdm import tqdm

from word2vec import train, load_model
from tokenize import tokenize_sentence, tokenize_word
from vocab import Vocabulary

class CornellMoviesDataset(object):

    def __init__(self, data_dir, word_size=100):
        self.data_dir = data_dir
        self.word_size = word_size

        self.movies = {}
        with open(self.data_dir / 'movie_titles_metadata.txt') as fp:
            for line in fp:
                movie = Movie.from_line(line)
                self.movies[movie.id] = movie

        self.lines = {}
        with open(self.data_dir / 'movie_lines.txt') as fp:
            for line in fp:
                line = Line.from_line(line, self.movies)
                self.lines[line.id] = line

        self.conversations = []
        with open(self.data_dir / 'movie_conversations.txt') as fp:
            for line in fp:
                conversation = Conversation.from_line(line, self.lines)
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

    @staticmethod
    def from_line(line, movies):
        line = [m.strip() for m in line.strip().split('+++$+++')]
        assert len(line) == 5, "Improperly formatted line: %s" % str(line)
        line = Line(line[0], movies[line[2]], line[3], line[4])
        return line

    def as_matrix(self, vocab):
        tokens = tokenize_word(self.raw_text.lower())
        mat = np.zeros((len(tokens), vocab.w2v.layer1_size))
        for i, token in enumerate(tokens):
            mat[i] = vocab.w2v[token]
        return mat

    def as_softmax(self, vocab):
        tokens = tokenize_word(self.raw_text.lower())
        mat = np.zeros((len(tokens), vocab.vocab_size))
        for i, token in enumerate(tokens):
            mat[i, vocab.forward_map[token]] = 1
        return mat

class Conversation(object):

    def __init__(self, lines, movie):
        self.lines = lines
        self.movie = movie
        self.length = len(self.lines)
        self.characters = set([l.character for l in self.lines])

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
        return [l.as_matrix(vocab) for l in self.lines]
