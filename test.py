import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from path import Path
from deepx.util import movies
from deepx.sequence import ConversationModel

word_size = 100
data_dir = Path('data/movies')

dataset = movies.CornellMoviesDataset(data_dir, word_size=word_size)

vocab = dataset.train_word2vec('test.w2v')

cons = ConversationModel('convmodel', vocab, 10, num_layers=1)
cons.compile()

lines = dataset.conversations[1].lines
in_line = lines[0].as_matrix(vocab)
out_line = lines[1].as_softmax(vocab)

in_line = in_line[:, np.newaxis]
out_line = out_line[:, np.newaxis]

start_token = np.zeros(vocab.vocab_size)
start_token[vocab.forward_map['eor']] = 1
start_token = np.tile(start_token, (1, 1))
