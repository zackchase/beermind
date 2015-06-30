import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from path import Path
from deepx.util import movies
from deepx.sequence import ConversationModel

word_size = 100
data_dir = Path('data/movies')

dataset = movies.CornellMoviesDataset(data_dir, word_size=word_size)

vocab = dataset.train_word2vec('test.w2v')

cons = ConversationModel('convmodel', vocab, 10, num_layers=1).compile()

out, state = cons.encode(dataset.conversations[0].lines[0].as_matrix(vocab)[:, np.newaxis],
                         np.zeros((1, 1, 10)),
                         np.zeros((1, 1, 10)))
