from tqdm import tqdm
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from path import Path
from deepx.util import movies
from deepx.sequence import ConversationModel

import sys
sys.setrecursionlimit(100000)

word_size = 100
data_dir = Path('data/movies')

wat = Path('wat.pkl')

if not wat.exists():
    dataset = movies.CornellMoviesDataset(data_dir, word_size=word_size)
    dataset.save('wat.pkl')
else:
    dataset = movies.CornellMoviesDataset.load('wat.pkl')

vocab = dataset.train_word2vec('test.w2v')


lines = dataset.conversations[1].lines
in_line, in_mask = lines[0].as_matrix(vocab, dataset.max_seq_length)
in_line, in_mask = in_line[np.newaxis, :], in_mask[np.newaxis, :]
out_line, out_mask = lines[1].as_softmax(vocab, dataset.max_seq_length)
out_line, out_mask = out_line[np.newaxis, :], out_mask[np.newaxis, :]

start_token = np.tile(vocab.w2v['eor'], (1, 1))

cons = ConversationModel('convmodel', vocab, 5, dataset.max_seq_length, num_layers=1)
cons.compile()

def iterate(iterations):
    for i in tqdm(xrange(iterations)):
        cons.rmsprop2(in_line, in_mask, out_line, out_mask, start_token)
    return cons.loss2(in_line, in_mask, out_line, out_mask, start_token)

iterate(100)

#in_line = in_line[:, np.newaxis]
#out_line = out_line[:, np.newaxis]

#start_token = np.zeros(vocab.vocab_size)
#start_token[vocab.forward_map['eor']] = 1
#start_token = np.tile(start_token, (1, 1))in_line, in_mask, out_line, out_mask, start_token
