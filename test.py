import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from path import Path
from deepx.util import movies

data_dir = Path('data/movies')

dataset = movies.CornellMoviesDataset(data_dir)

w2v = dataset.train_word2vec('test.w2v')

print "Conversation"
print dataset.conversations[0]

print "First line converted to matrices"
mat = dataset.conversations[0].lines[0].as_matrix(w2v)
print mat
print mat.shape
