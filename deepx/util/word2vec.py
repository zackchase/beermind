from gensim.models.word2vec import Word2Vec

def train(data_dir, sentences, model_name, size=100):
    model = Word2Vec(sentences, size=size, window=5, min_count=5, workers=4)
    model.save(data_dir / model_name)
    return model

def load_model(location):
    return Word2Vec.load(location)
