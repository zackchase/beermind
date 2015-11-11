
import cPickle
import nltk
from collections import defaultdict
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords as sw
import scipy.sparse as sp
from tqdm import tqdm

#with open("beer_core-train.pkl") as f:
#    bc_train = cPickle.load(f)


def ngrams(data):

    wordcount = defaultdict(int)
    punctuation = set(string.punctuation)
    stopwords = sw.words('english')

    for datum in tqdm(data):

        text = datum.text
        r = ''.join([c for c in datum.text.lower() if not c in punctuation])
        
        words = r.split()
        

        for i in xrange(len(words)):
            if words[i] not in stopwords:
                wordcount[words[i]] += 1

            if i < (len(words)-1):
                wordcount[(words[i],words[i+1])] += 1
            
            if i < (len(words)-2):
                wordcount[(words[i], words[i+1], words[i+2])] += 1

            if i < (len(words)-3):
                wordcount[(words[i], words[i+1], words[i+2], words[i+3])] += 1



def featurize(data, ngram_list):
    num_examples = len(data)
    num_features = len(ngram_list)
    
    punctuation = set(string.punctuation)
    stopwords = sw.words('english')

    X_train = sp.lil_matrix((num_examples, num_features))


    ng_dict={}
    for i in xrange(len(ngram_list)):
        ng = ngram_list[i]
        ng_dict[ng] = i
        

    for i, datum in tqdm(enumerate(data)):
        datum = data[i]

        r = ''.join([c for c in datum.text.lower() if not c in punctuation])
        words = r.split()

        for j in xrange(len(words)):
            if words[j] in ng_dict:
                X_train[i, ng_dict[words[j]]] += 1

            if j < (len(words)-1):
                bigram = (words[j], words[j+1])
                if bigram in ng_dict:
                    X_train[i, ng_dict[bigram]] += 1

            if j < (len(words) -2):
                trigram = (words[j], words[j+1], words[j+2])
                if trigram in ng_dict:
                    X_train[i, ng_dict[trigram]] += 1

            if j < (len(words) - 3):
                quadgram = (words[j], words[j+1], words[j+2], words[j+3])
                if quadgram in ng_dict:
                    X_train[i, ng_dict[quadgram]] += 1

    return X_train
            

def ngram_list_to_dict(nglist):
    ng_dict = defaultdict(int)
    for i in xrange(len(nglist)):
        ng_dict[nglist[i][1]] = nglist[i][0]

    return ng_dict

        


            
            




