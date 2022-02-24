import bcolz
import numpy as np
import pickle

class Word2Vec:
    def __init__(self):
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=f'glove/6B.50.dat', mode='w')

        with open(f'glove/glove.6B.50d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
            
        vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'glove/6B.50.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open(f'glove/6B.50_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'glove/6B.50_idx.pkl', 'wb'))

        vectors = bcolz.open(f'glove/6B.50.dat')[:]
        words = pickle.load(open(f'glove/6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'glove/6B.50_idx.pkl', 'rb'))

        self.glove = {w: vectors[word2idx[w]] for w in words}

    def v(self, word):
        try:
            return self.glove[word]
        except:
            return None
        

    def compare(g1, g2):
        return (g1 / np.linalg.norm(g1)) @ (g2 / np.linalg.norm(g2))