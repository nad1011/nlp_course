from gensim.test.utils import common_texts
import gensim.downloader
from gensim.models import Word2Vec

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
# print(list(gensim.downloader.info()['models'].keys()))
a1 = glove_vectors['name']
a2 = glove_vectors['vietnam']
print(glove_vectors.most_similar(a1+a2))
