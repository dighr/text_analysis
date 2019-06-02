# Based on this tutorial
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
from gensim.models import Word2Vec

# Creating word embedding model using google's Word2Vec algorithim
# size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token (word).
# window: (default 5) The maximum distance between a target word and words around the target word.
# min_count: (default 5) The minimum count of words to consider when training the model; words with an occurrence less than this count will be ignored.
# workers: (default 3) The number of threads to use while training.
# sg: (default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).
# Training word embedding using Word2Vec
# define training data


from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
print(X)
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
