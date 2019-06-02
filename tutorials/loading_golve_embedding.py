# Covert to word2vec, required only once
# from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot

# glove_input_file = '../models/glove.6B.100d.txt'
# word2vec_output_file = '../models/glove.6B.100d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)

# use it similar to the one before
# load the Stanford GloVe model
filename = '../models/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

sentence = "in the morning it can take a long time because I usually go to the well close to our house. " \
           "But then the person who has keys to the well may not be there so I have to wait a long time. " \
           "Then I may need to come back in the afternoon. " \
           # "So quite often, we go to the river which even though it's very far and we know that the water can make make us sick. " \
           # "Our ward cheif has a water pump and the quality is very good, but he charges us a lot"
# remove everything other than letter and spaces
whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
words = ''.join(filter(whitelist.__contains__, sentence)).lower().split()

X = model[words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
