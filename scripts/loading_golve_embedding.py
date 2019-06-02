# Covert to word2vec, required only once
# from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# glove_input_file = '../models/glove.6B.100d.txt'
# word2vec_output_file = '../models/glove.6B.100d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)

# use it similar to the one before
# load the Stanford GloVe model
filename = '../models/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

