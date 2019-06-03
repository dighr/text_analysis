import sys
from gensim.models import KeyedVectors

# Category -> words
data = {
  'water_sources': ['lake','spring','river','well'],
  'issues': ['quality', 'fees','time'],
    'time': ['morning', 'time'],
   'Places': ['tokyo','bejing','washington','mumbai', 'sanaa'],
}

# Words -> category
categories = {word: key for key, words in data.items() for word in words}

# Load the whole embedding matrix
filename = '../models/glove.6B.100d.txt.word2vec'
embeddings_index = KeyedVectors.load_word2vec_format(filename, binary=False)

# Embeddings for available words
data_embeddings = {}
for key in categories.keys():
    if key in embeddings_index:
        data_embeddings[key] = embeddings_index[key]


# Processing the query
def predicted_category(query):
  query_embed = embeddings_index[query]
  scores = {}
  for word, embed in data_embeddings.items():
    category = categories[word]
    dist = query_embed.dot(embed)
    # dist /= len(data[category])
    scores[category] = scores.get(category, 0) + dist

  # get category with max score
  current_category = ""
  max_score = -sys.maxint - 1
  for category, score in scores.items():
      if score > max_score:
          max_score = score
          current_category = category

  return current_category, scores


def process_sentence(sentence):
    # remove everything other than letter and spaces
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    words = ''.join(filter(whitelist.__contains__, sentence)).lower().split()

    output = {}
    for word in words:
        category, scores = predicted_category(word)
        if category in output:
            output[category].append(word)
        else:
            output[category] = [word]

    return output


# Testing
sentence = "in the morning it can take a long time because I usually go to the well close to our house. " \
           "But then the person who has keys to the well may not be there so I have to wait a long time. " \
           "Then I may need to come back in the afternoon. " \
           "So quite often, we go to the river which even though it's very far and we know that the water can make make us sick. " \
           "Our ward cheif has a water pump and the quality is very good, but he charges us a lot"


process_sentence(sentence)
