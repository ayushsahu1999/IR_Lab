from collections import defaultdict, deque
import numpy as np
import math
import sys
import gc
from copy import deepcopy

label_count_mapper = np.zeros((20, 20)) # since i know total number of docs I'm making this array directly
class_docs = {} # how much docs each class has
train_y = [] # training labels (1-20)
conf_matrix = np.zeros((20, 20))

with open('./matlab/train.label', 'r') as fp:
  for line in fp:
    label = int(line.strip())
    # its good to point out that doc_id 'n' will have label at 'n-1'th place
    # in simple words doc0 will have label at train_y[0] which will be between 1-20
    train_y.append(label)
    class_docs[label] = class_docs.get(label, 0) + 1
    
previous_doc = 1
freq = {} # (word, class): count
with open('./matlab/train.data', 'r') as fp:
  word_ids = [] # this will basically contain the words and their counts
  for line in fp:
    doc_id, word_id, count = map(int, line.strip().split(' '))
    # this means we have got all the words of our doc now we can take care of frequencies
    # for doc1
    # (1,5), (2,10), ...
    # doc1 => class1
    if previous_doc != doc_id:
      # doc N will have label at (N-1)th place
      class_of_doc = train_y[previous_doc-1] # getting class of doc
      for word, word_count in word_ids:
        # we are gonna have count of each word in each class in this dictionary
        # example. hello, class1 => 20 times
        freq[(word, class_of_doc)] = freq.get((word, class_of_doc), 0) + word_count
      previous_doc = doc_id
      word_ids = [(word_id, count)]
    else:
      word_ids.append((word_id, count)) # appending word_id
# for the last doc
class_of_doc = train_y[previous_doc-1] # getting class of doc
for word, count in word_ids:
  freq[(word, class_of_doc)] = freq.get((word, class_of_doc), 0) + word_count
# del word_ids

vocab = set([pair[0] for pair in freq.keys()])
v_len = len(vocab)

# finding how much words each class has
class_words = {}
for pair, word_count in freq.items():
  class_ = pair[1]
  class_words[class_] = class_words.get(class_, 0) + word_count

total_docs = len(train_y) # this should be 11269 if using their indexing (see train.data)
# sum(class_docs.values()) # again, this should be same as total_docs

prob_class = {}
prob_word_class = {}

# finding probability of each class
for i in class_docs:
  prob_class[i] = class_docs[i]/total_docs

# findinf probability of each word in each class
# we are doing this with smoothing too, for better results
for word in vocab:
  for class_ in class_words:
    freq_class = freq.get((word, class_), 0)
    # word/class
    prob_word_class[(word, class_)] = (freq_class + 1)/(class_words[class_] + v_len)


conf_matrix = np.zeros((20,20))
previous_doc = 1
y_expected = []
y_actual = []
with open('./matlab/test.label', 'r') as fp:
  for line in fp:
    y_expected.append(int(line.strip()))
total_test_docs = len(y_expected)
correct_classified = 0
with open('./matlab/test.data', 'r') as fp:
  word_ids = [] # this will basically contain the words and their counts
  j = 0
  for line in fp:
    doc_id, word_id, count = map(int, line.strip().split(' '))
    if previous_doc != doc_id:
      probs = deepcopy(prob_class)
      for i in probs:
        probs[i] = np.log(probs[i])
      for word, word_count in word_ids:
        for class_ in range(1,21):
          # print(prob_word_class.get((word, class_), 1e-5), end=' ')
          probs[class_] = probs[class_]  +  word_count * np.log(prob_word_class.get((word, class_), 1e-5))
          # probs[i] = probs[i] + word_count * p_word_class.get((word, i), 1e-4)
      _max_class = 1
      _max_val = - np.inf
      for i in probs:
        if probs[i] > _max_val:
          _max_val = probs[i]
          _max_class = i
      y_actual.append(_max_class)
      # print(_max_class, end = ' ')
      if y_expected[j] == _max_class:
        correct_classified+=1
      conf_matrix[_max_class-1][y_expected[j]-1] = conf_matrix[_max_class-1][y_expected[j]-1] + 1
      j += 1
      previous_doc = doc_id
      word_ids = [(word_id, count)]
    else:
      word_ids.append((word_id, count)) # appending word_id

# this is for the last word_ids (code can be taken into function to remove redunduncy)
for word, word_count in word_ids:
  for class_ in range(1,21):
    # print(prob_word_class.get((word, class_), 1e-5), end=' ')
    probs[class_] = probs[class_]  +  word_count * np.log(prob_word_class.get((word, class_), 1e-5))
    # probs[i] = probs[i] + word_count * p_word_class.get((word, i), 1e-4)
_max_class = 1
_max_val = - np.inf
for i in probs:
  if probs[i] > _max_val:
    _max_val = probs[i]
    _max_class = i
# print(_max_class, end = ' ')
y_actual.append(_max_class)
if y_expected[j] == _max_class:
  correct_classified+=1
conf_matrix[_max_class-1][y_expected[j]-1] = conf_matrix[_max_class-1][y_expected[j]-1] + 1

incorrect_classified = total_test_docs - correct_classified

print(correct_classified/total_test_docs)

from sklearn.metrics import f1_score
print ("F1 Score: ", f1_score(y_expected, y_actual, average=None))
print (conf_matrix)
