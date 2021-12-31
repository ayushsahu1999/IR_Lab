import numpy as np

doc_word_mapper = {}
l_list = []
index_dict = {}
freq_dict = {}

word_class_mapper = {} # [word, class]: count
label_count_mapper = np.zeros(20)
class_docs = {}
train_y = []

with open('matlab/test.label', 'r') as fp:
    for line in fp:
        label = int(line.strip())
        # its good to point out that doc_id 'n' will have label at 'n-1'th place
        # in simple words doc0 will have label at train_y[0] which will be between 1-20
        train_y.append(label)
        class_docs[label] = class_docs.get(label, 0)

i = 0
with open('matlab/test.data', 'r') as fp:
    for line in fp:
        doc_id, word_id, count = line.strip().split(' ')
        if word_id not in l_list:
            index_dict[word_id] = i
            i = i + 1
            l_list.append(word_id)

class_vector = np.zeros((20, len(l_list)))
class_count = np.zeros(20)

previous_doc = 1
with open('matlab/test.data', 'r') as fp:
    temp = np.zeros(len(l_list))
    total_count = 0
    for line in fp:
        doc_id, word_id, count = line.strip().split(' ')
        doc_id = int(doc_id)
        if previous_doc != doc_id:
            class_vector[train_y[int(previous_doc)]-1, :] += (temp/total_count)
            class_count[train_y[int(previous_doc)]-1] += 1
            temp = np.zeros(len(l_list))
            temp[index_dict[word_id]] = int(count)
            total_count = 0
            previous_doc = int(doc_id)
        else:
            temp[index_dict[word_id]] = int(count)
            total_count += int(count)

for i in range(20):
    class_vector[i] = class_vector[i] / class_count[i]

# TESTING

def cosine(a, class_vector):
    a=np.array(a)
    class_vector = np.array(class_vector)
    dist = np.zeros(20, dtype=np.float64)
    for i in range(20):
        dist[i] = np.sum(abs(a-class_vector[i]))
    return np.argmin(dist)

previous_doc = 1
y_expected = []
conf_matrix = np.zeros((20, 20))
with open('matlab/train.label', 'r') as fp:
    for line in fp:
        y_expected.append(int(line.strip()))

total_test_docs = len(y_expected)
correct_classified = 0

with open('matlab/train.data', 'r') as fp:
    j = 0
    temp = np.zeros(len(l_list))
    total_count = 0
    for line in fp:
        doc_id, word_id, count = line.strip().split(' ')

        doc_id = int(doc_id)

        if previous_doc != doc_id:
            original = y_expected[int(previous_doc)-1]
            original -= 1 ######## to make range 0-19
            temp = temp / total_count
            predicted = cosine(temp, class_vector)

            if (predicted == original):
                correct_classified += 1
            conf_matrix[original][predicted] += 1
            temp = np.zeros(len(l_list))
            previous_doc = doc_id

            if word_id not in l_list:
                continue
            temp[index_dict[word_id]] = int(count)

incorrect_classified = total_test_docs - correct_classified
print (correct_classified / total_test_docs)

# print ("Max: " + int(temp))

## Precision-Recall F-Score
# print (conf_matrix)

precision = np.zeros(20)
recall = np.zeros(20)
for i in range(20):
    temp = 0
    for j in range(20):
        temp += conf_matrix[i][j]
    precision[i] = conf_matrix[i][i] / float(temp)

for j in range(20):
    temp = 0
    for i in range(20):
        temp += conf_matrix[i][j]
    recall[j] = conf_matrix[j][j] / float(temp)

print (precision)
print (recall)

Precision = np.average(precision)
Recall = np.average(recall)
print ((Precision, Recall))

F_Measure = (Z * Precision * Recall) / (Precision + Recall)
print (F_Measure)