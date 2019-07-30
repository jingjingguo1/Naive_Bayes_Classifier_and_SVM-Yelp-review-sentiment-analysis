import sys
import string
import copy
from collections import Counter
from operator import itemgetter
import numpy as np
# from nbc import process_str, read_dataset, get_most_commons

#def process_str(s):
#    return s.translate(string.punctuation).lower().split()
#
## dataset format:
## list of (class_label, set of words)
#def read_dataset(file_name):
#    dataset = []
#    with open(file_name) as f:
#        for line in f:
#            index, class_label, text = line.strip().split('\t')
#            words = process_str(text)
#            dataset.append( (int(class_label), set(words)) )
#    return dataset
#
#def get_most_commons(dataset, skip=100, total=50):
#    my_list = []
#    for item in dataset:
#        my_list += list(item[1])
#
#    counter = Counter(my_list)
#
#    temp = counter.most_common(total+skip)[skip:]
#    words = [item[0] for item in temp]
#    return words
#
#def generate_vectors(dataset, common_words):
#    number_words= len(common_words)
#    d = {}
#    for i in range(len(common_words)):
#        d[common_words[i]] = i
#    
#    vectors = []
#    labels = []
#    for item in dataset:
#        vector = [0] * (number_words+1)
#        vector[0] = 1
#        for word in item[1]:
#            if word in d:
#                k = d[word]+1 # shift to the right by 1 due to intercept
#                vector[k] = 1
#        if item[0] == 0:
#            vectors.append(vector)
#            labels.append(-1)
##            vectors.append( (-1, vector) )
#        else:
#            vectors.append(vector)
#            labels.append(item[0])
##            vectors.append( (item[0], vector) )            
#    return np.array(vectors), np.array(labels)

def process_str(s):
    rem_punc = str.maketrans('', '', string.punctuation)
    return s.translate(rem_punc).lower().split()

def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), words) )

    return dataset

def get_most_commons(dataset, skip=100, total=100):
    counter = Counter()
    for item in dataset:
        counter = counter + Counter(set(item[1]))

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)

def score_function(train_vectors, train_labels, w):
    lamda = 0.01
    N = train_vectors.shape[0]
    W = np.linalg.norm(w)**2*lamda/2
                      
    y_hat = train_vectors.dot(w)
    yy_hat = train_labels.reshape((N,1))*y_hat
    G = 1- yy_hat[yy_hat<1].sum()/N
            
    return W+G
    
    

def svm_learn(train_vectors, train_labels):
    (m,n) = train_vectors.shape
    maxIter = 0
    lamda = 0.01
    w_new = np.ones([n,1]) # initialize
    eta = 0.5
    tol = 1000
    
    while (maxIter<1000 and tol > 1e-6):
#        old_score = score_function(train_vectors, train_labels, w_new)
        w_now = w_new
        y_hat = train_vectors.dot(w_new)
        print(train_labels.shape)
        yy_hat = train_labels.reshape((m,1))*y_hat
        weight_matrix = np.repeat(train_labels.reshape((m,1)), n, axis = 1)
        yx = weight_matrix * train_vectors
        gj = yx[np.where(yy_hat<1)[0].tolist(), :].mean(axis=0)
               
        grad = lamda * w_new - gj.reshape((n,1))
        
        w_new = w_new - eta * grad

#        new_score = score_function(train_vectors, train_labels, w_new)
#        tol = abs(new_score - old_score)
        tol = np.linalg.norm(w_new-w_now)
        print(tol)
        maxIter += 1
    return w_new

def svm(features, labels):
    # test sub-gradient SVM
    total = features.shape[1]
    lam = 1.; D = total
    x = features; y = (labels-0.5)*2
    w = np.zeros(D); wpr = np.ones(D)
    eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
    while True:
        if np.linalg.norm(w-wpr) < tol or i > MAXI:
            break
        f = w @ x.T
        pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
        pL = np.mean(pL,axis=1) + lam*w
        wpr = w
        w = w - eta*pL
        i += 1
        
    return w
        

def svm_predict(w, test_vectors):
    y_hat = test_vectors.dot(w)
    y_hat[y_hat>=0] = 1
    y_hat[y_hat<1] = -1 # int????
    return y_hat

def loss(test_labels, pred_labels):
    n_t = test_labels.shape[0]
    test_labels = test_labels.astype(int).reshape((n_t,1))
    pred_labels = pred_labels.astype(int)
    return (test_labels != pred_labels).sum()/n_t

def calc_error(pred, labels):
    error = sum(np.where(pred != labels, 1, 0))
    return (error / labels.size)

def svmMain(train_data_file, test_data_file):
    train_data = read_dataset(train_data_file)
    test_data = read_dataset(test_data_file)
    
    number_words = 4000

    common_words = get_most_commons(train_data, skip=100, total=number_words)

    train_vectors, train_labels = generate_vectors(train_data, common_words)
    test_vectors, test_labels = generate_vectors(test_data, common_words)
    
    w = svm(train_vectors, train_labels)
    pred_labels = svm_learn(w,test_vectors)
    
    zero_one_loss = calc_error(test_labels, pred_labels)
    print('ZERO-ONE-LOSS-SVM ' + str(zero_one_loss))
    return zero_one_loss

# svmMain("yelp_train1.txt", "yelp_test1.txt")

# outs = []
# out = [0]*9
# for j in [1,2,3,4,5,6,7,8,9,10]:
#     print(j)
#     for i in range(9):
#         num = 100 + 200 * i
#         if j == 10:
#             train_data_file = "Sc10_"+ str(num)+".dat"
#             test_data_file = "Sc10.dat"
#         else: 
#             train_data_file = "Sc0"+str(j)+"_"+ str(num)+".dat"
#             test_data_file = "Sc0" +str(j)+".dat"
#         error = svmMain(train_data_file, test_data_file)
#         out[i] = error
        
#     outs.append(out)
# print(outs)
trainingDataFileName = 'yelp_train0.txt'
testDataFileName = 'yelp_test0.txt'
x = svmMain(trainingDataFileName, testDataFileName)

'''# test generate_vectors
train_data = read_dataset("yelp_train1.txt")
test_data = read_dataset("yelp_test1.txt")
common_words = get_most_commons(test_data, skip=100, total=4000)
test_vectors = generate_vectors(test_data, common_words)
for i in range(10):
    print test_vectors[i][0]
    print len(test_vectors[i][1])
print i'''
    
    
# test classify
