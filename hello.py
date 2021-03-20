import nltk
import pandas
import spacy
from collections import defaultdict
import time
from tfidf import TfIdf
import sys
import os
import pickle
import io

class Tf_Idf:
    def __init__(self):
        self.weighted = False
        self.documents = []
        self.corpus_dict = {}

    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

        # normalizing the dictionary
        length = float(len(list_of_words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.documents.append([doc_name, doc_dict])

    def similarities(self, list_of_words):
        """Returns a list of all the [docname, similarity_score] pairs relative to a
list of words.
        """

        # building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # normalizing the query
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length

        # computing the list of similarities
        sims = []
        for doc in self.documents:
            score = 0.0
            doc_dict = doc[1]
            for k in query_dict:
                if k in doc_dict:
                    score += (query_dict[k] / self.corpus_dict[k]) + (
                      doc_dict[k] / self.corpus_dict[k])
            sims.append([doc[0], score])

        return sims


def Sort(sub_li):
    return(sorted(sub_li, key = lambda x: x[1]))  

#data = pandas.read_csv("/Users/as.tarlan02/Desktop/articles1.csv")
#data2 = pandas.read_csv("/Users/as.tarlan02/Desktop/articles2.csv")
#data3 = pandas.read_csv("/Users/as.tarlan02/Desktop/articles3.csv")



table = Tf_Idf()

with open('table.pickle', 'rb') as f:
    table = pickle.load(f)

#print(data.columns)
#words = defaultdict(list)
#ind = 0
#tic = time.perf_counter()
#data['id'][i]

#for index, row in data.iterrows():
#    cur_words = row['content'].lower().split()
#    table.add_document(row['id'], cur_words)


#with open('data.pickle', 'wb') as f:
#    pickle.dump(table, f)

#with open("table.pickle", "wb") as f:
#    pickle.dump(table, f)

result = table.similarities('washington')
result = Sort(result)

print(result[-1])
print(result[-2])
print(result[-3])
print(result[-4])
print(result[-5])



#toc = time.perf_counter()


