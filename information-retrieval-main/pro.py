import numpy as np
import math
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from natsort import natsorted
stop_w = stopwords.words("english")
stop_w.remove('in')
stop_w.remove('to')
stop_w.remove('where')
fli_nam = natsorted(os.listdir('files'))

documint_of_term = []
for files in fli_nam:
    with open(f'files/{files}', 'r') as f:
        doc = f.read()
        tok_doc = word_tokenize(doc)
        terms = []
        for word in tok_doc:
            if word not in stop_w:
                terms.append(word)
        documint_of_term.append(terms)


print()
print("----------------------------------positioal index ----------------------------")
print()
doc_num = 0
p_i = {}

for doc in documint_of_term:
    for pos, term in enumerate(doc):
        if term in p_i:
            p_i[term][0] = p_i[term][0]+1
            if doc_num in p_i[term][1]:
                p_i[term][1][doc_num].append[pos]
            else:
                p_i[term][1][doc_num] = [pos]
        else:
            p_i[term] = []
            p_i[term].append(1)
            p_i[term].append({})
            p_i[term][1][doc_num] = [pos]
    doc_num += 1
print(p_i)
print()
print()
print()
print("----------------------------------tf----------------------------")
print()
all_words = []

for doc in documint_of_term:
    for word in doc:
        all_words.append(word)


def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(get_term_freq(documint_of_term[0]).values(
), index=get_term_freq(documint_of_term[0]).keys())
for i in range(1, len(documint_of_term)):
    term_freq[i] = get_term_freq(documint_of_term[i]).values()

term_freq.columns = ['doc'+str(i) for i in range(1, 11)]
print(term_freq)


def get_weighted_term_freq(x):
    if x > 0:
        return math.log(x) + 1
    return 0


for i in range(1, len(documint_of_term)+1):
    term_freq['doc'+str(i)] = term_freq['doc'+str(i)
                                        ].apply(get_weighted_term_freq)
print()
print("----------------------------------w_tf----------------------------")

print(term_freq)

print()


tfd = pd.DataFrame(columns=['freq', 'idf'])
for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()
    tfd.loc[i, 'freq'] = frequency
    tfd.loc[i, 'idf'] = math.log10(10/(float(frequency)))
tfd.index = term_freq.index
print(tfd)
term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)
print()
print("----------------------------------term_freq_inve_doc_freq----------------------------")
print()
print(term_freq_inve_doc_freq)


document_length = pd.DataFrame()


def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x**2).sum())


for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column+'-len'] = get_docs_length(column)

print()
print("----------------------------------document_length----------------------------")
print()
print(document_length)


normalized_term_freq_idf = pd.DataFrame()


document_length = pd.DataFrame()


def get_docs_length(co1):
    return np.sqrt(term_freq_inve_doc_freq[co1].apply(lambda x: x**2).sum())


for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column+'_len'] = get_docs_length(column)
normalized_term_freq_idf = pd.DataFrame()


def get_normalized(co1, x):
    try:
        return x/document_length[co1+'_len'].values[0]
    except:
        return 0


for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(
        lambda x: get_normalized(column, x))
print()
print("----------------------------------normalized_term_freq_idf----------------------------")
print()
print(normalized_term_freq_idf)

print()

print()

q = input("Enter Your Query: ")
print("----------------------------------FOUND IN ---------------------------")
query = q
try:
    print("FOUND IN :")
    final_list = [[] for i in range(10)]
    for word in query.split():
        for key in p_i[word][1].keys():
            if final_list[key-1] != []:
                if final_list[key-1][-1] == p_i[word][1][key][0]-1:
                    final_list[key-1].append(p_i[word][1][key][0])
            else:
                final_list[key-1].append(p_i[word][1][key][0])

    for position, list in enumerate(final_list, start=1):
        #print(position , list)

        if len(list) == len(query.split()):

            print(position+1, list)
except:
    print('not found ')


def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0


query = pd.DataFrame(index=normalized_term_freq_idf.index)
query['tf'] = [1 if x in q.split() else 0 for x in (
    normalized_term_freq_idf.index)]
query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
product = normalized_term_freq_idf.multiply(query['w_tf'], axis=0)
query['idf'] = tfd['idf'] * query['w_tf']
query['tf_idf'] = query['w_tf'] * query['idf']
query['norm'] = 0
for i in range(len(query)):
    query['norm'].iloc[i] = float(
        query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))
query = query[(query.T != 0).any()]
print("----------------------------------query  ---------------------------")
print(query)
product2 = product.multiply(query['norm'], axis=0)
print()
print()
print()
scores = {}
for col in product2.columns:
    if 0 in product2[col].loc[q.split()].values:
        pass
    else:
        scores[col] = product2[col].sum()
print("----------------------------------simliarty  ---------------------------")
print(scores)
print()
print()

print("---------------------------query lanthe---------------------------------")
print(math.sqrt(sum([x**2 for x in query['idf'].loc[q.split()]])))
print()
print()
prod_res = product2[(scores.keys())].loc[q.split()]
print("---------------------------------- product ---------------------------")
print(prod_res)
print()
print("---------------------------------- sum product ---------------------------")
print(prod_res.sum())
print("---------------------------------- ranking  ---------------------------")
final_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for doc in final_score:
    print(doc[0], end=' ')
print()
print("---------------------------------------------------------------")
