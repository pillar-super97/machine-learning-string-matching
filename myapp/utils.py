import re, math
import numpy as np
import sparse_dot_topn.sparse_dot_topn as ct
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def ngrams(string, n=3):
    string = string.encode("ascii", errors="ignore").decode() 
    string = string.lower()
    chars_to_remove = [')', '(', '.', '|', '[', ']', '{', '}', "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) # remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ').replace('-', ' ').replace('#', ' ')
    string = string.title() # Capital at start of each word
    string = re.sub(' +',' ',string).strip() # combine whitespace
    string = ' ' + string + ' ' # pad
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def manipulate(value):
    return 1. / (1+ math.sqrt(value/10))*100


def tfidf_match(source_list, target_list, flag=None):
    """For each item in list1, find the match in list2"""
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(target_list)
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    distances, indices = nbrs.kneighbors(vectorizer.transform(source_list))
    if not flag:
        matches = [(distances[i][0], source_list[i], target_list[j[0]]) 
                for i, j in enumerate(indices)]
        matches = pd.DataFrame(matches, columns=['confidence', 'source', 'target'])
        matches['confidence'] = matches["confidence"].apply(manipulate)
    else:
        matches = [(source_list[i], target_list[j[0]]) for i, j in enumerate(indices)]
        matches = pd.DataFrame(matches, columns=['source', 'target'])
    return matches 

