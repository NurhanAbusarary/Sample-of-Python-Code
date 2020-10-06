### Importing the Packages

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import os
import re
import glob
import nltk
import numpy as np
import pandas as pd

from xml.dom import minidom

from LSTM_GLOVE import Glove

from bs4 import BeautifulSoup

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


glove      = Glove()
lemmatizer = WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag):
    """This method gets Parts Of Speech tags."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_word(answer):
    """This method convert the words to lower-case,
        removes the punctuation, remove the stop 
        words, and lemmatize the words."""
    
    text        = BeautifulSoup(str(answer), "html.parser").get_text()
    text        = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words       = word_tokenize(text)
     

    remove_stopwords = [w for w in words if not w in stopwords.words("english")]

    nltk_tagged = nltk.pos_tag(remove_stopwords)

    wordnet_tagged      = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            ## If there is no available tag, append the token as it is
            lemmatized_sentence.append(word)

        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return ' '.join(lemmatized_sentence)


def TFIDF(texts):
    """This method computes the TF-IDF matrix."""
    
    count_vect        = CountVectorizer()
    txt_freqs         = count_vect.fit_transform(texts)

    tfidf_transformer = TfidfTransformer().fit(txt_freqs)
    txt_tfidf         = tfidf_transformer.transform(txt_freqs)

    return txt_tfidf


def clean_data(df, question = None, reference_answer = None, answer = None):
    """This method cleand the questions, students' answers,
       and reference answers."""
    
    questions            = [lemmatize_word(ques) for ques in df[question].values]

    reference_answers    = [lemmatize_word(ref_ans) for ref_ans in df[reference_answer].values]

    answers              = [lemmatize_word(ans) for ans in df[answer].values]

    df[reference_answer] = reference_answers
    df[question]         = questions
    df[answer]           = answers

    return df
