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


#####################################################Pre-trained Word Embeddings##########################################################

def get_glove():
    """This method gets the Glove class."""
    
    global glove
    if not glove.loaded:
        glove = Glove(True)
    return glove


def encode_answers(df, category_col = None, question_ids = []):
    """This method prepare the data and after converting 
       it into numerical data, it makes the sure all the 
       words have the same length using  pad_sequences."""
    
    texts        = np.array(df['question'] + ' ' + df['referenceAnswers'] + ' ' + df['answer']).astype(str) 
    vocab_string = " ".join(texts)
    word_list    = vocab_string.split()
    word_list    = sorted(list(dict.fromkeys(word_list)))

    encoded_answers, vocabulary = pretrained_encoded(texts, word_list, category_col, question_ids)
    print(f"Pretrained encoded_answers shape: {np.array(encoded_answers).shape}")

    max_length = max([len(answer) for answer in encoded_answers])
    padded_answers = pad_sequences(encoded_answers, maxlen = max_length, dtype = 'float', padding = 'post').tolist()
    padded_answers = np.array(padded_answers)
    return max_length, padded_answers, vocabulary

def pretrained_encoded(answers, word_list, category_col, question_ids):
    """This method used the GloVe class to get the word embeddings."""
    
    counter     = 0
    glove       = get_glove()
    word2index  = glove.word2index
    found_words = [word for word in word_list if word in word2index]

    encoded_answers = []
    vocabulary      = {}
    for index, answer in enumerate(answers):
        try:
            arr = [int(question_ids[index])] if category_col else []
            for word in answer.split(' '):
                if word in word2index:
                    ## Create a mapping to the glove embedding word index
                    i = word2index[word]
                    vocabulary[i] = word
                    
                    ## Putpulate a custom embedding matrix with the words 
                    ## used indexed to the actual used vocabulary size.
                    i = found_words.index(word)
                    arr.append(i)
                else:
                    counter += 1 
                    print(f"ERROR! Word: {word} not found in dictionary")
                    arr.append(np.random.uniform())
        except:
            continue

        encoded_answers.append(arr)

    print('Total Number of Missing Words:', counter)

    ## If using pretrained embeddings, load them based on the vocabulary
    glove.load_custom_embedding(vocabulary)

    return encoded_answers, vocabulary


def generate_data(df, sample_size = 1, question_id = None):
    """This method converts textual data into numerical data."""

    if question_id:
         max_length, encoded_answers, vocabulary = encode_answers(df,
                                                                  question_id,
                                                                  df[question_id])
    else:
         max_length, encoded_answers, vocabulary = encode_answers(df,
                                                                  question_id,
                                                                  df[question_id])

    encoded_answer_df            = pd.DataFrame(encoded_answers)
    encoded_answer_df['correct'] = df['correct'].astype(float)

    ## Randomize the data
    randomized_data = encoded_answer_df.reindex(np.random.permutation(encoded_answer_df.index))

    ## max will allow sample size less the 100% of data.
    max = int(len(randomized_data) * sample_size)
    randomized_labels  = randomized_data['correct'].values[:max]
    randomized_answers = randomized_data.drop(['correct'], axis=1).values[:max]

    return randomized_answers, randomized_labels, max_length, vocabulary