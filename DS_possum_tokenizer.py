import pandas
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import DS_init
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import torch

def possum_tokenizer(df):
    tokenizer = Tokenizer(num_words=DS_init.num_ngramm)

    stopwords_mass = stopwords.words('russian')
    bag_of_threegramms = []
    x_mass = []
    y_mass = []

    # class_row = df[request].class_row
    if type(df) == pandas.DataFrame:
        for index, next_str in df.iterrows():

            arr_text = []

            # new_next_str = next_str.request.translate(None, string.punctuation)
            clean_new_next_str = next_str.request.translate(str.maketrans('', '', string.punctuation)).lower()
            kjsbfsf = word_tokenize(clean_new_next_str)
            str_split = clean_string(kjsbfsf, stopwords_mass)

            for word in str_split:
                arr_word = []
                word = "[" + word + "]"
                for i in range(len(word) - 2):
                    token_ngramm = word[i:i + 3]
                    arr_word.append(token_ngramm)
                    bag_of_threegramms.append(token_ngramm)
                arr_text.append(" ".join(arr_word))

            x_mass.append(arr_text)
            y_mass.append(next_str.class_row)

    tokenizer.fit_on_texts(bag_of_threegramms)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    x_mass = from_mass_tokens_to_vectors(x_mass, tokenizer)

    return x_mass, y_mass, len(bag_of_threegramms)


def from_mass_tokens_to_vectors(arr, tokenizer):
    arr_vec = []
    zero_list = np.array([0] * DS_init.num_of_ngramm_in_string)

    for text in arr:
        arr_vec.append(pad_sequences(tokenizer.texts_to_sequences(text), maxlen=DS_init.num_of_ngramm_in_string))
    return pad_sequences(arr_vec, value=zero_list, maxlen=DS_init.num_of_words)


def clean_string(string_req, stoplist):
    clear_list = []
    for string in string_req:

        if string not in stoplist:
            clear_list.append(string)

    return clear_list


def possum_from_seq_to_vec(seq, mass_seq):
    stopwords_mass = stopwords.words('russian')
    x_mass = []

    if type(mass_seq) == pandas.DataFrame:
        is_df = True
        iterator = mass_seq.iterrows()
    else:
        is_df = False
        iterator = mass_seq

    # class_row = df[request].class_row
    for next_str in iterator:
        if is_df:
            index, next_str = next_str

        arr_text = []
        # new_next_str = next_str.request.translate(None, string.punctuation)
        clean_new_next_str = next_str.request.translate(str.maketrans('', '', string.punctuation)).lower()
        kjsbfsf = word_tokenize(clean_new_next_str)
        str_split = clean_string(kjsbfsf, stopwords_mass)

        for word in str_split:
            arr_word = []
            word = "[" + word + "]"
            for i in range(len(word) - 2):
                token_ngramm = word[i:i + 3]
                arr_word.append(token_ngramm)
            arr_text.append(" ".join(arr_word))

        x_mass.append(arr_text)


    # теперь то же, но со строкой
    arr_seq = []
    arr_text = []
    clean_new_next_str = seq.translate(str.maketrans('', '', string.punctuation)).lower()
    kjsbfsf = word_tokenize(clean_new_next_str)
    str_split = clean_string(kjsbfsf, stopwords_mass)

    for word in str_split:
        arr_word = []
        word = "[" + word + "]"
        for i in range(len(word) - 2):
            token_ngramm = word[i:i + 3]
            arr_word.append(token_ngramm)
        arr_text.append(" ".join(arr_word))
    arr_seq.append(arr_text)

    handle = open('tokenizer.pickle', 'rb')
    tokenizer = pickle.load(handle)
    x_mass = from_mass_tokens_to_vectors(x_mass, tokenizer)
    seq_mass = from_mass_tokens_to_vectors(arr_seq, tokenizer)

    request_batch = torch.Tensor(np.tile(np.array(seq_mass[0]), (len(x_mass), 1, 1)))
    x_mass = torch.Tensor(x_mass)

    handle.close()

    return request_batch, x_mass



