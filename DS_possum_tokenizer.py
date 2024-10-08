from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import DS_init
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def possum_tokenizer(df):
    tokenizer = Tokenizer(num_words=DS_init.num_ngramm)

    stopwords_mass = stopwords.words('russian')
    bag_of_threegramms = []
    x_mass = []
    y_mass = []

    # class_row = df[request].class_row

    for index, next_str in df.iterrows():
        # if next_str.class_row == class_row:
        #     class_sim = 1
        # else:
        #     class_sim = 0

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