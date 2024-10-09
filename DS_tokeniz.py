from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import DS_init
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def possum_tokenizer(pd_text):
    tokenizer = Tokenizer(num_words=DS_init.num_ngramm)
    arr, bag_of_threegramms = from_pd_mass_to_mass_ngramms(pd_text)
    tokenizer.fit_on_texts(bag_of_threegramms)
    arr_of_tokens = from_mass_tokens_to_vectors(arr, tokenizer)
    return np.array(arr_of_tokens)


def from_mass_tokens_to_vectors(arr, tokenizer):
    arr_vec = []
    zero_list = np.array([0] * DS_init.num_of_words)
    for text in arr:
        arr_vec.append(pad_sequences(tokenizer.texts_to_sequences(text), maxlen=DS_init.num_of_ngramm_in_string))
    return pad_sequences(arr_vec, value=zero_list, maxlen=DS_init.num_of_words)

def from_pd_mass_to_mass_ngramms(pd_text):
    bag_of_threegramms = []
    # razbivaem na trigrammi
    arr = []
    for text in pd_text:
        arr_text = []
        str_split = word_tokenize(text)
        # str_split = text.split()
        for word in str_split:
            arr_word = []
            word = "[" + word + "]"
            for i in range(len(word) - 3):
                token_ngramm = word[i:i + 3]
                arr_word.append(token_ngramm)
                bag_of_threegramms.append(token_ngramm)
            arr_text.append(" ".join(arr_word))
        arr.append(arr_text)

    return arr, bag_of_threegramms


def create_new_df(request, df):
    # df = pd.read_excel(df_path, skiprows=1)
    # df.columns = ['string', 'class_row'] # задание имен выбранным столбцам
    stopwords_mass = stopwords.words('russian')
    bag_of_threegramms = []
    x_mass = []
    y_mass = []

    class_row = df[request].class_row

    for next_str in df:
        if next_str.class_row == class_row:
            class_sim = 1
        else:
            class_sim = 0

        arr_text = []

        new_next_str = next_str.string.translate(None, string.punctuation)
        clean_new_next_str = re.sub('[^A-Za-z0-9]+', '', new_next_str)

        str_split = clean_string(word_tokenize(clean_new_next_str), stopwords_mass)

        for word in str_split:
            arr_word = []
            word = "[" + word + "]"
            for i in range(len(word) - 3):
                token_ngramm = word[i:i + 3]
                arr_word.append(token_ngramm)
                bag_of_threegramms.append(token_ngramm)
            arr_text.append(" ".join(arr_word))

        x_mass.append(arr_text)
        y_mass.append(class_sim)

    return x_mass, y_mass


def clean_string(string_req, stoplist):
    clear_list = []
    for string in string_req:
        if string not in stoplist:
            clear_list.append(string)

    return clear_list