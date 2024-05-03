# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
# import glob
import pickle
import sys

import numpy as np
import pandas as pd
# import re
# import requests

# from . import clf_path, config

@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
# @main.command('dl-data')
# def dl_data():
#     """
#     Download training/testing data.
#     """
#     data_url_fake = config.get('data', 'urlfake')
#     data_url_true = config.get('data', 'urltrue')
#     data_file_fake = config.get('data', 'filefake')
#     data_file_true = config.get('data', 'filetrue')
#     print('downloading from %s to %s' % (data_url_fake, data_file_fake))
#     r = requests.get(data_url_fake)
#     with open(data_file_fake, 'wt') as f:
#         f.write(r.text)
#     print('downloading from %s to %s' % (data_url_true, data_file_true))
#     r = requests.get(data_url_true)
#     with open(data_file_true, 'wt') as f:
#         f.write(r.text)
    

def data2df():
    return (pd.read_csv("./nlp/data/fake.csv"), pd.read_csv("./nlp/data/true.csv"))

@main.command('stats')
def stats():
    """
    Read the data files and print interesting statistics.
    """
    df_fake, df_true = data2df()
    print('%d rows' % len(df_fake))
    print('label counts:')
    # print(df_fake.partisan.value_counts())

    print('%d rows' % len(df_true))
    print('label counts:')
    # print(df_true.partisan.value_counts())     

@main.command('train')
def train():
    """
    Train a classifier and save it.
    """
    # (1) Read the data...
    fake, true = data2df()

    true['label'] = 1
    fake['label'] = 0

    df_news = pd.concat([fake, true])
    df_news = df_news.sample(frac=1)
    df_news.reset_index(drop=True, inplace=True)

    import spacy
    import pycountry
    from sklearn.model_selection import train_test_split

    # spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    train_data_news, test_data_news, label_train_data_news, label_test_data_news = train_test_split(
        df_news.iloc[:, :-1], 
        df_news.iloc[:, -1:], 
        test_size=0.2, 
        stratify=df_news.iloc[:, -1:])

    countries = [country.name for country in pycountry.countries]
    # Generate a list of country names

    def replace_countries(text):
            for country in countries:
                text = text.replace(country, "country")
            return text

    def process_text_batch(docs):
        texts = [replace_countries(doc.replace('\xa0', ' ')) for doc in docs]
        docs = list(nlp.pipe(texts))
        return [[token.lemma_ for token in doc if not token.is_stop] for doc in docs]

    def parallel_process_text(data, batch_size=1000):
        processed_data = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            processed_batch = process_text_batch(batch)
            processed_data.extend(processed_batch)
        return processed_data

    X_train_text = parallel_process_text(train_data_news['text'].tolist())
    X_test_text = parallel_process_text(test_data_news['text'].tolist())

    X_train_text_join = [' '.join(doc) for doc in X_train_text]
    X_test_text_join = [' '.join(doc) for doc in X_test_text]

    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train_text_join)
    
    X_train_text_seq = tokenizer.texts_to_sequences(X_train_text_join)
    X_test_text_seq = tokenizer.texts_to_sequences(X_test_text_join)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from keras.utils import pad_sequences

    X_train_text_sq = pad_sequences(X_train_text_seq, maxlen=150)
    X_test_text_sq = pad_sequences(X_test_text_seq, maxlen=150)

    model_text = keras.Sequential()
    model_text.add(layers.Embedding(10000, 128))
    model_text.add(layers.GRU(units=96, dropout=0.4, return_sequences=True))
    model_text.add(layers.GlobalMaxPooling1D())
    model_text.add(layers.Dense(units=96, activation='elu'))
    model_text.add(layers.Dropout(rate=0.2))
    model_text.add(layers.Dense(units=128, activation='elu'))
    model_text.add(layers.Dropout(rate=0.2))
    model_text.add(layers.Dense(units=48, activation='elu'))
    model_text.add(layers.Dropout(rate=0.4))
    model_text.add(layers.Dense(units=96, activation='elu'))
    model_text.add(layers.Dropout(rate=0.2))
    model_text.add(layers.Dense(units=112, activation='elu'))
    model_text.add(layers.Dropout(rate=0.4))
    model_text.add(layers.Dense(units=96, activation='elu'))
    model_text.add(layers.Dropout(rate=0.2))
    model_text.add(layers.Dense(units=32, activation='elu'))
    model_text.add(layers.Dropout(rate=0.4))
    model_text.add(layers.Dense(units=96, activation='elu'))
    model_text.add(layers.Dropout(rate=0.4))
    model_text.add(layers.Dense(units=32, activation='relu'))
    model_text.add(layers.Dropout(rate=0.3))
    model_text.add(layers.Dense(1, activation='sigmoid'))

    model_text.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_text.fit(X_train_text_sq, label_train_data_news, epochs=4, validation_data=(X_test_text_sq, label_test_data_news))

    # (3) do cross-validation and print out validation metrics
    # (classification_report)
    score, acc = model_text.evaluate(X_test_text_sq, label_test_data_news, batch_size=128)
    print("The model achieves an accuracy of " + str(acc) + "%")

    # (4) Finally, train on ALL data one final time and
    # train. Save the classifier to disk.
    model_text.save('./nlp/data/model.keras')

    # Serialize the tokenizer to a file
    with open('./nlp/data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

@main.command('test')
def test():
    model = pickle.load(".nlp/data/model.pkl")



if __name__ == "__main__":
    sys.exit(main())
