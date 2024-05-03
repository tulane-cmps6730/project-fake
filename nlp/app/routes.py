from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm

import tensorflow as tf
from tensorflow import keras
import spacy
import pycountry
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from lime.lime_text import LimeTextExplainer

model_text = tf.keras.models.load_model('./nlp/data/model.keras')

# Load the tokenizer from the same file
with open('./nlp/data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print('read model_text %s' % str(model_text))

countries = [country.name for country in pycountry.countries]
def replace_countries(text):
    for country in countries:
        text = text.replace(country, "country")
    return text

def process_single_text(doc):
	text = doc.replace('\xa0', ' ')    	
	new_text = replace_countries(text)
	nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
       
	processed_doc = nlp(new_text)
	
	lemmatized_tokens = [token.lemma_ for token in processed_doc if not token.is_stop and not token.is_punct]
	
	clean_text = ' '.join(lemmatized_tokens)

	return clean_text

from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_proba(texts):
    # Preprocess texts (cleaning, lemmatizing)
    processed_texts = texts
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=150)
    # Predict and format for LIME
    predictions = model_text.predict(padded_sequences)
    return np.hstack((1-predictions, predictions))


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		input_field = form.input_field.data
                        
		processed_input = process_single_text(input_field)

		# Initialize the explainer
		explainer = LimeTextExplainer(class_names=["False", "True"])

		# Generate explanation
		exp = explainer.explain_instance(processed_input, predict_proba, num_features=10)
            
		exp_html = exp.as_html()
        
		style = '''
		<style>
		    body, p, li { color: #fff; background-color: white; } /* Adjust text and background colors */
            .highlight { color: red; } /* Custom class adjustments */
        </style>
        '''
	
		html_output = style + exp_html  # Concatenate style with the explanation HTML
            
		return render_template('base.html', form=form, explanation=html_output)
            
		#return redirect('/index')
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
