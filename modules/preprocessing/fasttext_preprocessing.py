from modules.module import Preprocessing
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class FasttextPreprocessing(Preprocessing):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")

    def read_data(self, path='../../data/Rilla NLP Dataset - Sheet1.csv'):
        self.data = pd.read_csv(path)

    def preprocess_data(self, text):
        doc = self.nlp(text)

        for ent in doc.ents:
            text = text.replace(ent.text, ent.label_)

        doc = self.nlp(text.lower())
        return ' '.join([token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS])

    def save_data(self):
        with open('../../data/fasttext_data.txt', 'w') as write_file:
            write_file.write(' '.join(self.data.Content.to_list()))
        with open('../../data/fasttext_cleaned_data.txt', 'w') as write_file:
            write_file.write(' '.join(self.data.cleaned_content.to_list()))

    def get_data(self):
        self.read_data()
        self.data['cleaned_content'] = self.data['Content'].apply(
            self.preprocess_data)
        self.save_data()

        return self.data
