import re

import spacy
import nltk

from textblob import TextBlob


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")


def clean_text(sent: str) -> str:
    sent = sent.lower()  # Text to lowercase
    pattern = "[^\w\s]"  # Removing punctuation
    sent = re.sub(pattern, "", sent)
    pattern = "\w*\d\w*"  # Removing words with numbers in between
    sent = re.sub(pattern, "", sent)
    return sent


# Write your function to Lemmatize the texts
def lemmmatize_text(text: str) -> str:
    sent = []
    doc = nlp(text)
    for token in doc:
        sent.append(token.lemma_)
    return " ".join(sent)


def get_pos_tags(text: str) -> str:
    sent = []
    blob = TextBlob(text)
    sent = [word for (word, tag) in blob.tags if tag in ("NN", "NNS", "NNP", "NNPS")]
    return " ".join(sent)
