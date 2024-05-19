import random

import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from assets.lexicon import BOT_CONFIG


def get_generative_replica(text: str, mega_dataset: dict[str: list[tuple[str, str]]]) -> str:
    text = text.lower()
    words = text.split(' ')
    for word in words:
        if word in mega_dataset:
            for question, answer in mega_dataset[word]:
                if abs(len(text) - len(question)) / len(question) < 0.2:
                    distance = nltk.edit_distance(text, question)
                    difference = distance / len(question)
                    if difference < 0.2:
                        return answer


# Returns the subject of the message
def get_intent(text: str, clf: LinearSVC, vectorizer: TfidfVectorizer) -> str:
    text_vector = vectorizer.transform([text]).toarray()[0]
    intent = clf.predict([text_vector])[0]
    for example in BOT_CONFIG['intents'][intent]['examples']:
        distance = nltk.edit_distance(text.lower(), example.lower())
        difference = distance / len(example)
        if difference < 0.5:
            return intent


# Returns the reply to the message
def response_by_intent(intent: str) -> str:
    responses = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(responses)


# Returns the quantity of failure responses
def get_failure_phrase() -> str:
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)
