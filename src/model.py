from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from assets.lexicon import BOT_CONFIG


def train_model() -> tuple[LinearSVC, TfidfVectorizer]:
    X_text = []
    y = []
    for intent, value in BOT_CONFIG['intents'].items():
        X_text += value['examples']
        y += [intent] * len(value['examples'])

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), sublinear_tf=True)
    X = vectorizer.fit_transform(X_text)
    clf = LinearSVC()
    clf.fit(X, y)
    return clf, vectorizer
