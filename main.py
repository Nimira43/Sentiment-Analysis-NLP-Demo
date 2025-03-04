import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

data = {
  'text': [
    'I love this product, it is amazing!',
    'The worst service ever, I was very disappointed.',
    'This is a brilliant product. Well worth the money.',
    'I think that it was an average show, good in some places.',
    'Outstanding hotel, the staff were amazing!',
    'Never again. This was a dire restaurant!',
    'Memorable performance by the terrific cast.',
    'Poor performance by them. Expected better.',
    'Rude staff but the food was quite nice to be fair.',
    'This is the worst experience ever.',
    'Absolutely fantastic, highly recommend it.',
    'Not bad, could be better.',
    'I hate it, very disappointing.'
  ],
  'sentiment': ['positive', 'negative', 'positive', 'neutral', 'positive', 'negative', 'positive', 'negative', 'neutral', 'negative', 'positive', 'neutral', 'negative']
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%') # Accuracy: 66.67%

def predict_sentiment(text):
  return model.predict([text])[0]

print(predict_sentiment('It was very average.'))