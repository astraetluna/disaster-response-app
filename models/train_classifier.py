import sys
import pandas as pd
import re
import joblib
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import pickle

def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("Messages", con=engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis = 1)
    category_names = Y.columns.tolist()
    return X,Y,category_names 


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    default_stopwords = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in default_stopwords]
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    model =  Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    # set parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
        }
    # optimize model
    model = GridSearchCV(model, param_grid=parameters, cv=2, verbose=1, n_jobs=1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_test_pred= model.predict(X_test)
    print("Testing metrics")
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_test_pred[:, i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()