import json
import plotly
import plotly.plotly as py
import plotly.graph_objs as goj
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    category_count= list(df[df.columns[4:]].sum().sort_values(ascending=False))
    category_names = df.columns[4:]
    natural_disaster = df[['fire','floods','storm','earthquake','cold','other_weather']].sum().sort_values(ascending=False)
    natural_disaster_names = list(natural_disaster.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count,
                    marker=dict(color='rgb(51, 204, 51)',line=dict(color='rgb(15, 62, 15)',width=1.5,)),
                    opacity=0.6
                )
            ],

            'layout': {
                'title': 'Number of Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle' : 30
                }
            }
        },
                {
            'data': [
                Bar(
                    x=natural_disaster_names,
                    y=natural_disaster,
                    marker=dict(color='rgb(25, 102, 25)',line=dict(color='rgb(15, 62, 15)',width=1.5,)),
                    opacity=0.6
                )
            ],

            'layout': {
                'title': 'Number of Messages per Natural Disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Natural Disaster"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()