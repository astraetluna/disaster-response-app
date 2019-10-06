# Disaster Response App - Natural Language Processing with Machine Learning Pipelines
1. In this project, we analyze disaster data to build a model for an API that classifies new input data.
A data set used to create a machine learning pipeline contains real messages that were sent during disaster events. The machine learning pipeline analyzes and categorizes these events, so that an emergency worker can input a new message and get classification results in several categories. The web app also displays some visualizations of the data.

2. The following demo shows, how the app works:  
<div style = "display: flex; justify-content: center">
<img src='media/demo.gif' width="785" height="368" />
</div>

<br>


# Project Components

## Most relevant files
- `process_data.py` - ETL pipeline - writes a data cleaning pipeline that, loads the messages and categories datasets, merges the two datasets, cleans the data, stores it in a SQLite database
- `train_classifier.py` -  Machine Learning Pipeline - Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file
- `run.py` - Flask Web App - Display visualization from the data, provides input field for messages from users and returns classification for categories of disaster events
- `classifier.pkl` - final model, which can be downloaded from [here](https://1drv.ms/u/s!AjI9m5VTVpPOhZ8dC88M0K1Hkq8nCw?e=uewmc1) because of github 100 MB upload limit.

## Project structure

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # ETL Pipeline
|- DisasterResponse.db   # processed data saved to database

- models
|- train_classifier.py # ML Pipeline
|- classifier.pkl  # saved model  - because of github upload limit < 100 MB, you can downloaded it above 

- media
|- demo.gif #  demo of the browser app

- README.md

```

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


# Licensing, Authors, and Acknowledgements

Data comes frome [Figure-eight](https://www.figure-eight.com/). Thanks to [Udacity](https://www.udacity.com/courses/all) for creating a beautiful learning experience. 