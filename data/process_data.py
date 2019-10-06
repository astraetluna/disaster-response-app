import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge both datasets
    df = pd.merge(right = categories, left = messages, on = "id", sort = False)
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns from the dataframe 
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories
    row = categories.iloc[1]
    category_colnames = row.replace(regex=["[\-\d]"], value="")

    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string and convert column from string to numeric
        categories[column] = categories[column].replace(regex=["[^\d]"], value="") 
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column 
    df = df.drop(columns="categories")
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    # save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')   

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()