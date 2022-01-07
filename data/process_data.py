import sys

#import other libraries
import os
import pandas as pd
import numpy as np
import chardet
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Description: This method load the file from two csv files into a Dataframe
    Args:
        messages_filepath abd categories_filepath

    Return: Dataframe
    '''
    # load messages dataset
    try:
        if os.path.exists(messages_filepath) and os.path.exists(categories_filepath):
            #reate the data infor a dataframe
            messages = pd.read_csv(messages_filepath)
            categories = pd.read_csv(categories_filepath)

            # merge the messages and categories datasets based on id field
            df = messages.merge(categories, on='id')
    except:
        print("An Error occured while reading the csv file")

    return df


def clean_data(df):
    '''
    Description: This method performs a cleaning operation on the dataset
    Args:
        df - Dataframe

    Return: Dataframe
    '''
    #create a dataframe of the 36 individual category columns
    #Converte the column values to a string and split based on ;
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use row data to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =  list(row.apply(lambda x: x.split('-')[0]))
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to either 0 or 1 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")
    
    #Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates from the dataset if any
    duplicates = df.duplicated().count()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Description: This method will save the data from the dataframe into a newly created database
    Args:
        df - Dataframe
        database_filename: The Sqlite Database file (in this case)

    Return: None
    '''
    database_filename = 'sqlite:///{}'.format(database_filename)
    engine = create_engine(database_filename)
    df.to_sql('messages_categories', con=engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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