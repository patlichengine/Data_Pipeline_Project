import sys
#import other libraries
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

#import tokenization and lemmatization libraries from the nltk library
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD


def load_data(database_filepath):
    database_filepath = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('messages_categories', con=engine) 
    

    X = df.message.values
    #Y represents the rest of the columns other than the message, original, genre
    Y = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    
    #get the category column names
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token to clean all the tokens
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Description: This function defines a model pipeline which takes in the tnokenize function
    Args:
        None
    Return:
        pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    
    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 50, 100]
    }
    
    # Since most of the pipeline abov took a long time to fit the model, I had it adjusted using the 
    # TruncatedSVD and the AdaBoostClassifier algorithms to provide improvement to the pipeline.
    pipeline2 = Pipeline([
        ('vect', CountVectorizer()),
        ('best', TruncatedSVD()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # create grid search object that supports multiple processing using the n_jobs (number of parallel processes)
    grid = GridSearchCV(pipeline2, param_grid=parameters, n_jobs=5)


    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Description: Use the classification_report of the sklearn.metrics to estimate the f1, precision and recal 
    for each of the categories in the dataset.
    
    Args:
        model - Model pipeline 
        X_test - testing data from the messages 
        Y_test - testing data from the list of the categories
        category_names - the category labels or attributes
    '''
    #get the y predicted value the using the test data
    y_pred = model.predict(X_test)
    #for each of the y_test collection
        
    for col_index, numeric_col in enumerate(Y_test):
        #print the column name to the title of the classificaation report data
        if numeric_col in category_names:
            print(numeric_col)
            y_test_col = Y_test[numeric_col]

            #estimate the predict range using array slicing with col_index step
            y_pred_set = y_pred[:, col_index]
            #print(y_pred_set, ', ', col_index)

            #report the f1 score, precision and recall the column
            print(classification_report(y_test_col, y_pred_set))


def save_model(model, model_filepath):
    '''
    Description: Function that saves the final model
    
    Args:
        model: result of the GridSearchCV
        model_filepath: file path of final model
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)



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