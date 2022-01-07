# Disaster Response Pipeline Project
![Data Science Udacity Nanodegree](https://miro.medium.com/max/2000/1*BFD7xbM1Db0UiilLuTHuYA.jpeg)
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Visit my Medium page. ![My Medium page](https://medium.com/@patlichengine/building-machine-learning-model-for-solving-real-world-problems-1a2dcd2174a8)
