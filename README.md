# Mental Health Chatbot

This is a Python based mental health chatbot that asks users how they have been feeling lately, and it performs sentiment and sylistic
analysis on their input. The program instantiates and trains a variety of machine learning models, but it ultimately uses a multi-layer
perceptron model to ensure accuracy.

The program requires the use of a pretrained Word2Vec model from Google which can be downloaded and either placed in the same directory as 
the program or elsewhere. If placed elsewhere, the EMBEDDING_FILE path name will need to be modified accordingly. The csv files are there
to help train and test the ML models.
