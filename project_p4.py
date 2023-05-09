# Mental Health Chatbot
# Daniel Valencia
# =========================================================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk

# Add pre-trained word2vec representation file named w2v.pkl to project directory or update file path accordingly
EMBEDDING_FILE = "w2v.pkl"

# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)

# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: a list of document strings, and a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info
# user_input: A string of arbitrary length
# Returns: user's name as string
def extract_user_info(user_input):
    match = re.search(r"(^|\s)[A-Z][A-Za-z.\-&\']*\s[A-Z][A-Za-z.\-&\']*(\s[A-Z][A-Za-z.\-&\']*){0,2}", user_input)
    if match:
        user_name = match.group()
    else:
        user_name = ""

    # removes leading/trailing whitespace
    name = user_name.strip()

    return name


# Function to convert a given string into a list of tokens
# Args:
# inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: preprocessing
# Args:
# user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    # Tokenizes the string and removes tokens with punctuation
    punc = string.punctuation
    tokens = get_tokens(user_input)
    no_punc_tokens = []

    for token in tokens:
        if len(token) == 1:
            if token not in punc:
                no_punc_tokens.append(token)
        else:
            no_punc_tokens.append(token)

    # Splits string into tokens and converts them to lowercase
    lower_tokens = [x.lower() for x in no_punc_tokens]

    # Creates modified string of space-separated tokens
    modified_input = " ".join(lower_tokens)
    return modified_input


# Function: vectorize_train
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()

    # Preprocess training docs
    processed_docs = []
    for doc in training_documents:
        p_doc = preprocessing(doc)
        processed_docs.append(p_doc)

    # Builds sparse document-term matrix for all docs and terms
    tfidf_train = vectorizer.fit_transform(processed_docs)

    return vectorizer, tfidf_train


# Function: vectorize_test
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    # Preprocess user input and transform it into sparse TFIDF representation
    processed_input = preprocessing(user_input)
    tfidf_test = vectorizer.transform([processed_input])

    return tfidf_test


# Function: train_nb_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    nb_model = GaussianNB()

    # Convert sparse matrix to dense numpy array and train GaussianNB model
    np_arr = training_data.toarray()
    nb_model.fit(np_arr, training_labels)

    return nb_model

# Function: get_model_prediction(nb_model, tfidf_test)
# nb_model: A trained GaussianNB model
# tfidf_test: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
# Returns: A predicted label for the provided test data (int, 0 or 1)
def get_model_prediction(nb_model, tfidf_test):
    # Initialize the output label
    label = 0

    # Convert sparse matrix to dense numpy array and predict
    np_arr = tfidf_test.toarray()
    label = nb_model.predict(np_arr)

    return label


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    # Default array if token does not exist in the dictionary
    word_vector = np.zeros(300,)

    # If token exists, then the token's corresponding array value will be returned
    if token in word2vec:
        word_vector = word2vec[token]

    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300,)

    # Preprocess user input and split the string into list of tokens
    proc_token = preprocessing(user_input)
    tokens = get_tokens(proc_token)

    # find the corresponding word vector for each token and add each array to a list to create a matrix of arrays
    word_vecs = []
    for token in tokens:
        word_vec = w2v(word2vec, token)
        word_vecs.append(word_vec)

    # find average of each column in the matrix and store average values into new numpy array
    n_arr = np.array(word_vecs)
    embedding = np.average(n_arr, axis = 0)

    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the three imported machine learning models, and
# returns them for later downstream use
def instantiate_models():
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)

    return logistic, svm, mlp


# Function: train_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model(model, word2vec, training_documents, training_labels):
    embeddings = []

    # preprocess each training document and add it to new list
    for doc in training_documents:
        embedding = string2vec(word2vec, doc)
        embeddings.append(embedding)

    # train model based on list of embeddings from Word2Vec
    model.fit(embeddings, training_labels)

    return model


# Function: test_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    embeddings = []

    # preprocess test documents and add them to new list of embeddings
    for doc in test_documents:
        embedding = string2vec(word2vec, doc)
        embeddings.append(embedding)

    # run prediction labels and use those for each model score
    pred_labels = model.predict(embeddings)
    precision = precision_score(test_labels, pred_labels)
    recall = recall_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels)
    accuracy = accuracy_score(test_labels, pred_labels)

    return precision, recall, f1, accuracy


# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    tokens = nltk.tokenize.word_tokenize(user_input)

    punc = string.punctuation
    no_punc_tokens = []

    for token in tokens:
        if len(token) == 1:
            if token not in punc:
                no_punc_tokens.append(token)
        else:
            no_punc_tokens.append(token)

    num_words = len(no_punc_tokens)

    return num_words

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    sum = 0

    sents = nltk.tokenize.sent_tokenize(user_input)
    for sent in sents:
        count = count_words(sent)
        sum += count

    num_sents = len(sents)
    if num_sents == 0:
        wps = 0.0
    else:
        wps = sum / num_sents

    return wps


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []
    tokens = nltk.tokenize.word_tokenize(user_input)

    tagged_input = nltk.pos_tag(tokens)

    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    pronouns = ['PRP', 'PRP$', 'WP', 'WP$']
    past = ['VBD', 'VBN']
    for token in tagged_input:
        if token[1] == 'PRP':
            num_prp += 1

        if token[1] in pronouns:
            num_pronouns += 1
        elif token[1] == 'DT':
            num_articles += 1
        elif token[1] in past:
            num_past += 1
        elif token[1] == 'MD':
            num_future += 1
        elif token[1] == 'IN':
            num_prep += 1

    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0
    tokens = nltk.tokenize.word_tokenize(user_input)

    negations = ['no', 'not', 'never']
    for token in tokens:
        if token in negations:
            num_negations += 1

    neg_list = re.findall('[A-Za-z]+n\'t', user_input)
    count = len(neg_list)
    num_negations += count

    return num_negations


# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep,
#               num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    if num_words > num_words_threshold:
        informative_correlates.append(psychological_correlates["num_words"])
    if wps > wps_threshold:
        informative_correlates.append(psychological_correlates["wps"])

    counts = {
    "pronouns": num_pronouns,
    "prp": num_prp,
    "articles": num_articles,
    "past": num_past,
    "future": num_future,
    "prep": num_prep,
    "negations": num_negations
    }

    while len(informative_correlates) < 3:
        max_key = max(counts, key=counts.get)

        if max_key == "pronouns":
            informative_correlates.append(psychological_correlates["num_pronouns"])
        elif max_key == "prp":
            informative_correlates.append(psychological_correlates["num_prp"])
        elif max_key == "articles":
            informative_correlates.append(psychological_correlates["num_articles"])
        elif max_key == "past":
            informative_correlates.append(psychological_correlates["num_past"])
        elif max_key == "future":
            informative_correlates.append(psychological_correlates["num_future"])
        elif max_key == "prep":
            informative_correlates.append(psychological_correlates["num_prep"])
        elif max_key == "negations":
            informative_correlates.append(psychological_correlates["num_negations"])

        counts.pop(max_key)

    return informative_correlates

# Function: welcome_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states
def welcome_state():
    print("Welcome! Thank you for chatting with me today. I am eager to know how you have been feeling.")

    return "get_name"


# Function: get_name_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that requests the user's name and then
# processes the user's response to extract the name.
def get_name_state():
    # Request the user's name and accept a user response of arbitrary length
    user_input = input("Could I please have your full name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    # Show name and thank the user
    user_input = print(f"Thanks {name}!")

    return "sentiment_analysis"


# Function: sentiment_analysis_state
# model: The trained classification model used for predicting sentiment (best one)
# word2vec: The word2vec dictionary
# Returns: A string indicating the next state
#
# This function implements a state that asks the user for input and predicts their sentiment
def sentiment_analysis_state(model, word2vec, first_time=False):
    # Check the user's current sentiment
    user_input = input("How have you been feeling the past 2 weeks?\n")

    # Predict user's sentiment
    w2v_test = string2vec(word2vec, user_input)

    label = None
    label = model.predict(w2v_test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    # helps determine appropriate next state
    if first_time == True:
        return "stylistic_analysis"

    return "check_next_state"


# Function: stylistic_analysis_state
# This function does not take any arguments
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative linguistic correlates to
# psychological status.
def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")

    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)


    print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
          "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
       num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)

    print("Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))

    print()
    return "check_next_state"


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the sentiment analysis
# ("sentiment_analysis"), or redo the stylistic analysis ("stylistic_analysis").
def check_next_state():
    # give user list of options to choose from
    print("1. Redo sentiment analysis")
    print("2. Redo stylistic analysis")
    print("3. Quit the chatbot")
    user_input = input("Please type in the number of the option you wish to continue with: \n")

    # check for incorrect input
    while int(user_input) > 3 or int(user_input) < 1:
        user_input = input("Incorrect option entered. Please try again with a number between 1 - 3.\n")

    # select next state
    if user_input == "1":
        next_state = "sentiment_analysis"
    elif user_input == "2":
        next_state = "stylistic_analysis"
    else:
        next_state = "quit"

    return next_state


# Function: run_chatbot
# model: A trained classification model
# word2vec: The pretrained Word2Vec dictionary (leave empty if not using word2vec based model)
# Returns: This function does not return any values
#
# This function implements the main chatbot system --- it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state()
# (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, word2vec):
    get_name = welcome_state()
    sentiment = get_name_state()

    style = sentiment_analysis_state(model, word2vec, True)
    next_state = stylistic_analysis_state()

    option = check_next_state()
    while option != "quit":
        if option == "sentiment_analysis":
            next_state = sentiment_analysis_state(model, word2vec, False)
            option = check_next_state()
            continue
        elif option == "stylistic_analysis":
            next_state = stylistic_analysis_state()
            option = check_next_state()
            continue

    return


if __name__ == "__main__":

    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)

    # Instantiate and train the machine learning models
    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, documents, labels)
    svm = train_model(svm, word2vec, documents, labels)
    mlp = train_model(mlp, word2vec, documents, labels)

    # Test the machine learning models to see how they perform on the small test set provided.
    # Write a classification report to a CSV file with this information.
    # Loading the dataset
    test_documents, test_labels = load_as_list("test.csv")
    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row
    i = 0
    while i < len(models): # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_documents, test_labels)
        if models[i] == None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i],"N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()

    run_chatbot(mlp, word2vec)