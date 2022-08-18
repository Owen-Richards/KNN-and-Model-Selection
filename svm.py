
# Starting code for HW5 SVM

import numpy as np
np.random.seed(37)
import random
import numpy as np
import pandas
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']

# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing. 
# For example, as a start you can use one hot encoding for the categorical variables and normalization 
# for the continuous variables. Also, look out for missing values. 

def load_data(csv_file_path):
    # your code here
    data = pandas.read_csv(csv_file_path, sep=', ', names=col_names_x+col_names_y, engine='python')

    label = None
    if not label:
        label = LabelEncoder()
        label.fit(['<=50K', '>50K'])
    y = label.transform(data[col_names_y[0]].values)

    standardize = None
    numerical = data[numerical_cols].values
    if not standardize:
        standardize = StandardScaler()
        standardize.fit(numerical)
    numerical = standardize.transform(numerical)

    onehot = None
    # X has 108 features, but SVC is expecting 107 features as input.
    filtered_categorical = categorical_cols[:-1]
    category = data[filtered_categorical].values
    if not onehot:
        onehot = OneHotEncoder(sparse=False)
        onehot.fit(category)
    category = onehot.transform(category)
    x = np.concatenate([numerical, category], axis=1)
    return x, y   


# Fold function taken from the knn colab file
# Same as in the colab
def fold(x, y, i, three_folds):

    sizeof_fold = int(len(x) / three_folds)

    x_train = np.concatenate([ x[:i * sizeof_fold], x[(i + 1) * sizeof_fold:]])
    x_test = x[i * sizeof_fold:(i + 1) * sizeof_fold]

    y_train = np.concatenate([y[:i * sizeof_fold],y[(i + 1) * sizeof_fold:]])
    y_test = y[i * sizeof_fold:(i + 1) * sizeof_fold]

    return x_train, y_train, x_test, y_test

# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    x_train, y_train = load_data(training_csv)
    # hard code hyperparameter configurations, an example:
    param_set = [
        {'kernel': 'rbf', 'C': 1},
        {'kernel': 'rbf', 'C': 3},
        {'kernel': 'rbf', 'C': 5},
        {'kernel': 'rbf', 'C': 7},
        {'kernel': 'rbf', 'C': 9},
        {'kernel': 'rbf', 'C': 15},
        {'kernel': 'poly', 'C': 1, 'degree': 1},
        {'kernel': 'poly', 'C': 1, 'degree': 3},
        {'kernel': 'poly', 'C': 1, 'degree': 5},
        {'kernel': 'poly', 'C': 1, 'degree': 7},
        {'kernel': 'poly', 'C': 3, 'degree': 1},
        {'kernel': 'poly', 'C': 3, 'degree': 3},
        {'kernel': 'poly', 'C': 3, 'degree': 5},
        {'kernel': 'poly', 'C': 3, 'degree': 7},
        {'kernel': 'linear', 'C': 1},
        {'kernel': 'linear', 'C': 3},
        {'kernel': 'linear', 'C': 5},
        {'kernel': 'linear', 'C': 7},
    ]
    # your code here
    # iterate over all hyperparameter configurations
    # perform 3 FOLD cross validation
    # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model 
    best_score = 0
    best_parameter = None
    three_folds = 3

    for parameter in param_set:
        train_accuracy = []
        test_accuracy = []
        for i in range(three_folds):

            x_train_fold, y_train_fold, x_test_fold, y_test_fold = fold(x_train, y_train, i, three_folds)
            model = SVC(**parameter, gamma='auto')
            model.fit(x_train_fold, y_train_fold)
            train_accuracy.append(model.score(x_train_fold, y_train_fold))
            test_accuracy.append(model.score(x_test_fold, y_test_fold))

        test_acc = sum(test_accuracy) / three_folds
        train_acc = sum(train_accuracy) / three_folds
        print(parameter, 'train accuracy: %.8f' % train_acc, 'validation accuracy: %.8f' % test_acc)

        if test_acc > best_score:
            best_score = test_acc
            best_parameter = parameter

    best_model = SVC(**best_parameter, gamma='auto')
    best_model.fit(x_train, y_train)
    return best_model, best_score

# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format 
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # fill in train_and_select_model(training_csv) to 
    # return a trained model with best hyperparameter from 3-FOLD 
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter. 
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(training_csv)

    print("The best model was scored : ",cv_score)
    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.