# Gender-Predictor
Consider a labeled dataset of names and their respective gender. Apply suitable data preprocessing
steps such as handling of null values, data reduction, discretization. For prediction of class labels of given
data instances, build classifier models using different techniques (Naïve Bayes Algorithm, Decision Tree Classifier, Random Forest Classifier).

Analyze the confusion matrix and compare these models. Also apply cross validation while
preparing the training and testing datasets.

METHODOLOGY :
1. Import necessary libraries.

2. Load the dataset – Read the csv file

3. Data Pre-processing – remove null values and missing data replace them with the mean value, standard
deviation or the most common values. It also includes dropping of any irrelevant data.

4. Transforming data into numeric categorical data converting categories male, female into 0 or 1.

5. Split the dataset into training and testing data

6. Building the classifier models.

7. Compare accuracies to find which classification model has the best accuracy.

8. Determine confusion matrix.

DATASET :
Names Dataset with their respective gender.

REQUIREMENTS :

1     SOFTWARE REQUIREMENTS:
                            Windows/Linux Operating System
                            Juypter Notebook
                            Python 3

2 LIBRARIES/PACKAGES USED:
                            Numpy
                            Pandas
                            Sklearn

CONCLUSION:
Thus, we applied three different classification algorithms (Decision Tree, Random Forest and Naive Bayes) on the Indian names’ dataset. The efficiency of Naive Bayes and Random Forest classifiers are nearly equal which is less than the efficiency of Decision Tree classifier.
