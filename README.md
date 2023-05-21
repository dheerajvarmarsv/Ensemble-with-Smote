# Ensemble with Smote
 This Jupyter Notebook contains a Python script that demonstrates the process of predicting insurance fraud using various machine learning algorithms and ensemble methods. It also incorporates the use of SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance in the dataset.
 
 Here's a breakdown of the code:

Mounting Google Drive: The code mounts the Google Drive to the Colab environment to access the datasets stored in Google Drive.

Importing Libraries: The necessary libraries are imported, including vecstack, pandas, numpy, and various modules from sklearn for classification algorithms, hyperparameter tuning, and evaluation metrics. The code also imports the SMOTE module from imblearn for oversampling.

Loading Datasets: The code reads the training and test datasets from Google Drive and stores them in train_data and test_data variables, respectively.

Preprocessing: The code separates the target column from the training data and assigns them to X_train1 and y_train variables. It also creates a separate variable X_test1 for the test data.

Decision Tree Classifier: The code constructs a default decision tree classifier, fits it with the training data, and predicts the target variable for the test data. It then prints the accuracy score, confusion matrix, and performs hyperparameter tuning using RandomizedSearchCV.

Random Forest Classifier: Similar to the decision tree classifier, the code constructs a random forest classifier, performs hyperparameter tuning, and prints the accuracy score, confusion matrix, and classification report.

MultiLayer Perceptron Classifier (MLP): The code constructs an MLP classifier, performs hyperparameter tuning, and prints the accuracy score, confusion matrix, and classification report.

K-Nearest Neighbor Classifier (KNN): The code constructs a KNN classifier, performs hyperparameter tuning, and prints the accuracy score, confusion matrix, and classification report.

Linear Support Vector Machine Classifier (LinearSVC): The code constructs a linear SVM classifier, fits it with the training data, and predicts the target variable for the test data. It then prints the accuracy score and confusion matrix.

Support Vector Machine Classifier (SVC): The code constructs an SVM classifier, fits it with the training data, and predicts the target variable for the test data. It then prints the accuracy score and confusion matrix.

Gradient Boosting Classifier: The code constructs a gradient boosting classifier, fits it with the training data, and predicts the target variable for the test data. It then prints the accuracy score, confusion matrix, and performs hyperparameter tuning using RandomizedSearchCV.

SMOTE (Synthetic Minority Oversampling Technique): The code applies SMOTE to the training data to balance the classes by oversampling the minority class.

Ensemble Methods - Stacking: The code performs stacking using ensemble methods such as gradient boosting, random forest, decision tree, KNN, MLP, and SVM classifiers. It trains the ensemble model using the stacked features obtained from the previous models and predicts the target variable for the test data. It prints the accuracy score for each ensemble model.

Prediction Probability: The code calculates the prediction probabilities for the predicted classes and stores them in a dataframe called pred_Probability.

Please note that this code assumes that the necessary datasets are available in the specified paths on Google Drive, and it uses the training data to train the models and evaluates them on the test data.
