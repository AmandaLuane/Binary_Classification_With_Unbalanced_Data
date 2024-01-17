import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
%matplotlib inline
random_state = 0
import time

AmesFeatures = pd.read_csv('./AmesFeatures.csv')
y = (AmesFeatures['SalePrice'] > 200000) & (AmesFeatures['SalePrice'] < 230000)
X = AmesFeatures.drop(columns=['Id', 'SalePrice'])
display(y.value_counts())
display(X.shape)
display(X)

# Splt data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = random_state)

# Present the imbalance of the class labels for both test and training
print(f"TRAINING SET: {display(y_train.value_counts())}\n")
print(f"TEST SET: {display(y_test.value_counts())}")

# Train the logistic regression
from sklearn.linear_model import LogisticRegression

# Create an Instance
logreg = LogisticRegression()
# Train 
logreg.fit(X_train, y_train)

# Predict (hard and soft) on training and test features

# Pass the Training Data to Get the Final Predictions
y_train_pred = logreg.predict(X_train)
# Pass the Training Data to Get the Raw Probabilites
y_train_proba = logreg.predict_proba(X_train)[:,1]

# Pass the Testing Data to Get the Final Predictions
y_test_pred = logreg.predict(X_test)
# Pass the Testing Data to Get the Raw Probabilites
y_test_proba = logreg.predict_proba(X_test)[:,1]

# Evaluate predictions using confusion matrix and its metrics
from sklearn import metrics
from sklearn.metrics import classification_report

# Confusion Matrix
cm_train = metrics.confusion_matrix(y_train, y_train_pred)
cm_test = metrics.confusion_matrix(y_test, y_test_pred)

# Matrix ~ Classification Report
cr_train = classification_report(y_train, y_train_pred)
cr_test = classification_report(y_test, y_test_pred)

print(f"CONFUSION MATRIX ~ TRAINING SET:\n{cm_train}\n")
print(f"CONFUSION MATRIX ~ TRAINING SET:\n{cm_test}\n")
print (f"\nCLASSIFICATON REPORT ~ TRAINING SET:\n{cr_train}\n")
print (f"\nCLASSIFICATON REPORT ~ TEST SET:\n{cr_test}")

# Evaluate predictions using ROC For Training
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_proba)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC FOR TRAINING SET')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Evaluate predictions using AUC of ROC For Test
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('ROC FOR TEST SET')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Evaluate predictions using AUC of ROC For TRAINING & TEST
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, label=f'Training Set (AUC = {roc_auc_train:.2f})')
plt.plot(fpr_test, tpr_test, label=f'Test Set (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE FOR TRAINING & TEST SETS')
plt.legend()
plt.show()

# Print AUC of ROC
print("\nAUC OF ROC:")
print("TRAINING SET:", roc_auc_train)
print("TEST SET:", roc_auc_test)


# Train the logistic regression with balanced weights
# Create an Instance
logreg_bal = LogisticRegression(class_weight='balanced', random_state=random_state)
# Train 
logreg_bal.fit(X_train, y_train)


# Predict (hard and soft) on training and test features

# Pass the Training Data to Get the Final Predictions
y_train_pred_bal = logreg_bal.predict(X_train)
# Pass the Training Data to Get the Raw Probabilites
y_train_proba_bal = logreg_bal.predict_proba(X_train)[:,1]

# Pass the Testing Data to Get the Final Predictions
y_test_pred_bal = logreg_bal.predict(X_test)
# Pass the Testing Data to Get the Raw Probabilites
y_test_proba_bal = logreg_bal.predict_proba(X_test)[:,1]


# Evaluate predictions using confusion matrix and its metrics

# Confusion Matrix
cm_train_bal = metrics.confusion_matrix(y_train, y_train_pred_bal)
cm_test_bal = metrics.confusion_matrix(y_test, y_test_pred_bal)

# Matrix ~ Classification Report
cr_train_bal = classification_report(y_train, y_train_pred_bal)
cr_test_bal = classification_report(y_test, y_test_pred_bal)

print(f"CONFUSION MATRIX W/BALANCED WEIGHTS ~ TRAINING SET:\n{cm_train_bal}\n")
print(f"CONFUSION MATRIX W/BALANCED WEIGHTS ~ TRAINING SET:\n{cm_test_bal}\n")
print (f"\nCLASSIFICATON REPORT W/BALANCED WEIGHTS ~ TRAINING SET:\n{cr_train_bal}\n")
print (f"\nCLASSIFICATON REPORT W/BALANCED WEIGHTS ~ TEST SET:\n{cr_test_bal}")



# Evaluate predictions using ROC For Training
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_proba_bal)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC FOR TRAINING SET W/BALANCED WEIGHTS')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Evaluate predictions using AUC of ROC For Test
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba_bal)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('ROC FOR TEST SET W/BALANCED WEIGHTS')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Evaluate predictions using AUC of ROC For TRAINING & TEST
fpr_train_bal, tpr_train_bal, _ = roc_curve(y_train, y_train_proba_bal)
roc_auc_train_bal = auc(fpr_train_bal, tpr_train_bal)

fpr_test_bal, tpr_test_bal, _ = roc_curve(y_test, y_test_proba_bal)
roc_auc_test_bal = auc(fpr_test_bal, tpr_test_bal)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_train_bal, tpr_train_bal, label=f'Training Set (AUC = {roc_auc_train_bal:.2f})')
plt.plot(fpr_test_bal, tpr_test_bal, label=f'Test Set (AUC = {roc_auc_test_bal:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE FOR TRAINING & TEST SETS W/BALANCED WEIGHTS')
plt.legend()
plt.show()

# Print AUC of ROC
print("\nAUC OF ROC T W/BALANCED WEIGHTS:")
print("TRAINING SET W/BALANCED WEIGHTS:", roc_auc_train_bal)
print("TEST SET W/BALANCED WEIGHTS:", roc_auc_test_bal)

# Train logistic regression cross-validation (cv = 5)
# Create an Instance
logreg_cv = LogisticRegressionCV(cv=5, class_weight='balanced', random_state=random_state)
# Train 
logreg_cv.fit(X_train, y_train)

# Predict (hard and soft) on training and test features

# Pass the Training Data to Get the Final Predictions
y_train_pred_cv = logreg_cv.predict(X_train)
# Pass the Training Data to Get the Raw Probabilites
y_train_proba_cv = logreg_cv.predict_proba(X_train)[:,1]

# Pass the Testing Data to Get the Final Predictions
y_test_pred_cv = logreg_cv.predict(X_test)
# Pass the Testing Data to Get the Raw Probabilites
y_test_proba_cv = logreg_cv.predict_proba(X_test)[:,1]

# Evaluate predictions using confusion matrix and its metrics

# Confusion Matrix
cm_train_cv = metrics.confusion_matrix(y_train, y_train_pred_cv)
cm_test_cv = metrics.confusion_matrix(y_test, y_test_pred_cv)

# Matrix ~ Classification Report
cr_train_cv = classification_report(y_train, y_train_pred_cv)
cr_test_cv = classification_report(y_test, y_test_pred_cv)

print(f"CONFUSION MATRIX W/CROSS-VALIDATION ~ TRAINING SET:\n{cm_train_cv}\n")
print(f"CONFUSION MATRIX W/CROSS-VALIDATION ~ TRAINING SET:\n{cm_test_cv}\n")
print (f"\nCLASSIFICATON REPORT W/CROSS-VALIDATION ~ TRAINING SET:\n{cr_train_cv}\n")
print (f"\nCLASSIFICATON REPORT W/CROSS-VALIDATION ~ TEST SET:\n{cr_test_cv}")

# Evaluate predictions using ROC For Training
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_proba_cv)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC FOR TRAINING SET W/CROSS-VALIDATION')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Evaluate predictions using AUC of ROC For Test
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba_cv)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('ROC FOR TEST SET W/CROSS-VALIDATION')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Evaluate predictions using AUC of ROC For TRAINING & TEST
fpr_train_cv, tpr_train_cv, _ = roc_curve(y_train, y_train_proba_cv)
roc_auc_train_cv = auc(fpr_train_cv, tpr_train_cv)

fpr_test_cv, tpr_test_cv, _ = roc_curve(y_test, y_test_proba_cv)
roc_auc_test_cv = auc(fpr_test_cv, tpr_test_cv)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_train_cv, tpr_train_cv, label=f'Training Set (AUC = {roc_auc_train_cv:.2f})')
plt.plot(fpr_test_cv, tpr_test_cv, label=f'Test Set (AUC = {roc_auc_test_cv:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE FOR TRAINING & TEST SETS W/CROSS-VALIDATION')
plt.legend()
plt.show()

# Print AUC of ROC
print("\nAUC OF ROC W/CROSS-VALIDATION:")
print("TRAINING SET W/CROSS-VALIDATION:", roc_auc_train_cv)
print("TEST SET W/CROSS-VALIDATION:", roc_auc_test_cv)

# Train logistic regression cross-validation (cv = 10)
# Create an Instance
logreg_cv10 = LogisticRegressionCV(cv=10, class_weight='balanced', random_state=random_state)
# Train 
logreg_cv10.fit(X_train, y_train)

# Predict (hard and soft) on training and test features

# Pass the Training Data to Get the Final Predictions
y_train_pred_cv10 = logreg_cv10.predict(X_train)
# Pass the Training Data to Get the Raw Probabilites
y_train_proba_cv10 = logreg_cv10.predict_proba(X_train)[:,1]

# Pass the Testing Data to Get the Final Predictions
y_test_pred_cv10 = logreg_cv10.predict(X_test)
# Pass the Testing Data to Get the Raw Probabilites
y_test_proba_cv10 = logreg_cv10.predict_proba(X_test)[:,1]

# Evaluate predictions using confusion matrix and its metrics

# Confusion Matrix
cm_train_cv10 = metrics.confusion_matrix(y_train, y_train_pred_cv10)
cm_test_cv10 = metrics.confusion_matrix(y_test, y_test_pred_cv10)

# Matrix ~ Classification Report
cr_train_cv10 = classification_report(y_train, y_train_pred_cv10)
cr_test_cv10 = classification_report(y_test, y_test_pred_cv10)

print(f"CONFUSION MATRIX W/CROSS-VALIDATION OF 10 ~ TRAINING SET:\n{cm_train_cv10}\n")
print(f"CONFUSION MATRIX W/CROSS-VALIDATION OF 10 ~ TRAINING SET:\n{cm_test_cv10}\n")
print (f"\nCLASSIFICATON REPORT W/CROSS-VALIDATION OF 10 ~ TRAINING SET:\n{cr_train_cv10}\n")
print (f"\nCLASSIFICATON REPORT W/CROSS-VALIDATION OF 10 ~ TEST SET:\n{cr_test_cv10}")

# Evaluate model using ROC For Training
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_proba_cv10)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC FOR TRAINING SET W/CROSS-VALIDATION OF 10')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Evaluate model using ROC For Test
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba_cv10)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('ROC FOR TEST SET W/CROSS-VALIDATION OF 10')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Evaluate predictions using AUC of ROC For TRAINING & TEST
fpr_train_cv10, tpr_train_cv10, _ = roc_curve(y_train, y_train_proba_cv10)
roc_auc_train_cv10 = auc(fpr_train_cv10, tpr_train_cv10)

fpr_test_cv10, tpr_test_cv10, _ = roc_curve(y_test, y_test_proba_cv10)
roc_auc_test_cv10 = auc(fpr_test_cv10, tpr_test_cv10)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_train_cv10, tpr_train_cv10, label=f'Training Set (AUC = {roc_auc_train_cv10:.2f})')
plt.plot(fpr_test_cv10, tpr_test_cv10, label=f'Test Set (AUC = {roc_auc_test_cv10:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE FOR TRAINING & TEST SETS W/CROSS-VALIDATION OF 10')
plt.legend()
plt.show()

# Print AUC of ROC
print("\nAUC OF ROC W/CROSS-VALIDATION OF 10:")
print("TRAINING SET W/CROSS-VALIDATION OF 10:", roc_auc_train_cv10)
print("TEST SET W/CROSS-VALIDATION OF 10:", roc_auc_test_cv10)

# Determine cost of increasing folds

# Training Time - Cross-Validation 5-Fold
start_time_5_folds = time.time()
logreg_cv5 = LogisticRegressionCV(cv=5, class_weight='balanced', random_state=random_state)
logreg_cv5.fit(X_train, y_train)
end_time_5_folds = time.time()
training_time_5_folds = end_time_5_folds - start_time_5_folds

# Training Time - Cross-Validation 10-Fold
start_time_10_folds = time.time()
logreg_cv10 = LogisticRegressionCV(cv=10, class_weight='balanced', random_state=random_state)
logreg_cv10.fit(X_train, y_train)
end_time_10_folds = time.time()
training_time_10_folds = end_time_10_folds - start_time_10_folds

# Print training times
print(f"TRAINING TIME - 5 FOLD: {training_time_5_folds:.2f} seconds")
print(f"TRAINING TIME - 10 FOLD: {training_time_10_folds:.2f} seconds")

# Calculate the cost of increasing folds in terms of training run-time
fold_increase_cost = training_time_10_folds - training_time_5_folds
print(f"TIME OF INCREASING THE FOLDS: {fold_increase_cost:.2f} seconds")
