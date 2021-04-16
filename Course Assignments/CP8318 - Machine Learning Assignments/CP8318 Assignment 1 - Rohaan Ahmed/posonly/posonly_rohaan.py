import importlib
import numpy as np
import util
import utils
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import lr_classifier
from lr_classifier import LogisticRegression


# def main(train_path, valid_path, test_path, save_path):
#     """Problem 2: Logistic regression for incomplete, positive-only labels.

#     Run under the following conditions:
#         1. on t-labels,
#         2. on y-labels,
#         3. on y-labels with correction factor alpha.

#     Args:
#         train_path: Path to CSV file containing training set.
#         valid_path: Path to CSV file containing validation set.
#         test_path: Path to CSV file containing test set.
#         save_path: Path to save predictions.
#     """
#     output_path_true = save_path.replace(WILDCARD, 'true')
#     output_path_naive = save_path.replace(WILDCARD, 'naive')
#     output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

#     # *** START CODE HERE ***



#     # *** END CODER HERE

# if __name__ == '__main__':
#     main(train_path='train.csv',
#         valid_path='valid.csv',
#         test_path='test.csv',
#         save_path='posonly_X_pred.txt')


train_path='train.csv'
valid_path='valid.csv'
test_path='test.csv'
save_path='posonly_X_pred.txt'
WILDCARD = 'X'

output_path_true = save_path.replace(WILDCARD, 'true')
output_path_naive = save_path.replace(WILDCARD, 'naive')
output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

## LOGISTIC REGRESSION USING ALL TRUE LABELS ##

step_size=0.03
max_iter=int(1e6)
eps=1e-5

### QUESTION 2 ###

# Part (a): Train and test on true labels
# Make sure to save predicted probabilities to output_path_true using np.savetxt()

model = LogisticRegression(step_size=step_size, max_iter=max_iter, eps=eps)

x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
theta = model.fit(x_train, y_train)
h_x_train = model.predict(x_train)
h_x_train = h_x_train.round(decimals=0, out=None)
util.plot(x_train, y_train, theta, save_path='test.png', correction=1.0)
plt.title('Trained = T-Labels, Evaluated = T-Labels Train')
plt.legend()
plt.savefig('Plots/2_1_1.png')
plt.show()

x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
h_x_test = model.predict(x_test)
np.savetxt(fname=output_path_true, X=h_x_test)
h_x_test = h_x_test.round(decimals=0, out=None)
util.plot(x_test, y_test, theta, save_path='test.png', correction=1.0)
plt.title('Trained = T-Labels, Evaluated = T-Labels Test')
plt.legend()
plt.savefig('Plots/2_1_2.png')
plt.show()

utils.plot(x_test[:,1:], y_test, h_x_test)
plt.savefig('Plots/2_1_3.png')
plt.show()

cm = confusion_matrix(y_test, h_x_test)
accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
accuracy = "{:.0%}".format(accuracy)
ax = plt.axes()
sns.heatmap(cm, annot=True, ax = ax)
ax.set_title('2.1 - Confusion Matrix. Accuracy = ' + accuracy)
plt.savefig('Plots/2_1_4.png')
plt.show()

# Part (b): Train on y-labels and test on true labels
# Make sure to save predicted probabilities to output_path_naive using np.savetxt()

x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
theta = model.fit(x_train, y_train)
h_x_train = model.predict(x_train)
h_x_train = h_x_train.round(decimals=0, out=None)
util.plot(x_test, y_test, theta, save_path='test.png', correction=1.0)
plt.title('Trained = Y-Labels, Evaluated = Y-Labels Train')
plt.legend()
plt.savefig('Plots/2_2_1.png')
plt.show()

x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
h_x_test = model.predict(x_test)
np.savetxt(fname=output_path_naive, X=h_x_test)
h_x_test = h_x_test.round(decimals=0, out=None)
util.plot(x_test, y_test, theta, save_path='test.png', correction=1.0)
plt.title('Trained = Y-Labels, Evaluated = T-Labels Test')
plt.legend()
plt.savefig('Plots/2_2_2.png')
plt.show()

cm = confusion_matrix(y_test, h_x_test)
accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
accuracy = "{:.0%}".format(accuracy)
ax = plt.axes()
sns.heatmap(cm, annot=True, ax = ax)
ax.set_title('2.2 - Confusion Matrix. Accuracy = ' + accuracy)
plt.savefig('Plots/2_2_3.png')
plt.show()

utils.plot(x_test[:,1:], y_test, h_x_test)
plt.savefig('Plots/2_2_4.png')
plt.show()


# Part (f): Apply correction factor using validation set and test on true labels
# Plot and use np.savetxt to save outputs to output_path_adjusted

x_val, y_val = util.load_dataset(train_path, label_col='y', add_intercept=True)
h_x_val_p = model.predict(x_val)
h_x_val = h_x_val_p.round(decimals=0, out=None)
util.plot(x_val, y_val, theta, save_path='test.png', correction=1.0)
plt.title('Trained = Y-Labels, Evaluated = Y-Labels Validation')
plt.legend()
plt.savefig('Plots/2_6_1.png')
plt.show()

# Before Adjustment
x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
h_x_test = model.predict(x_test)
h_x_test = h_x_test.round(decimals=0, out=None)
util.plot(x_test, y_test, theta, save_path='test.png', correction=1.0)
plt.title('Trained = Y-Labels, Evaluated = T-Labels Test before Adjustment')
plt.legend()
plt.savefig('Plots/2_6_2.png')
plt.show()

utils.plot(x_test[:,1:], y_test, h_x_test)
plt.savefig('Plots/2_6_3.png')
plt.show()

cm = confusion_matrix(y_test, h_x_test)
accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
accuracy = "{:.0%}".format(accuracy)
ax = plt.axes()
sns.heatmap(cm, annot=True, ax = ax)
ax.set_title('2.6 - Confusion Matrix. Accuracy before Adjustment = ' + accuracy)
plt.savefig('Plots/2_6_4.png')
plt.show()

# Calculating Adjustment Factor Alpha

vp = x_val[y_val == 1][:, :]
h_x_val_pos = h_x_val_p[y_val == 1]
alpha = h_x_val_pos.sum() / len(vp)

# After Adjustment

x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
h_x_test = model.predict(x_test)
h_x_test = h_x_test / alpha
np.savetxt(fname=output_path_adjusted, X=h_x_test)
h_x_test = h_x_test.round(decimals=0, out=None)
h_x_test = np.where(h_x_test > 1, 1, h_x_test)
util.plot(x_test, h_x_test, theta, save_path='test.png', correction=alpha)
plt.title('Trained = Y-Labels, Evaluated = T-Labels Test after Adjustment')
plt.legend()
plt.savefig('Plots/2_6_5.png')
plt.show()

utils.plot(x_test[:,1:], y_test, h_x_test)
plt.savefig('Plots/2_6_6.png')
plt.show()

cm = confusion_matrix(y_test, h_x_test)
accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
accuracy = "{:.0%}".format(accuracy)
ax = plt.axes()
sns.heatmap(cm, annot=True, ax = ax)
ax.set_title('2.6 - Confusion Matrix. Accuracy after Adjustment = ' + accuracy)
plt.savefig('Plots/2_6_7.png')
plt.show()

### Part (f) Using Gradient Descent instead of Newton's Method ###

x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
theta_2 = model.fit_gradient(x_train, y_train)
h_x_train = model.predict(x_train)
h_x_train = h_x_train.round(decimals=0, out=None)
util.plot(x_test, y_test, theta, save_path='test.png', correction=1.0)
plt.title('Trained = Y-Labels, Evaluated = Y-Labels Train')
plt.legend()
plt.savefig('Plots/grad_desc_1.png')
plt.show()

x_val, y_val = util.load_dataset(train_path, label_col='y', add_intercept=True)
h_x_val_p = model.predict(x_val)
h_x_val = h_x_val_p.round(decimals=0, out=None)
util.plot(x_val, y_val, theta, save_path='test.png', correction=1.0)
plt.title('Trained = Y-Labels, Evaluated = Y-Labels Validation')
plt.legend()
plt.savefig('Plots/grad_desc_2.png')
plt.show()

# Calculating Adjustment Factor Alpha

vp = x_val[y_val == 1][:, :]
h_x_val_pos = h_x_val_p[y_val == 1]
alpha = h_x_val_pos.sum() / len(vp)

# After Adjustment

x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
h_x_test = model.predict(x_test)
h_x_test = h_x_test / alpha
np.savetxt(fname=output_path_adjusted, X=h_x_test)
h_x_test = h_x_test.round(decimals=0, out=None)
h_x_test = np.where(h_x_test > 1, 1, h_x_test)
util.plot(x_test, h_x_test, theta, save_path='test.png', correction=alpha)
plt.title('Trained = Y-Labels, Evaluated = T-Labels Test after Adjustment')
plt.legend()
plt.savefig('Plots/grad_desc_3.png')
plt.show()

cm = confusion_matrix(y_test, h_x_test)
accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
accuracy = "{:.0%}".format(accuracy)
ax = plt.axes()
sns.heatmap(cm, annot=True, ax = ax)
ax.set_title('2.6 - Confusion Matrix. Accuracy after Adjustment = ' + accuracy)
plt.savefig('Plots/grad_desc_4.png')
plt.show()