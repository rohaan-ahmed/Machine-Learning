import numpy as np
import util
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, X, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        
        # First Method
        phi = y.mean()
        mu = np.array([ X[y==k].mean(axis=0) for k in [0,1]])
        X_u = X.copy()
        for k in [0,1]: X_u[y==k] -= mu[k] # X i mu
        Epsilon = X_u.T.dot(X_u) / len(y)
        invEpsilon = np.linalg.pinv(Epsilon)
        mu0 = mu[0,:]
        mu1 = mu[1,:]
        
        # # Another Method
        # phi = y.mean()
        # mu0 = X[y==0]
        # mu0 = mu0.sum(axis=0)/len(mu0)
        # mu1 = X[y==1]
        # mu1 = mu1.sum(axis=0)/len(mu1)
        # X_u0 = X[y==0]
        # X_u1 = X[y==1]
        # X_u0 = X_u0 - mu0
        # X_u1 = X_u1 - mu1
        # X_u = np.concatenate((X_u0, X_u1), axis=0)
        # Epsilon = (1/len(y)) * np.dot(X_u.T,X_u)
        # invEpsilon = np.linalg.pinv(Epsilon)

        theta1= -1 * invEpsilon.dot(mu0-mu1).reshape(1,-1)
        theta0=-1*np.log((1-phi)/phi)+0.5*((mu0.T.dot(invEpsilon)).dot(mu0)-mu1.T.dot(invEpsilon).dot(mu1)).reshape(1,-1)
    
        self.theta = np.concatenate((theta0, theta1), axis=1)
        
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        theta = self.theta
        theta0 = self.theta[0,0]
        theta1 = self.theta[0,1:]
     
        result = np.dot(x, theta1.T) + theta0
        result = -1 * result
        result = np.exp(result)
        result = 1 + result
        result = 1 / result
    
        return result
        
        # *** END CODE HERE
        
# if __name__ == '__main__':
#     main(train_path='ds1_train.csv',
#          valid_path='ds1_valid.csv',
#          save_path='gda_pred_1.txt')

#     main(train_path='ds2_train.csv',
#          valid_path='ds2_valid.csv',
#          save_path='gda_pred_2.txt')

for i in range (1,3):
    ds = str(i)
    train_path='ds' + ds + '_train.csv'
    valid_path='ds' + ds + '_valid.csv'
    save_path='gda_pred_'+ ds + '.txt'
    
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    clf = GDA()
    theta = clf.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)

    y_pred = clf.predict(x_eval)
    np.savetxt(fname = save_path, X = y_pred)
    
    theta = np.reshape(theta,(3,))
    util.plot(x = x_eval, y = y_eval, theta = theta, save_path = 'Plots/1_2_Plot_ds' + ds + '.png', correction=1.0)
    plt.title('1.2 Plot of ds' + ds)
    plt.legend()
    plt.savefig('Plots/1_2_Plot_ds' + ds + '.png')
    plt.show()
    
    y_pred = np.where(y_pred >= 0.5, 1, 0)

    # util.plot(x = x_eval, y = y_pred.reshape((100,)), theta = theta, save_path = 'Plots/1_2_Plot_ds' + ds + '.png', correction=1.0)
    # plt.title('y_pred')
    # plt.legend()
    # plt.show()
    
    cm = confusion_matrix(y_eval, y_pred)
    accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
    accuracy = "{:.0%}".format(accuracy)
    ax = plt.axes()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_title('1.2 - Confusion Matrix ds' + ds + '. Accuracy = ' + accuracy)
    plt.savefig('Plots/1_2_CM_ds' + ds + '.png')
    plt.show()

def create_poly(k, X) -> np.array:
    """
    Generates a polynomial feature map using the data x.
    The polynomial map should have powers from 0 to k
    Output should be a numpy array whose shape is (n_examples, k+1)

    Args:
        X: Training example inputs. Shape (n_examples, 2).
    """
    # *** START CODE HERE ***
    raw = X[:, :]  # (n_examples, 1)
    features = np.concatenate([raw ** i for i in range(1, k + 1)], axis=1)
    assert features.shape == (len(raw), (k*2))
    return features
    # *** END CODE HERE ***

def reject_outliers(x, y, m=2):
    data = np.array([x[:,0], x[:,1], y[:]]).T
    data = data[abs(data[:,0] - np.mean(data[:,0])) < m * np.std(data[:,0])]
    data = data[abs(data[:,1] - np.mean(data[:,1])) < m * np.std(data[:,1])]
    return data[:,:2].reshape(-1,2), data[:,2].reshape(-1,1)

def adjusted_for_outliers(m):
    for i in range (1,2):
        k = 1
        m = m
        ds = str(i)
        train_path='ds' + ds + '_train.csv'
        valid_path='ds' + ds + '_valid.csv'
        save_path='gda_pred_'+ ds + 'adjusted_for_outliers.txt'
        
        x_train, y_train = util.load_dataset(train_path, add_intercept=False)
        
        # x_train[:,0] = (x_train[:,0] - x_train[:,0].mean()) / x_train[:,0].std()
        # x_train[:,1] = (x_train[:,1] - x_train[:,1].mean()) / x_train[:,1].std()
        
        # x_train = reject_outliers(x_train)
        
        # x_train = create_poly(k, x_train)
        x_train, y_train = reject_outliers(x_train, y_train, m = m)
        
        clf = GDA()
        theta = clf.fit(x_train, y_train.reshape((y_train.shape[0],)))
        
        x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
        
        # x_eval[:,1] = (x_eval[:,1] - x_eval[:,1].mean()) / x_eval[:,1].std()
        # x_eval[:,0] = (x_eval[:,0] - x_eval[:,0].mean()) / x_eval[:,0].std()
        
        # x_eval = create_poly(k, x_eval)
        
        x_eval, y_eval = reject_outliers(x_eval, y_eval, m = m)
        y_pred = clf.predict(x_eval)
        np.savetxt(fname = save_path, X = y_pred)

        theta = np.reshape(theta,(theta.shape[1],))
        util.plot(x = x_eval, y = y_eval.reshape((y_eval.shape[0],)), theta = theta, save_path = 'Plots/1_5_Plot_ds' + ds + '_adjusted.png', correction=1.0)
        plt.title('1.5 Plot of ds' + ds + ' - Adjusted for Outliers. m = ' + str(m))
        plt.legend()
        plt.savefig('Plots/1_5_Plot_ds' + ds + 'adjusted_for_outliers.png')
        plt.show()
        
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        
        # util.plot(x = x_eval, y = y_pred.reshape((y_pred.shape[0],)), theta = theta, save_path = 'Plots/1_2_Plot_ds' + ds + '.png', correction=1.0)
        # plt.title('y_pred')
        # plt.legend()
        # plt.show()
    
        cm = confusion_matrix(y_eval, y_pred)
        accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
        accuracy = "{:.0%}".format(accuracy)
        ax = plt.axes()
        sns.heatmap(cm, annot=True, ax = ax)
        ax.set_title('1.5 - Confusion Matrix ds' + ds + ' - Adjusted for Outliers. Accuracy = ' + accuracy)
        plt.savefig('Plots/1_5_CM_ds' + ds + '_adjusted_for_outliers.png')
        plt.show()

def adjusted_by_normalizing():
    for i in range (1,2):
        k = 1
        ds = str(i)
        train_path='ds' + ds + '_train.csv'
        valid_path='ds' + ds + '_valid.csv'
        save_path='gda_pred_'+ ds + '_adjusted_by_normalizing.txt'
        
        x_train, y_train = util.load_dataset(train_path, add_intercept=False)
        
        x_train[:,0] = (x_train[:,0] - x_train[:,0].mean()) / x_train[:,0].std()
        x_train[:,1] = (x_train[:,1] - x_train[:,1].mean()) / x_train[:,1].std()
        
        # x_train = reject_outliers(x_train)
        
        # x_train = create_poly(k, x_train)
        # x_train, y_train = reject_outliers(x_train, y_train, m = 2)
        
        clf = GDA()
        theta = clf.fit(x_train, y_train.reshape((y_train.shape[0],)))
        
        x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
        
        x_eval[:,1] = (x_eval[:,1] - x_eval[:,1].mean()) / x_eval[:,1].std()
        x_eval[:,0] = (x_eval[:,0] - x_eval[:,0].mean()) / x_eval[:,0].std()
        
        # x_eval = create_poly(k, x_eval)
        
        # x_eval, y_eval = reject_outliers(x_eval, y_eval, m = 2)
        
        y_pred = clf.predict(x_eval)
        np.savetxt(fname = save_path, X = y_pred)

        theta = np.reshape(theta,(theta.shape[1],))
        util.plot(x = x_eval, y = y_eval.reshape((y_eval.shape[0],)), theta = theta, save_path = 'Plots/1_5_Plot_ds' + ds + '_adjusted.png', correction=1.0)
        plt.title('1.5 Plot of ds' + ds + ' - Adjusted by Normalizing')
        plt.legend()
        plt.savefig('Plots/1_5_Plot_ds' + ds + '_adjusted_by_normalizing.png')
        plt.show()
        
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        
        # util.plot(x = x_eval, y = y_pred.reshape((y_pred.shape[0],)), theta = theta, save_path = 'Plots/1_2_Plot_ds' + ds + '.png', correction=1.0)
        # plt.title('y_pred')
        # plt.legend()
        # plt.show()
    
        cm = confusion_matrix(y_eval, y_pred)
        accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
        accuracy = "{:.0%}".format(accuracy)
        ax = plt.axes()
        sns.heatmap(cm, annot=True, ax = ax)
        ax.set_title('1.5 - Confusion Matrix ds' + ds + ' - Adjusted by Normalizing. Accuracy = ' + accuracy)
        plt.savefig('Plots/1_5_CM_ds' + ds + '_adjusted_by_normalizing.png')
        plt.show()

def adjusted_by_logs():
    for i in range (1,2):
        ds = str(i)
        train_path='ds' + ds + '_train.csv'
        valid_path='ds' + ds + '_valid.csv'
        save_path='gda_pred_'+ ds + 'adjusted_by_logs.txt'
        
        x_train, y_train = util.load_dataset(train_path, add_intercept=False)
        
        x_train = np.log(x_train**2)
        
        clf = GDA()
        theta = clf.fit(x_train, y_train.reshape((y_train.shape[0],)))
        
        x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
        
        x_eval = np.log(x_eval**2)

        y_pred = clf.predict(x_eval)
        np.savetxt(fname = save_path, X = y_pred)

        theta = np.reshape(theta,(theta.shape[1],))
        util.plot(x = x_eval, y = y_eval.reshape((y_eval.shape[0],)), theta = theta, save_path = 'Plots/1_5_Plot_ds' + ds + '_adjusted.png', correction=1.0)
        plt.title('1.5 Plot of ds' + ds + ' - Adjusted by Logarithm')
        plt.legend()
        plt.savefig('Plots/1_5_Plot_ds' + ds + 'adjusted_by_logs.png')
        plt.show()
        
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        
        # util.plot(x = x_eval, y = y_pred.reshape((y_pred.shape[0],)), theta = theta, save_path = 'Plots/1_2_Plot_ds' + ds + '.png', correction=1.0)
        # plt.title('y_pred')
        # plt.legend()
        # plt.show()
    
        cm = confusion_matrix(y_eval, y_pred)
        accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
        accuracy = "{:.0%}".format(accuracy)
        ax = plt.axes()
        sns.heatmap(cm, annot=True, ax = ax)
        ax.set_title('1.5 - Confusion Matrix ds' + ds + ' - Adjusted by Logarithm. Accuracy = ' + accuracy)
        plt.savefig('Plots/1_5_CM_ds' + ds + 'adjusted_by_logs.png')
        plt.show()

def adjusted_by_exp():
    for i in range (1,2):
        ds = str(i)
        train_path='ds' + ds + '_train.csv'
        valid_path='ds' + ds + '_valid.csv'
        save_path='gda_pred_'+ ds + 'adjusted_by_exp.txt'
        
        x_train, y_train = util.load_dataset(train_path, add_intercept=False)
        x_train[:,0] = (x_train[:,0] - x_train[:,0].mean()) / x_train[:,0].std()
        x_train[:,1] = (x_train[:,1] - x_train[:,1].mean()) / x_train[:,1].std()
        x_train = np.exp(x_train)
        
        clf = GDA()
        theta = clf.fit(x_train, y_train.reshape((y_train.shape[0],)))
        
        x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
        x_eval[:,1] = (x_eval[:,1] - x_eval[:,1].mean()) / x_eval[:,1].std()
        x_eval[:,0] = (x_eval[:,0] - x_eval[:,0].mean()) / x_eval[:,0].std()
        x_eval = np.exp(x_eval)
        
        y_pred = clf.predict(x_eval)
        np.savetxt(fname = save_path, X = y_pred)

        theta = np.reshape(theta,(theta.shape[1],))
        util.plot(x = x_eval, y = y_eval.reshape((y_eval.shape[0],)), theta = theta, save_path = 'Plots/1_5_Plot_ds' + ds + '_adjusted.png', correction=1.0)
        plt.title('1.5 Plot of ds' + ds + ' - Adjusted by Exponential')
        plt.legend()
        plt.savefig('Plots/1_5_Plot_ds' + ds + 'adjusted_by_exp.png')
        plt.show()
        
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        
        # util.plot(x = x_eval, y = y_pred.reshape((y_pred.shape[0],)), theta = theta, save_path = 'Plots/1_2_Plot_ds' + ds + '.png', correction=1.0)
        # plt.title('y_pred')
        # plt.legend()
        # plt.show()
    
        cm = confusion_matrix(y_eval, y_pred)
        accuracy = round( (cm[0][0] + cm[1][1]) / cm.sum(), 4)
        accuracy = "{:.0%}".format(accuracy)
        ax = plt.axes()
        sns.heatmap(cm, annot=True, ax = ax)
        ax.set_title('1.5 - Confusion Matrix ds' + ds + ' - Adjusted by Exponential. Accuracy = ' + accuracy)
        plt.savefig('Plots/1_5_CM_ds' + ds + 'adjusted_by_exp.png')
        plt.show()
        
adjusted_by_normalizing()
adjusted_for_outliers(m = 1.5)
adjusted_by_logs()
# adjusted_by_exp()