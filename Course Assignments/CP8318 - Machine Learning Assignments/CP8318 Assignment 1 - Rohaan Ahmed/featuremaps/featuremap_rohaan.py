import util
import numpy as np
import matplotlib.pyplot as plt
import sys

np.seterr(all='raise')

global theta_normal
global theta_bgd
global theta_sgd

factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.current_k = 0

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(
            np.matmul(X.T, X),
            np.matmul(X.T, y)
        )
        return
        # *** END CODE HERE ***

    def fit_GD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        m = len(y)
        batch_size = m
        
        print('\nCurrent Mapping is to k = %d' % self.current_k)
        if self.current_k == 3:
            learning_rate = 2e-5
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-10
        elif self.current_k == 5:
            learning_rate = 2e-8
            max_iterations = int(2e7/m)
            convergence_thresh = 1e-9
        elif self.current_k == 10:
            learning_rate = 2e-15
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-9
        elif self.current_k == 20:
            learning_rate = 2e-35
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-30
        else:
            learning_rate = 2e-5
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-10

        max_iterations = int(1e5)
        print('Max BGD Iterations: %d' % max_iterations)

        theta = np.random.randn(X.shape[1],)
        theta_old = theta + 1e5
        # theta = np.ones(X.shape[1],)
        
        for it in range(max_iterations+1):
        # it = 0
        # while (abs( (theta_old - theta).max() ) > convergence_thresh and it < max_iterations):
        #     it += 1
            if (it%10000 == 0):
                print('\r', 'BGD Iteration: %d' % (it), sep='', end='', flush=True)
            theta_old = theta
    
            for i in range(0,m,batch_size):
                X_i = X[i:i+batch_size]
                y_i = y[i:i+batch_size]
                hypothesis_i = np.dot(X_i,theta)
                theta = theta -(1/m)*learning_rate*( X_i.T.dot((hypothesis_i - y_i)))
        
        return it
        # *** END CODE HERE ***

    def fit_SGD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m = len(y)
        batch_size = 1
        

        print('\nCurrent Mapping is to k = %d' % self.current_k)
        if self.current_k == 3:
            learning_rate = 2e-5
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-10
        elif self.current_k == 5:
            learning_rate = 3e-6
            convergence_thresh = 1e-10
            max_iterations = int(1e6)
        elif self.current_k == 10:
            learning_rate = 2e-15
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-10
        elif self.current_k == 20:
            learning_rate = 2e-35
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-32
        else:
            learning_rate = 2e-5
            max_iterations = int(1e7/m)
            convergence_thresh = 1e-10
        
        max_iterations = int(5e5)
        print('Max SGD Iterations: %d' % max_iterations)
        
        theta = np.random.randn(X.shape[1],)
        theta_old = theta + 1e5
        # theta = np.ones(X.shape[1],)
        
        for it in range(max_iterations+1):
        # it = 0
        # while (abs( (theta_old - theta).max() ) > convergence_thresh and it < max_iterations):
        #     it += 1
            if (it%10000 == 0):
                print('\r', 'SGD Iteration: %d' % (it), sep='', end='', flush=True)
            theta_old = theta
    
            for i in range(0,m,batch_size):
                X_i = X[i:i+batch_size]
                y_i = y[i:i+batch_size]
                hypothesis_i = np.dot(X_i,theta)
                theta = theta -(1/m)*learning_rate*( X_i.T.dot((hypothesis_i - y_i)))
        
        self.theta = theta
        return it
    
        # *** END CODE HERE ***
        
    def create_poly(self, k, X) -> np.array:
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        raw = X[:, 1].reshape(-1, 1)  # (n_examples, 1)
        features = np.concatenate([raw ** i for i in range(k + 1)], axis=1)
        assert features.shape == (len(raw), k+1)
        return features
        # *** END CODE HERE ***

    def create_sin(self, k, X) -> np.array:
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        poly = self.create_poly(k, X)
        sin = np.sin(X[:, 1].reshape(-1, 1))
        features = np.concatenate([poly, sin], axis=1)
        assert features.shape == (len(X), k+2)
        return features
        # *** END CODE HERE ***

    def predict(self, X) -> np.array:
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = X.shape
        assert d == len(self.theta)
        pred = np.matmul(X, self.theta)
        assert pred.shape == (n,)
        return pred
        # *** END CODE HERE ***

def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''

        # *** START CODE HERE ***
        model = LinearModel()  # Create model
        model.current_k = k
        # Apply feature mappings
        if sine:
            train_features = model.create_sin(k, train_x)
            plot_features = model.create_sin(k, plot_x)
        else:
            train_features = model.create_poly(k, train_x)
            plot_features = model.create_poly(k, plot_x)
            
        # # Fit and Predict using Normal Equation
        model.fit(train_features, train_y)
        plot_y_normal = model.predict(plot_features)
        theta_normal = model.theta
        
        # # Fit and Predict using Batch Gradient Descent
        # BGD_iterations = model.fit_GD(train_features, train_y)
        # plot_y_GD = model.predict(plot_features)
        # theta_bgd = model.theta
        
        # Fit and Predict using Stochastic Gradient Descent
        SGD_iterations = model.fit_SGD(train_features, train_y)
        plot_y_SGD = model.predict(plot_features)
        theta_sgd = model.theta

        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        # plt.plot(plot_x[:, 1], plot_y_normal, label='k=%d %s' % (k,'Normal'))
        # plt.plot(plot_x[:, 1], plot_y_GD, label='k=%d %s Iterations: %s' % (k,'BGD', np.format_float_scientific(round(BGD_iterations, -4))))
        # plt.plot(plot_x[:, 1], plot_y_SGD, label='k=%d %s Iterations: %s' % (k,'SGD', np.format_float_scientific(round(SGD_iterations,-4))))
        
        # plt.plot(plot_x[:, 1], plot_y_normal, label='k=%d %s' % (k,'Normal'))
        # plt.plot(plot_x[:, 1], plot_y_GD, label='k=%d' % (k))
        # plt.title('Batch Gradient Descent with Small Dataset')
        plt.plot(plot_x[:, 1], plot_y_SGD, label='k=%d' % (k))
        plt.title('Stochastic Gradient Descent with Small Dataset')
    
    
    # plt.title('alfa=%s, Iterations=%d, Batch Size=%d'% (np.format_float_scientific(learning_rate),it,batch_size))
    plt.legend()
    plt.savefig(filename)
    plt.show()
    
    return model.theta
       
def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    # # (1.2 and 1.3) polynomial feature mapping
    # theta1 = run_exp(train_path, sine=False, ks=[3], filename="Plots/CP8318_PS1_Q1_3_Combined_k_3.png")
    # (1.4) polynomial feature mapping, k=[3, 5, 10, 20]
    # theta2 = run_exp(train_path, sine=False, ks=[3, 5, 10, 20], filename="Plots/CP8318_PS1_Q1_4_Combined_k_3_5_10_20.png")
    # theta2 = run_exp(train_path, sine=False, ks=[3, 5, 10, 20], filename="Plots/CP8318_PS1_Q1_2_NormalFit_k_3_5_10_20.png")
    # # (1.5) polynomial and sinusoidal feature mapping, k=[0, 1, 2, 3, 5, 10, 20]
    # theta3 = run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename="Plots/CP8318_PS1_Q1_5_Combined_sine.png")
    # theta3 = run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5], filename="Plots/CP8318_PS1_Q1_5_SGD_sine.png")
    # theta3 = run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename="Plots/CP8318_PS1_Q1_5_BGD_sine.png")
    # # (1.6) small dataset, polynomial feature mapping, k=[1, 2, 5, 10, 20]
    # theta4 = run_exp(small_path, sine=False, ks=[1, 2, 3], filename="Plots/CP8318_PS1_Q1_6_SGD_small_dataset.png")
    # return theta1, theta2, theta3, theta4
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(
        train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv'
    )
