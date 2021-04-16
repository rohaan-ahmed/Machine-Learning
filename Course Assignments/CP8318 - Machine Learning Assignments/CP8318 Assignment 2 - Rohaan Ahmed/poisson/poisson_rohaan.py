import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    # *** END CODE HERE ***

class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        lr=1e-5

        # Fit
        
        theta = np.matrix(np.ones(x.shape[1]))
        theta_old = np.matrix(np.ones(x.shape[1])) * 1e10
        
        iterations = 0
        
        while(np.sum(np.abs(theta - theta_old)) > 1e-5):
            theta_old = theta[:]
            
            theta_update = np.dot(theta, x.T)
            theta_update = np.exp(theta_update)
            theta_update = theta_update - y.T
            theta_update = np.dot(theta_update, x)
            
            theta = theta - lr * theta_update
            iterations += 1
            
        print("theta: \n" + str(theta))
        # print("avg difference: " + str(abs(np.mean(theta) - np.mean(theta_old))))
        print("iterations to convergence: " + str(iterations))
        
        self.theta = theta
        
        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.array(np.exp(np.dot(x,self.theta.T)))
        # *** END CODE HERE ***

# if __name__ == '__main__':
#     main(lr=1e-5,
#         train_path='train.csv',
#         eval_path='valid.csv',
#         save_path='poisson_pred.txt')

train_path='train.csv'
eval_path='valid.csv'
save_path='poisson_pred.txt'

x_train, y_train = util.load_dataset(train_path, add_intercept=True)

poisson_reg = PoissonRegression()
theta = poisson_reg.fit(x_train, y_train)

x_mean = np.array([x_train[:,0].mean(), 
          x_train[:,1].mean(), 
          x_train[:,2].mean(), 
          x_train[:,3].mean(), 
          x_train[:,4].mean()])
x_var =  np.array([x_train[:,0].var(), 
          x_train[:,1].var(), 
          x_train[:,2].var(), 
          x_train[:,3].var(), 
          x_train[:,4].var()])
# Predict
    
x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
y_eval = np.array(y_eval[:].reshape(-1,1))

h_x = poisson_reg.predict(x_eval)

plt.figure()
# here `y_eval` is the true label for the validation set and `p_eval` is the predicted label.
plt.scatter(y_eval,h_x, alpha=0.4, c='red', label='Ground Truth vs Predicted')
plt.plot([0, 25], [0, 25], color = 'blue', linewidth = 1, label='$x = y$ trendline')
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title('2.3 Poisson Regression Plot on Validation Data')
plt.legend()
plt.savefig('Plots/2_3_poisson_valid.png')