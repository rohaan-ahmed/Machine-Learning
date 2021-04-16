import numpy as np
import utils
import time

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        start_time = time.perf_counter()
        
        n, d = x.shape
        # if self.theta is None:
        self.theta = np.zeros(d, dtype=np.float32)

        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            hess = self._hessian(x)

            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)

            loss = self._loss(x, y)
            if self.verbose:
                print('[iter: {:02d}, loss: {:.7f}]'.format(i, loss))

            if np.max(np.abs(prev_theta - self.theta)) < self.eps:
                break

        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))
            print('Newton\'s Metho Execution Time: %fs' % (time.perf_counter() - start_time))
            
        return self.theta
        # *** END CODE HERE ***

    def fit_gradient(self, x, y, learning_rate=0.3):
        """Run Gradient Descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        start_time = time.perf_counter()
        
        n, d = x.shape
        # if self.theta is None:
        self.theta = np.zeros(d, dtype=np.float32)
        print('Initial theta (logreg): {}'.format(self.theta))

        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            # hess = self._hessian(x)

            prev_theta = np.copy(self.theta)
            # self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)
            
            self.theta = self.theta - learning_rate*grad

            loss = self._loss(x, y)
            if self.verbose:
                print('[iter: {:02d}, loss: {:.7f}]'.format(i, loss))

            if np.max(np.abs(prev_theta - self.theta)) < self.eps:
                break

        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))
            print('Gradient Descent Execution Time: %fs' % (time.perf_counter() - start_time))
            
        return self.theta
        # *** END CODE HERE ***
        
    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_hat = self._sigmoid(x.dot(self.theta))

        return y_hat

    def _gradient(self, x, y):
        """Get gradient of J.

        Returns:
            grad: The gradient of J with respect to theta. Same shape as theta.
        """
        n, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        grad = 1 / n * x.T.dot(probs - y)

        return grad

    def _hessian(self, x):
        """Get the Hessian of J given theta and x.

        Returns:
            hess: The Hessian of J. Shape (dim, dim), where dim is dimension of theta.
        """
        n, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        diag = np.diag(probs * (1. - probs))
        hess = 1 / n * x.T.dot(diag).dot(x)

        return hess

    def _loss(self, x, y):
        """Get the empirical loss for logistic regression."""
        eps = 1e-10
        hx = self._sigmoid(x.dot(self.theta))
        loss = -np.mean(y * np.log(hx + eps) + (1 - y) * np.log(1 - hx + eps))

        return loss

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # *** END CODE HERE ***

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def main(train_path, test_path):

    train_x, train_y = utils.load_dataset(train_path)
    test_x, test_y = utils.load_dataset(test_path)
    train_x_inter = add_intercept(train_x)
    test_x_inter = add_intercept(test_x)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_x_inter, train_y)

    pred_y_prob = classifier.predict(test_x_inter)
    test_pred_y = (pred_y_prob > 0.5).astype(int)

    utils.plot(test_x, test_y, test_pred_y)


if __name__ == '__main__':
    main('train_data.csv', 'test_data.csv')