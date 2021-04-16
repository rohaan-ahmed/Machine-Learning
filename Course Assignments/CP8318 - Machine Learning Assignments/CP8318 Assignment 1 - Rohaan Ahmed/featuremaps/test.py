import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
    
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
        pred = pred.reshape(n,)
        return pred
        # *** END CODE HERE ***
        
train_path='train.csv'
small_path='small.csv'
eval_path='test.csv'
        
sine=False
ks=[3]
filename="Plots/CP8318_PS1_Q1_2_ABC"

train_x, train_y = util.load_dataset(train_path, add_intercept=True)
plot_x = np.ones([1000, 2])
plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
plt.figure()
plt.scatter(train_x[:, 1], train_y)

''' USING BGD '''
for k in ks:
    '''
    Our objective is to train models and perform predictions on plot_x data
    '''
    # *** START CODE HERE ***
    model = LinearModel()  # Create model
    # Apply feature mappings
    if sine:
        train_features = model.create_sin(k, train_x)
        plot_features = model.create_sin(k, plot_x)
    else:
        train_features = model.create_poly(k, train_x)
        plot_features = model.create_poly(k, plot_x)
        
    # Fit model using Normal Equation
    # model.fit(train_features, train_y)
    
    learning_rate = 0.0002
    iterations = 10000
    X = train_features
    y = train_y
    batch_size = 1
    theta = np.random.randn(X.shape[1],1)
    
    m = len(y)
    cost_history = np.zeros(iterations)
    n_batches = int(m/batch_size)
    
    for it in range(iterations):
        cost =0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0,m,batch_size):
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            
            # X_i = np.c_[np.ones(len(X_i)),X_i]
           
            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            
            predictions = X.dot(theta)
            cost += (1/2*m) * np.sum(np.square(predictions-y))

            cost_history[it]  = cost
    
    model.theta = theta
    # Make predictions
    
    plot_y_GD = model.predict(plot_features)
    

    '''
    Here plot_y are the predictions of the linear model on the plot_x data
    '''
    plt.ylim(-2, 2)
    plt.plot(plot_x[:, 1], plot_y_GD, label='k=%d' % k)

plt.legend()
# plt.savefig(filename + '_NormalFit.png')
# plt.show()
# plt.clf()

