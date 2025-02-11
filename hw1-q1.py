#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        #initialize true value vector
        y = np.ones(6,)*(-1)
        y[y_i] = 1
        print('y:', y)

        #calculating score z
        z = self.W @ x_i  

        #getting prediction vector y_hat by applying sign(z)
        y_hat = np.where(z >= 0, 1, -1)
        # 3. If (prediction) y_hat[j] != y[j] (true value), update w[j]
        for j in range(len(y_hat)):
                if y_hat[j] != y[j]:
                    self.W[j] = self.W[j] + (y[j] * x_i)
        
        #raise NotImplementedError # Q1.1 (a)


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # calculating score z
        z = self.W @ x_i

        # calucalting softmax 
        P = np.exp(z)/np.sum(np.exp(z))

        # correct class
        y = np.zeros(6)
        y[y_i] = 1
        
        # gradient of loss function
        loss_grad = np.outer((P - y), x_i)

        # weight updatea
        self.W = self.W - learning_rate * (loss_grad + l2_penalty * self.W)

       #raise NotImplementedError # Q1.2 (a,b)


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        self.W1 = np.append(np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features)), np.zeros((hidden_size, 1)), axis=1)
        self.W2 = np.append(np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size)), np.zeros((n_classes, 1)), axis=1)
    
    def fprop(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.

        # add bias term 
        #print("shape of x: ", X.shape)
        X_b =np.append(X, np.ones((X.shape[0], 1)), axis=1)
        if (self.W1.shape[1] != X_b.T.shape[0]): print("W: ", self.W1.shape, "\nX_b_t: ", X_b.T.shape)
        z1 = self.W1 @ X_b.T
        
        #activation 
        h = z1 * (z1 > 0)
        h = np.append(h, np.ones((1, h.shape[1])), axis=0)
        z2 = self.W2 @ h
        #normalize z2 to avoid errors with exponentials
        #z2 = z2 / np.max(z2)
        #print('z2 norm: ', z2.shape)

        return (X_b, z1, h, z2)
    
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        _, _, _, z2 = self.fprop(X)
        
        predicted_labels = np.zeros((z2.shape[1],))
        for i in range(z2.shape[1]):
            row = z2.T[i]
            softmax = np.exp(row - max(row))/np.sum(np.exp(row - max(row)))
            predicted_labels[i] = softmax.argmax()
        
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        print("y_hat & y shapes: ", y_hat.shape, " , ", y.shape)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        #Stochastic gradient loss
        #random_indices = np.random.choice(X.shape[0], size=1, replace=False)
        #random_rows = X[random_indices]
        Loss = 0

        for i in range(X.shape[0]):
            image = X[i].reshape((X.shape[1],1)).T
            #image = image
            X_b, z1, h, z2 = self.fprop(image)
            #if i < 1 : print("\nX_b shape: ", X_b.shape, "\nz1 shape: ", z1.shape, "\nh shape:", h.shape, "\nz2 shape:", z2.shape)
            
            #normalize z2 scores:
            z2 = z2 - max(z2)
            z_sum = np.sum(np.exp(z2))

            y_true = np.zeros((6, 1))
            y_true[y[i]] = 1
            
            Loss += (y_true.T @ (z2) + np.log(z_sum))[0][0]

            L_grad = (np.exp(z2)/z_sum) - y_true

            W2_grad = L_grad @ h.T
            h_grad = (self.W2.T @ L_grad)            
            
            #derivative of relu when z>0 = 1, else 0
            z1_grad = h_grad[:-1] * np.where(z1>0, 1, 0)

            W1_grad = z1_grad @ X_b

            #updates (with biases inside weight vectors)
            self.W1 = self.W1 - learning_rate * W1_grad
            self.W2 = self.W2 - learning_rate * W2_grad
 
        L_epoch = Loss/X.shape[0]
        print("denominator: ", X.shape[0] )
        print("Loss: ", L_epoch)
        return L_epoch
        #raise NotImplementedError # Q1.3 (a)


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
