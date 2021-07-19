from base import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    cancer = load_breast_cancer()
    data = cancer['data']
    target = cancer['target']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    model = Model(30, 64, 2)
    train_loss, test_acc = model.train_and_test(X_train, y_train, X_test, y_test)
    x = list(range(1, 1+model.episodes))
    plot('train_loss', 'episodes', 'loss', x, train_loss)
    plot('test_acc', 'episodes', 'acc', x, test_acc)