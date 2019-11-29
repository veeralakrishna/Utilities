# quadratic weighted kappa

import numpy as np 
from numba import jit 
from keras.callbacks import Callback
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
import pickle
import warnings
warnings.filterwarnings("ignore")

# quadratic weighted kappa
@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.argmax(a1, 1) if a1.ndim > 1 else np.asarray(a1, dtype=int)
    a2 = np.argmax(a2, 1) if a2.ndim > 1 else np.asarray(a2, dtype=int)
    
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))
    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)
    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)
    e = e / a1.shape[0]
    return 1 - o / e

# TEST 
true_ = np.random.randint(4, None,100000)
true_1 = np.random.randint(4, None,100000)

t1 = np.eye(np.max(true_) + 1)[true_]
t2 = np.eye(np.max(true_1) + 1)[true_1]

print('1 - 2d')
# %timeit qwk3(t1, t2)
print(qwk3(t1, t2))

print('2 - 1d')
# %timeit qwk3(true_, true_)
print(qwk3(true_, true_1))

#  KERAS
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        tr_s = qwk3(y_train, y_pred)
        logs['tr_qwk3'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]
        y_pred = self.model.predict(X_valid)
        val_s = qwk3(y_valid, y_pred)
        logs['val_qwk3'] = val_s
        print('tr qwk3', tr_s, 'val qwk3', val_s)
        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)
            
#  Feval for Lightgbm 
def lgb_qwk3(preds, train_data):
    y_true = train_data.get_label()
    preds = preds.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'qwk', qwk3(y_true, preds), True

# For Sklearn [use for feature selection]
sklearn_qwk3 = make_scorer(qwk3, greater_is_better=True)
