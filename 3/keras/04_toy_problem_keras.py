import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)

'''
데이터를 생성한다
'''
N = 300
X, y = datasets.make_moons(N, noise=0.3)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

'''
모델을 생성한다
'''
model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])

'''
모델을 학습시킨다
'''
model.fit(X_train, y_train, epochs=500, batch_size=20)

'''
예측 정확도를 평가한다
'''
loss_and_metrics = model.evaluate(X_test, y_test)
print(loss_and_metrics)
