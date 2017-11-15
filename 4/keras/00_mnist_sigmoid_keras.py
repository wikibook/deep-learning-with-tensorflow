import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split

np.random.seed(0)

'''
데이터를 생성한다
'''
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 10000  # MNIST의 일부를 사용한다
indices = np.random.permutation(range(n))[:N]  # 무작위로 N장을 선택

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]  # 1-of-K 표현으로 변환한다

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

'''
모델을 설정한다
'''
n_in = len(X[0])  # 784
n_hidden = 200
# n_hidden = 4000
n_out = len(Y[0])  # 10

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('sigmoid'))

# model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))

# model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))

# model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

'''
모델을 학습시킨다
'''
epochs = 100
batch_size = 200

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

'''
예측 정확도를 평가한다
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
