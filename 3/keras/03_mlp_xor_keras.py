import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(123)

'''
데이터를 생성한다
'''
# XORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

'''
모델을 설정한다
'''
model = Sequential()

# 입력층-은닉층
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

# 은닉층-출력층
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

'''
모델을 학습시킨다
'''
model.fit(X, Y, epochs=4000, batch_size=4)

'''
학습 결과를 확인한다
'''
classes = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=4)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
