import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0)

'''
모델 파일을 위한 설정
'''
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

'''
데이터를 생성한다
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

'''
모델을 설정한다
'''
model = Sequential([
    Dense(1, input_dim=2),
    Activation('sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

'''
모델을 학습시킨다
'''
model.fit(X, Y, epochs=200, batch_size=1)
model.save(MODEL_DIR + '/model.hdf5')
print('Model saved')
