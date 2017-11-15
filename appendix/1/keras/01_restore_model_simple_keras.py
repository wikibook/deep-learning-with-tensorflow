import os
import numpy as np
from keras.models import load_model
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
저장한 모델을 읽어들이고 다시 실험한다
'''
model = load_model(MODEL_DIR + '/model.hdf5')

'''
학습이 끝난 모델로 실험한다
'''
classes = model.predict_classes(X, batch_size=1, verbose=0)
prob = model.predict_proba(X, batch_size=1, verbose=0)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
