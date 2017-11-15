import os
import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

'''
모델 파일을 위한 설정
'''
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

'''
데이터를 생성한다
'''
# OR 게이트
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

'''
모델을 설정한다
'''
w = tf.Variable(tf.zeros([2, 1]), name='w')
b = tf.Variable(tf.zeros([1]), name='b')

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
모델을 학습시킨다
'''
init = tf.global_variables_initializer()
saver = tf.train.Saver()  # 모델 저장용
sess = tf.Session()
sess.run(init)

# 학습시킨다
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

# 모델을 저장한다
model_path = saver.save(sess, MODEL_DIR + '/model.ckpt')
print('Model saved to:', model_path)
