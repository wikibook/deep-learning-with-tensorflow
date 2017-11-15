import os
import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

'''
로그 파일을 위한 설정
'''
LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')

if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)

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

x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
t = tf.placeholder(tf.float32, shape=[None, 1], name='t')
y = tf.nn.sigmoid(tf.matmul(x, w) + b, name='y')

with tf.name_scope('loss'):
    cross_entropy = \
        - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
tf.summary.scalar('cross_entropy', cross_entropy)  # 텐서보드용으로 등록한다

with tf.name_scope('train'):
    train_step = \
        tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
모델을 학습시킨다
'''
init = tf.global_variables_initializer()
sess = tf.Session()

file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)  # 텐서보드를 지원한다
summaries = tf.summary.merge_all()  # 등록한 변수를 하나로 정리한다

sess.run(init)

# 학습시킨다
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

    summary, loss = sess.run([summaries, cross_entropy], feed_dict={
        x: X,
        t: Y
    })
    file_writer.add_summary(summary, epoch)  # 텐서보드에 기록한다
