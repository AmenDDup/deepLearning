# 神经网络


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取数据
mnist = input_data.read_data_sets('/MNIST/', one_hot=True)

# 设置参数
numClasses = 10 # 十分类任务
inputSize = 784 # 输入参数大小 28*28*1
numHiddenUnits = 64 # 隐藏神经元个数
trainingIterations = 10000 # 迭代次数
batchSize = 64
X = tf.placeholder(tf.float32, shape=[None, inputSize]) # 输入
y = tf.placeholder(tf.float32, shape=[None, numClasses]) # 输出

# 参数初始化，权重W1,W2，偏置参数B1,B2
W1 = tf.Variable(tf.truncated_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits, numClasses], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [numClasses])

# 从前到后计算
hiddenLayerOutput = tf.matmul(X, W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput) # 激活函数relu
finalOutput = tf.matmul(hiddenLayerOutput, W2) + B2
# 对数损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=finalOutput))
# 优化器计算梯度进行更新
opt = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(loss)
# 准确率计算
correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# 迭代优化
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    _, trainingLoss = sess.run([opt, loss], feed_dict={X: batchInput, y: batchLabels})
    if i % 100 == 0:
        trainAccuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print("step %d, training accuracy %g" % (i, trainAccuracy))

