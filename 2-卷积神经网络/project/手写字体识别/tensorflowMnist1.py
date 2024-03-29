# 卷积神经网络

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 读取数据
mnist = input_data.read_data_sets('/MNIST/', one_hot=True)
# mnist = tf.keras.datasets.mnist
x = tf.placeholder("float", shape=[None, 28, 28, 1]) # 一次迭代样本数、长度、宽度、颜色通道
y_ = tf.placeholder("float", shape=[None, 10])

# 卷积层1
# 权重参数初始化
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)) # 卷积核5*5*1，得到32个特征图
b_conv1 = tf.Variable(tf.constant(.1, shape=[32])) # 需要32个置偏参数
# 卷积计算流程
h_conv1 = tf.nn.conv2d(input=x, filter=W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积和池化函数
def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积层2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(.1, shape=[64]))
h_conv2 = tf.Variable(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层1
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+ b_fc1)

# 转换特征图，一个图对应一个值
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层2
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(.1, shape=[10]))
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 损失函数
crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss) # 优化器，在学习过程中让学习率逐渐减少
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accurcay = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# 迭代
sess = tf.Session()
sess.run(tf.global_variables_initializer())
batchSize = 50
for i in range(1000):
    batch = mnist.train.next_batch(batchSize)
    trainingInputs = batch[0].reshape([batchSize, 28, 28, 1])
    trainingLabels = batch[1]
    if i % 100 == 0:
        trainingAccuracy = accurcay.eval(session=sess, feed_dict={x: trainingInputs, y_: trainingLabels, keep_prob: 1.0})
        print("step %d, training accuracy %g" %(i, trainingAccuracy))
    trainStep.run(session=sess, feed_dict={x: trainingInputs, y_: trainingLabels, keep_prob: 0.5})