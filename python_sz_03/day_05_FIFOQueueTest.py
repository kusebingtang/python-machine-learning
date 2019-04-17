import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 模拟异步子线程 存入样本， 主线程 读取样本

# 1、定义一个队列，1000
fifoQueue = tf.FIFOQueue(1000,tf.float32)

# 2、定义要做的事情 循环 值，+1， 放入队列当中
var = tf.Variable(0.0)

# 实现一个自增  tf.assign_add
data = tf.assign_add(var,tf.constant(1.0))
en_q = fifoQueue.enqueue(data)

# 3、定义队列管理器op, 指定多少个子线程，子线程该干什么事情
queueRunner = tf.train.QueueRunner(fifoQueue, enqueue_ops=[en_q]*2)


# 初始化变量的OP
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # 开启线程管理器
    coord = tf.train.Coordinator()

    # 真正开启子线程
    threads = queueRunner.create_threads(sess,coord=coord,start=True)


    # 主线程，不断读取数据训练
    for i in  range(300):
        print(sess.run(fifoQueue.dequeue()))

    # 回收你
    coord.request_stop()
    coord.join(threads)

