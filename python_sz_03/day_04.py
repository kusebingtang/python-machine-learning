import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# # 实现一个加法运算
# a = tf.constant(5.0)
# b = tf.constant(6.0)
#
# sum1 = tf.add(a, b)
#
# # 默认的这张图，相当于是给程序分配一段内存
# graph = tf.get_default_graph()
#
# print(graph)
#

#
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     print(sess.run(sum1))
#     print(a.graph)
#     print("---------")
#     print(a.shape)
#     print("-------")
#     print(a.name)
#     print("-------")
#     print(a.op)



# 形状的概念
# 静态形状和动态性状
# 对于静态形状来说，一旦张量形状固定了，不能再次设置静态形状, 不能夸维度修改 1D->1D 2D->2D
# 动态形状可以去创建一个新的张量,改变时候一定要注意元素数量要匹配  1D->2D  1->3D
#
# plt = tf.placeholder(tf.float32, [None, 2])
#
# print(plt)
#
# plt.set_shape([3, 2])
#
# print(plt)
#
# # plt.set_shape([2, 3]) # 不能再次修改
#
# plt_reshape = tf.reshape(plt, [2, 3])
#
# print(plt_reshape)
#
# with tf.Session() as sess:
#     pass



#变量op

a = tf.constant(3.0, name="a")
b = tf.constant(4.0, name="b")

c = tf.add(a,b, name="c")

var = tf.Variable(tf.random_normal([2,3],mean=0.0, stddev=1.0), name="variable")

print(a,var)

# 必须显示的做一步初始化op

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)
    fileWriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

    print(sess.run([c,var]))

