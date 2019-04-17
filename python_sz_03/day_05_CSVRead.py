import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



def csvread(filelist):
    """
    读取CSV文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    # 1、构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2、构造csv阅读器读取队列数据（按一行）
    reader = tf.TextLineReader()

    key, value = reader.read(file_queue)

    # 3、对每行内容解码
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [["None"], ["None"]]

    example, label = tf.decode_csv(value, record_defaults=records)

    # 4、想要读取多个数据，就需要批处理
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    print(example_batch, label_batch)
    return example_batch, label_batch


if __name__ == "__main__":
    # 1、找到文件，放入列表   路径+名字  ->列表当中
    fileNameList = os.listdir("/Users/binbin/Documents/PyCharmWorkSpace/machinelearning/python_sz_03/data/csvdata")

    fileList = [os.path.join("/Users/binbin/Documents/PyCharmWorkSpace/machinelearning/python_sz_03/data/csvdata",file) for file in fileNameList]

    example_batch, label_batch = csvread(fileList)
    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        print(sess.run([example_batch,label_batch]))

        # 回收子线程
        coord.request_stop()

        coord.join(threads)


