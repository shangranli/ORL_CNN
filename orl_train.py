import tensorflow as tf
import orl_inference
import orl_input
import numpy as np
import os




LEARNING_RATE_BASE = 0.05
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH="ORL_model/"
MODEL_NAME = "orl_model"

train_size=5
test_size=10-train_size
train_num=40*train_size
test_num=40*test_size

BATCH_SIZE = test_num
train_data,train_lab,test_data,test_lab = orl_input.input_orl2d(train_size)


def train():
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                        orl_inference.IMAGE_HIGH,
                        orl_inference.IMAGE_WIDTH,
                        orl_inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,[None,orl_inference.OUTPUT_NODE], name='y-input')

    #计算前向传播
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = orl_inference.inference(x,False,regularizer)

    #定义滑动平均类
    global_step = tf.Variable(0,trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    #计算交叉熵及平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    '''#设置学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               10,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)'''

    #定义训练op
    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_average_op]):
        train_op = tf.no_op(name='train')

    #定义持久化类
    saver = tf.train.Saver()


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(TRAINING_STEPS):
            start = (i*BATCH_SIZE) % train_num
            end = min(start+BATCH_SIZE,train_num)
          
            _,loss_value,step = sess.run([train_op,loss,global_step],
                                         feed_dict={x:train_data[start:end,:,:,:],y_:train_lab[start:end,:]})

            if i%10 ==0:
                print("after %d training steps,cross entropy on all data is %g" %(i,loss_value))

            if i%1000 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)


def main(argv=None):
    train()

if __name__ =='__main__':
    main()
    
