import time
import tensorflow as tf
import orl_input
import orl_inference
import orl_train
import numpy as np



train_data,train_lab,test_data,test_lab = orl_input.input_orl2d(5)
test_shape = np.shape(test_data)

def evaluate():
    x = tf.placeholder(tf.float32,
                       [test_shape[0],
                        orl_inference.IMAGE_HIGH,
                        orl_inference.IMAGE_WIDTH,
                        orl_inference.NUM_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.float32,[None,orl_inference.OUTPUT_NODE], name='y-input')

    y=orl_inference.inference(x, False,None)

    #计算正确率
    '''correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))'''

    #预测样本的类别
    prediction_labels = tf.argmax(y,1) + 1

    variable_averages = tf.train.ExponentialMovingAverage(orl_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    #while True:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(orl_train.MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            #accuracy_score = sess.run(accuracy, feed_dict={x:test_data, y_:test_lab})
            #print("After %s training steps, validation accuracy = %g" % (global_step,accuracy_score))

            #for i in range(test_shape[0]):
            prediction_lab = sess.run(prediction_labels,feed_dict={x:test_data, y_:test_lab})
            print(prediction_lab)
                                                 

        else:
            print("No checkpoint file found")
            return

      

def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
