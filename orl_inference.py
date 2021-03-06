import tensorflow as tf


OUTPUT_NODE = 40

IMAGE_WIDTH = 92
IMAGE_HIGH = 112
NUM_CHANNELS = 1
NUM_LABELS = 40

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512


#定义前向传播过程
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1_conv1'):
        conv1_weights = tf.get_variable(
            "weights",
            [CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer =tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biase", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.0))
        
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights, strides=[1,1,1,1], padding='SAME')
        bias = tf.nn.bias_add(conv1,conv1_biases)
        actived_conv1 = tf.nn.relu(bias)

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(actived_conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biase",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        bias= tf.nn.bias_add(conv2,conv2_biases)
        actived_conv2 = tf.nn.relu(bias)

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(actived_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope("layer5-fc1"):
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

        fc1_weights = tf.get_variable("weights",[nodes,FC_SIZE],
                                      initializer =tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:tf.add_to_collection('losses',regularizer(fc1_weights))
        fc_biases = tf.get_variable("biase",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc_biases)
        if train:fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable("weight",[FC_SIZE,NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases

    return logit
        
