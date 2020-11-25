from __future__ import print_function
import tensorflow as tf
import numpy as np
from wavelet_filters_alpha_beta_multi import wavelet_filter

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/Data/MINST/", one_hot=True)

# Parameters
training_epochs = 100
batch_size = 200
num_steps = 60000 // batch_size
display_step = 100

_SAVE_PATH = "./tensorboard/MNISTpreprocessedfullyconnect/"

# Recording the results
error_list = []
training_accuracy_list = []

# Network Parameters
n_hidden = 784  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
num_wavelet_filters = 4
mnist_test_x = mnist.test.images
mnist_test_y = mnist.test.labels

# Constants
sqrt_2 = tf.sqrt(2.)

# tf Graph input
X = tf.placeholder("float", [None, num_input])  # (128,784)
Y = tf.placeholder("float", [None, num_classes])  # (128,10)
learning_rate = tf.placeholder("float", [])

# Learning rate decay with the num of epochs
def learaning_rate_decay(epochs):
    Learning_rate = 0.001
    if epochs < 5:
        Learning_rate = Learning_rate/2
    elif 5 <= epochs < 10:
        Learning_rate = Learning_rate/5
    elif 10 <= epochs < 20:
        Learning_rate = Learning_rate/10
    elif 20 <= epochs < 40:
        Learning_rate = Learning_rate/20
    elif 40 <= epochs < 100:
        Learning_rate = Learning_rate/50
    elif 100 <= epochs:
        Learning_rate = Learning_rate/100

    return Learning_rate


def input_preprocessing(input_data):
    pre_processed_data = input_data / 255.0

    return pre_processed_data


# Store layers weight & bias
# Weights
weights = {
    'wd1': tf.Variable(tf.random_normal([4*4*64*4, 32])),
    'wd2': tf.Variable(tf.random_normal([32, 32])),
    'wo': tf.Variable(tf.random_normal([32, num_classes])),
}
# Biases
biases = {
    'bd1': tf.Variable(tf.random_normal([32])),
    'bd2': tf.Variable(tf.random_normal([32])),
    'bo': tf.Variable(tf.random_normal([num_classes]))
}

# Wavelet parameters alpha and beta
alpha_1_1 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_1_1 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_1_2 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_1_2 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_1_3 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_1_3 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_1_4 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_1_4 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))

tf.summary.scalar('alpha_1_1', alpha_1_1)
tf.summary.scalar('beta_1_1', beta_1_1)
tf.summary.scalar('alpha_1_2', alpha_1_2)
tf.summary.scalar('beta_1_2', beta_1_2)
tf.summary.scalar('alpha_1_3', alpha_1_3)
tf.summary.scalar('beta_1_3', beta_1_3)
tf.summary.scalar('alpha_1_4', alpha_1_4)
tf.summary.scalar('beta_1_4', beta_1_4)

alpha_2_1 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_2_1 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_2_2 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_2_2 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_2_3 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_2_3 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_2_4 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_2_4 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))

tf.summary.scalar('alpha_2_1', alpha_2_1)
tf.summary.scalar('beta_2_1', beta_2_1)
tf.summary.scalar('alpha_2_2', alpha_2_2)
tf.summary.scalar('beta_2_2', beta_2_2)
tf.summary.scalar('alpha_2_3', alpha_2_3)
tf.summary.scalar('beta_2_3', beta_2_3)
tf.summary.scalar('alpha_2_4', alpha_2_4)
tf.summary.scalar('beta_2_4', beta_2_4)

alpha_3_1 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_3_1 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_3_2 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_3_2 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_3_3 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_3_3 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
alpha_3_4 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))
beta_3_4 = tf.Variable(tf.truncated_normal(shape=[], mean=0.0, stddev=0.5))

tf.summary.scalar('alpha_3_1', alpha_3_1)
tf.summary.scalar('beta_3_1', beta_3_1)
tf.summary.scalar('alpha_3_2', alpha_3_2)
tf.summary.scalar('beta_3_2', beta_3_2)
tf.summary.scalar('alpha_3_3', alpha_3_3)
tf.summary.scalar('beta_3_3', beta_3_3)
tf.summary.scalar('alpha_3_4', alpha_3_4)
tf.summary.scalar('beta_3_4', beta_3_4)


def wavelet_product(wavelet, wavelet_transposed, x, wavelet_level):
    output_list = []
    in_channel = 0
    out_channels = 0
    window = 0

    if wavelet_level == '1':
        window = 28
        in_channel = 1
        out_channels = 1
    elif wavelet_level == '2':
        window = 14
        in_channel = 4
        out_channels = 4
    elif wavelet_level == '3':
        window = 7
        in_channel = 16
        out_channels = 16

    _X = tf.reshape(x, shape=[-1, window, window, in_channel])

    for i in range(batch_size):
        for j in range(in_channel):
            wavelet_transform = tf.matmul(tf.matmul(wavelet, _X[i, :, :, j]),
                                          wavelet_transposed)
            output_list.append(wavelet_transform)
    output = tf.reshape(tf.stack(output_list), shape=[-1, window, window, out_channels])

    del output_list[:]
    in_channel = 0
    out_channels = 0

    return output


def output_split(input, split_lvl):  # ok
    split_window = 0
    in_channel = 0
    out_channels = 0

    coefficients_list = []

    if split_lvl == '1':
        split_window = 28
        half_window_1 = 14
        half_window_2 = 14
        in_channel = 1
        out_channels = 4
    elif split_lvl == '2':
        split_window = 14
        half_window_1 = 7
        half_window_2 = 7
        in_channel = 4
        out_channels = 16

    elif split_lvl == '3':
        split_window = 7
        half_window_1 = 4
        half_window_2 = 3
        in_channel = 16
        out_channels = 64

    input = tf.reshape(input, shape=[-1, split_window, split_window, in_channel])

    for j in range(in_channel):
        for i in range(batch_size):
            Approximate_details = input[i, 0:half_window_1, 0:half_window_1, j]
            Horizontal_detail = input[i, 0:half_window_1, half_window_2:split_window, j]
            Vertaical_details = input[i, half_window_2:split_window, 0:half_window_1, j]
            Diagonal_details = input[i, half_window_2:split_window, half_window_2:split_window, j]
            wavelet_coefficients = tf.stack(
                [Approximate_details, Horizontal_detail, Vertaical_details, Diagonal_details], 0)
            coefficients_list.append(wavelet_coefficients)

    wavelet_coefficients_out = tf.stack(coefficients_list)
    wavelet_coefficients_out = tf.reshape(wavelet_coefficients_out,
                                          shape=[-1, half_window_1, half_window_1, out_channels])
    split_window = 0
    in_channel = 0
    out_channels = 0
    return wavelet_coefficients_out


def wavelet_transform(input_1, input_2, input_3, input_4, alpha_1, alpha_2, alpha_3, alpha_4, beta_1, beta_2, beta_3,
                      beta_4, win_size, level):
    # Filters 1
    wavelet_1 = wavelet_filter(alpha_1, beta_1, 'no', level)
    wavelet_transposed_1 = wavelet_filter(alpha_1, beta_1, 'yes', level)
    # wavelet transform
    wavelet_layer_sub_1 = wavelet_product(wavelet_1, wavelet_transposed_1, input_1, win_size)

    # Filters 2
    wavelet_2 = wavelet_filter(alpha_2, beta_2, 'no', level)
    wavelet_transposed_2 = wavelet_filter(alpha_2, beta_2, 'yes', level)
    # wavelet transform
    wavelet_layer_sub_2 = wavelet_product(wavelet_2, wavelet_transposed_2, input_2, win_size)

    # Filters 3
    wavelet_3 = wavelet_filter(alpha_3, beta_3, 'no', level)
    wavelet_transposed_3 = wavelet_filter(alpha_3, beta_3, 'yes', level)
    # wavelet transform
    wavelet_layer_sub_3 = wavelet_product(wavelet_3, wavelet_transposed_3, input_3, win_size)

    # Filters 4
    wavelet_4 = wavelet_filter(alpha_4, beta_4, 'no', level)
    wavelet_transposed_4 = wavelet_filter(alpha_4, beta_4, 'yes', level)
    # wavelet transform
    wavelet_layer_sub_4 = wavelet_product(wavelet_4, wavelet_transposed_4, input_4, win_size)

    return wavelet_layer_sub_1, wavelet_layer_sub_2, wavelet_layer_sub_3, wavelet_layer_sub_4


# Create network model -------------------------------------------------

def neural_net(x, weights, biases):
    # Wavelet transform_level 1
    wavelet_layer_1, wavelet_layer_2, wavelet_layer_3, wavelet_layer_4 = wavelet_transform(x, x, x, x, alpha_1_1,
                                                                                           alpha_1_2, alpha_1_3,
                                                                                           alpha_1_4, beta_1_1,
                                                                                           beta_1_2, beta_1_3, beta_1_4,
                                                                                           '1', '1')
    # Output window split level 2
    wavelet_layer_1_split = output_split(wavelet_layer_1, '1')
    wavelet_layer_2_split = output_split(wavelet_layer_2, '1')
    wavelet_layer_3_split = output_split(wavelet_layer_3, '1')
    wavelet_layer_4_split = output_split(wavelet_layer_4, '1')


    # Wavelet transform_level 2
    wavelet_layer_2_1, wavelet_layer_2_2, wavelet_layer_2_3, wavelet_layer_2_4 = wavelet_transform(
        wavelet_layer_1_split, wavelet_layer_2_split, wavelet_layer_3_split, wavelet_layer_4_split, alpha_2_1,
        alpha_2_2, alpha_2_3, alpha_2_4, beta_2_1, beta_2_2, beta_2_3, beta_2_4, '2', '2')
    # Output window split level 2
    wavelet_layer_2_1_split = output_split(wavelet_layer_2_1, '2')
    wavelet_layer_2_2_split = output_split(wavelet_layer_2_2, '2')
    wavelet_layer_2_3_split = output_split(wavelet_layer_2_3, '2')
    wavelet_layer_2_4_split = output_split(wavelet_layer_2_4, '2')


    # Wavelet transform_level 3
    wavelet_layer_3_1, wavelet_layer_3_2, wavelet_layer_3_3, wavelet_layer_3_4 = wavelet_transform(
        wavelet_layer_2_1_split, wavelet_layer_2_2_split, wavelet_layer_2_3_split, wavelet_layer_2_4_split, alpha_3_1,
        alpha_3_2, alpha_3_3, alpha_3_4, beta_3_1, beta_3_2, beta_3_3, beta_3_4, '3', '3')
    # Output window split level 3
    wavelet_layer_3_1_split = tf.reshape(output_split(wavelet_layer_3_1, '3'), shape=[-1, 4, 4, 64])
    wavelet_layer_3_2_split = tf.reshape(output_split(wavelet_layer_3_2, '3'), shape=[-1, 4, 4, 64])
    wavelet_layer_3_3_split = tf.reshape(output_split(wavelet_layer_3_3, '3'), shape=[-1, 4, 4, 64])
    wavelet_layer_3_4_split = tf.reshape(output_split(wavelet_layer_3_4, '3'), shape=[-1, 4, 4, 64])

    # Concatanation
    concatenat_feature_maps = tf.stack(
        [wavelet_layer_3_1_split, wavelet_layer_3_2_split, wavelet_layer_3_3_split, wavelet_layer_3_4_split], 0)

    # Flatten
    flatten_feature_maps = tf.reshape(concatenat_feature_maps, shape=[-1, weights['wd1'].get_shape().as_list()[0]])

    # fully connected layer 1
    fullyconnect_1 = tf.nn.relu(tf.matmul(flatten_feature_maps, weights['wd1']) + biases['bd1'])

    # fully connected layer 2
    fullyconnect_2 = tf.nn.relu(tf.matmul(fullyconnect_1, weights['wd2']) + biases['bd2'])

    # output/classification layer
    out = tf.matmul(fullyconnect_2, weights['wo']) + biases['bo']

    Global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    return out, Global_step


# Construct model
logits, Global_step = neural_net(X, weights, biases)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, global_step=Global_step)
tf.summary.scalar("Loss", loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100
tf.summary.scalar("Accuracy/train", accuracy)
test_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100
tf.summary.scalar("Accuracy/test", test_accuracy)

merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)

except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    print("Training the model")
    for epoch in range(training_epochs):
        for step in range(1, num_steps + 1):
            training_examples = step * batch_size
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = input_preprocessing(batch_x)
            # Run optimization op (backprop)
            i_global, Optimizer = sess.run([Global_step, optimizer], feed_dict={X: batch_x, Y: batch_y,
                                                                                learning_rate: learaning_rate_decay(
                                                                                    epoch)})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Golbal step: ", i_global, ", Epoch: " + str(epoch + 1) + ", Training examples: " + str(
                    training_examples) + ", Minibatch Loss= " + "{:.4f}".format(
                    loss) + ", Training Accuracy= " + "{:.3f}".format(acc), "%", ", Current learning rate: ",
                      learaning_rate_decay(epoch))

                summary = tf.Summary(value=[tf.Summary.Value(tag="Accuracy/test", simple_value=acc), ])
                training_accuracy_list.append(acc)
                error_list.append(loss)

        print("Testing for the current epoch...")

        epoch_accuracy_list = []
        for i in range(10):
            index = np.random.randint(len(mnist_test_x), size=batch_size)
            test_batch_x = input_preprocessing(mnist_test_x[index])
            test_batch_y = mnist_test_y[index]

            test_acc = sess.run(test_accuracy, feed_dict={X: test_batch_x, Y: test_batch_y})
            epoch_accuracy_list.append(test_acc)

        test_acc_epoch = sum(epoch_accuracy_list) / 10
        del epoch_accuracy_list[:]
        print("Test accuracy for current epoch: ", test_acc_epoch, "%")

        data_merged, global_1 = sess.run([merged, Global_step], feed_dict={X: test_batch_x, Y: test_batch_y})

        summary = tf.Summary(value=[tf.Summary.Value(tag="Accuracy/test", simple_value=test_acc_epoch), ])
        test_acc_epoch = 0
        train_writer.add_summary(data_merged, global_1)
        train_writer.add_summary(summary, global_1)

        saver.save(sess, save_path=_SAVE_PATH, global_step=Global_step)
        print("Saved checkpoint.")

    print("--- Optimization Finished ! ---")
    print("Testing !")
    # Calculate accuracy for MNIST test images

    test_accuracy_list = []
    for i in range(100):
        test_index = np.random.randint(len(mnist_test_x), size=batch_size)
        test_batch_x = input_preprocessing(mnist_test_x[test_index])
        test_batch_y = mnist_test_y[test_index]

        test_acc = sess.run(test_accuracy, feed_dict={X: test_batch_x, Y: test_batch_y})
        test_accuracy_list.append(test_acc)

    test_accuracy = sum(test_accuracy_list) / 100
    del test_accuracy_list[:]
    print("Test Accuracy: ", test_accuracy, "%")

    with open("error_list (epoch with tensorboard version preprocessed single fully connect)", "w") as error_data:
        error_data.write("Batch error per 100 batch : %s\n" % ('\n'.join(map(str, error_list))) + ",")

    with open("training_accuracy_list (epoch with tensorboard version preprocessed single fully connect)",
              "w") as accuracy_data:
        accuracy_data.write(
            "Batch accuracy per 100 batch : %s \n" % ('\n' + ', '.join(map(str, training_accuracy_list))))
