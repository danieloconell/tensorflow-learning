import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

number_nodes_l1 = 500
number_nodes_l2 = 500
number_nodes_l3 = 500

number_classes = 10
batch_size = 100

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")


def neural_network(data):
    hidden_l1 = {'weights':tf.Variable(tf.random_normal([784, number_nodes_l1])),
                      'biases':tf.Variable(tf.random_normal([number_nodes_l1]))}

    hidden_l2 = {'weights':tf.Variable(tf.random_normal([number_nodes_l1, number_nodes_l2])),
                      'biases':tf.Variable(tf.random_normal([number_nodes_l2]))}

    hidden_l3 = {'weights':tf.Variable(tf.random_normal([number_nodes_l2, number_nodes_l3])),
                      'biases':tf.Variable(tf.random_normal([number_nodes_l3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([number_nodes_l3, number_classes])),
                    'biases':tf.Variable(tf.random_normal([number_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_l1['weights']), hidden_l1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_l2['weights']), hidden_l2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_l3['weights']), hidden_l3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_the_beast(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    number_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(number_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', number_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_the_beast(x)