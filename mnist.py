import tensorflow as tf
from tensorflow.example.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

number_nodes_l1 = 500
number_nodes_l2 = 500
number_nodes_l3 = 500

number_classes = 10
batch_Size = 100

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")


def neural_network(data):
    hidden_l1 = {"weights": tf.Variable(tf.random_normal([784, number_nodes_l1])),
                "biases": tf.Variable(tf.random_normal([number_nodes_l1]))}

    hidden_l2 = {"weights": tf.Variable(tf.random_normal([number_nodes_l1, number_node_l2])),
                "biases": tf.Variable(tf.random_normal([number_nodes_l2]))}

    hidden_l3 = {"weights": tf.Variable(tf.random_normal([number_nodes_l2, number_nodes_l3])),
                "biases": tf.Variable(tf.random_normal([number_nodes_l3]))}

    output_la = {"weights": tf.Variable(tf.random_normal([number_nodes_l3, number_classes])),
                "biases": tf.Variable(tf.random_normal([number_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_l1["weights"]) + hidden_l1["biases"])
    l1 = tf.relu(l1)

    l2 = tf.add(tf.matmul(data, hidden_l2["weights"]) + hidden_l2["biases"])
    l2 = tf.relu(l2)

    l3 = tf.add(tf.matmul(data, hidden_l3["weights"]) + hidden_l3["biases"])
    l3 = tf.relu(l3)

    output = tf.add(tf.matmul(data, output_la["weights"]) + output_la["biases"])

    return output

def train_the_beast(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    optimiser = tf.train.AdamOptimizer
