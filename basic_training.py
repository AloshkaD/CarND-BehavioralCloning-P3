import tensorflow as tf
from model import train_network

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
flags.DEFINE_integer('validation_split_percentage', 0.05, "The batch size.")
flags.DEFINE_boolean('use_weights', False, "Whether to use prior trained weights.")
flags.DEFINE_float('lr', 0.001, "Optimizer learning rate.")
flags.DEFINE_float('dropout_prob', 0.1, "Percentage of neurons to misfire during training.")
flags.DEFINE_string('activation', 'relu', "The activation function used by the network.")

def main(_):
    train_network(
        output_shape=(40, 80, 3),
        nb_epoch=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        validation_split_percentage=FLAGS.validation_split_percentage,
        learning_rate=FLAGS.lr,
        dropout_prob=FLAGS.dropout_prob,
        activation=FLAGS.activation,
        use_weighs=FLAGS.use_weights
    )

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()