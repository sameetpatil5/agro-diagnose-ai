from dotenv import load_dotenv
import tensorflow as tf

load_dotenv()

print(tf.__version__)

# Perform a simple GPU computation
with tf.device("/GPU:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:\n", c)
