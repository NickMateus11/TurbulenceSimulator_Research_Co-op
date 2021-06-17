'''
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
'''

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import datetime
import sys

# requires GPU to be accessible
if len(tf.config.list_physical_devices('GPU')) == 0:
    sys.exit(1)

# Num of multiplications to perform
n = 10
# size of matrix (NxN)
N = 1000

'''
Example: compute A^n + B^n on 2 GPUs
Sample results on 8 cores with 2 GTX-980:
 * Single GPU computation time: 0:00:11.277449
 * Multi GPU computation time: 0:00:07.131701
'''
# Create random large matrix
A = np.random.rand(N, N).astype('float32')
B = np.random.rand(N, N).astype('float32')

# Create a graph to store results
c1 = []
c2 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

'''
Single GPU computing
'''
with tf.device('/gpu:0'):
    a = tf.compat.v1.placeholder(tf.float32, [N, N])
    b = tf.compat.v1.placeholder(tf.float32, [N, N])
    # Compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(a, n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tf.compat.v1.Session() as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_1 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1-t1_1))