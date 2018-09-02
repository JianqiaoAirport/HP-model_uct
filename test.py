import numpy as np
import tensorflow as tf
import time
import datetime
import random
import copy

di = {(0, 0): 1, (1, 0): -1, (2, 0): -1}

# print(list(di.keys())[list(di.values()).index(-1)])

for d in di:
    print(di[d])

print(di.values())

for v in di.values():
    print(v)

a = (1, 2)

b = (4, 6)

print(a+b)

print(di[tuple(np.array([1, 0]))])

a = tf.placeholder(tf.float32, shape=[None, 20, 15], name='a')
b = tf.reshape(a, [-1, 15])
c = tf.constant(0.1, shape=[15, 30])
d = tf.matmul(b, c)
e = tf.reshape(d, [-1, 20, 30])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    input_a = np.zeros((128, 20, 15))
    input_a[0, 5, 2] = 1
    input_a[0, 5, 3] = 2
    input_a[0, 5, 4] = 3
    input_a[1, 1, 9] = 9
    x = sess.run(b, feed_dict={a: input_a})
    y = sess.run(d, feed_dict={a: input_a})
    z = sess.run(e, feed_dict={a: input_a})
    print(x)

a = [2.1, 2, 3, 4]
b = np.array(a)

print(b % 1)

print(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

noise = np.random.dirichlet(0.25*np.ones(6))
print(noise)


ones = np.ones((5, 1))
print(np.sum(ones == 1))


global ax_s

ax_s = 2

def tes():
    global ax_s
    ax_s += 1

tes()

print(ax_s)

class A:
    def __init__(self, a):
        self.a = a

a = A(a=1)

b = a
b.a = 3

print(a.a)

max_num = max([("a", 3), ("b", 2), ("c", 3)], key=lambda v: v[1])

print(max_num)

l1 = ['a', 'b', 'c', 'd']
p = [0.1, 0.2, 0.5, 0.2]






def weighted_sample(population, weights, k):
    """
    This function draws a random sample of length k
    from the sequence 'population' according to the
    list of weights
    """
    sample = set()
    population = list(population)
    weights = list(weights)
    while len(sample) < k:
        choice = np.random.choice(population, p=weights)
        sample.add(choice)
        index = population.index(choice)
        weights.pop(index)
        population.remove(choice)
        weights = [x / sum(weights) for x in weights]
    return list(sample)


batch = weighted_sample(l1, p, 2)
pass
