# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
import pprint

from mnist import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# 取出训练好的线性模型
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)

saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")

# 取出训练好的卷积模型
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)

saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convalutional.ckpt")


# 根据输入调用线性模型并返回识别结果
def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


# 根据输入调用卷积模型并返回识别结果
def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    # pprint.pprint(request.json)
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    pprint.pprint(output1)
    pprint.pprint(output2)
    return jsonify(results=[output1, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8889)
