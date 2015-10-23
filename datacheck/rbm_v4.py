#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 Restricted Boltzmann Machine (RBM)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials


"""

"""
Hinton lr
"""

import sys
import numpy
import copy
import os

numpy.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, numpy_rng=None):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)


        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0


        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.mom_W = W * 0.0
        self.hbias = hbias
        self.vbias = vbias

        self.lr = 0.1

        # self.params = [self.W, self.hbias, self.vbias]


    def contrastive_divergence(self, lr=0.1, lr_div=100., k=1, input=None, alpha=0.9):
        if input is not None:
            self.input = input

        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples

        dW = (numpy.dot(self.input.T, ph_sample)
                        - numpy.dot(nv_samples.T, nh_means))
        avrW = numpy.mean(numpy.abs(self.W))
        avrdW = numpy.mean(numpy.abs(dW))
        dW += alpha * self.mom_W
        self.lr = avrW / (avrdW * lr_div)
        self.W += self.lr * dW
        self.mom_W = dW
        self.vbias += self.lr * numpy.mean(self.input - nv_samples, axis=0)
        self.hbias += self.lr * numpy.mean(ph_sample - nh_means, axis=0)

        # cost = self.get_reconstruction_cross_entropy()
        # return cost

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)

        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)

        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]


    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v) * 0.998 + 0.001

        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

    def train_rbm(self, lr=0.1, k=1, training_epochs=1000, lr_div=None):

        # train
        for epoch in xrange(training_epochs):
            self.contrastive_divergence(lr=lr, k=k, lr_div=lr_div)
            # cost = self.get_reconstruction_cross_entropy()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost

        # print self.reconstruct(v)


def test_rbm(learning_rate=0.1, k=1, data_n=6, data_size=6, n_hidden=8,
             data=None, lr_div=None, fds=[]):

    res = []
    dataset = []
    rng = numpy.random.RandomState(123)

    for line in open("dataset.dat", "r"):
        data = [int(n) for n in line.split()]
        dataset.append(data)

    dataset = numpy.array(dataset)

    rbm = RBM(input=dataset, n_visible=len(dataset[0]), n_hidden=n_hidden, numpy_rng=rng)
    trained_epochs = 0
    fds = [10, 50, 100, 500, 1000, 2000]
    for fd in fds:
        rbm.train_rbm(lr=learning_rate, k=k, training_epochs=(fd-trained_epochs), lr_div=5000)
        # text = "%d %\nf" % (n_hidden, rbm.get_reconstruction_cross_entropy())
        # fd[1].write(text)
        # print text
        res.append(rbm.get_reconstruction_cross_entropy())
        trained_epochs = fd

    return numpy.array(res)

if __name__ == "__main__":

    file = open("result.dat", "w")
    for i in xrange(50):
        print i, "th"
        file.write(str(i)+" ")
        for res in test_rbm(n_hidden=i):
            file.write(str(res)+" ")
        file.write("\n")
    #rbm_script()
