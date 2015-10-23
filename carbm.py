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
2014/06/23
DRBMのアルゴリズム改良版
・DRBM_B の隠れ層を毎回追加
・増やすニューロンを増やすアプローチ
・最初のニューロンの数を増やす
・学習率とモーメント項の調整

・エントロピー推移からの予測モデル

2014/09/17
continuous action rbm
"""

import sys
import numpy
import copy
import random

numpy.seterr(all='ignore')



def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3,
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
        self.mom_W = W * 0.
        self.hbias = hbias
        self.vbias = vbias

        # self.params = [self.W, self.hbias, self.vbias]

    def contrastive_divergence(self, k=1, input=None, alpha=0.9, vbias_renew=True):
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
        # print dW.shape[1], self.mom_W.shape[1]
        dW += alpha * self.mom_W
        self.lr = avrW / (avrdW * 100.)
        if self.lr == float("inf") or self.lr != self.lr:
            self.lr = 10e-5
        self.W += self.lr * dW
        self.mom_W = dW
        if vbias_renew:
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
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v) * 0.98 + 0.01
        # print >> sys.stderr, "sig : ", sigmoid_activation_v
        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))
        if (cross_entropy != cross_entropy):
            cross_entropy = 0.0 #nan check
            # print sigmoid_activation_h
            # print "--------"
            # print self.W.T
            # print sigmoid_activation_v
            # print "    log :", numpy.log(sigmoid_activation_v)
            # print "1 - log :", numpy.log(1 - sigmoid_activation_v)
            # print "non!! add :", numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            #(1 - self.input) * numpy.log(1 - sigmoid_activation_v),
            #           axis=1)
        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

    def train_rbm(self, training_epochs=1000, vbias_renew=True):
        best_cost = float("inf")
        counter = 0
        # print "nhidden", self.n_hidden
        for epoch in xrange(training_epochs):
            cost = self._train_rbm(k=1, vbias_renew=vbias_renew)
            if best_cost > cost:
                best_cost = cost
                counter = 0
            elif cost * 0.8 > best_cost:
                counter += 1
            if counter > 5:
                break
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        return best_cost


    def _train_rbm(self, k=1, vbias_renew=True):
        self.contrastive_divergence(k=k, vbias_renew=vbias_renew)
        cost = self.get_reconstruction_cross_entropy()
        return cost

ADD_NUM = 5
INITIAL_NEURON = 1


class DRBM(RBM):

    def __init__(self, input=None, n_visible=2, n_hidden=3,
        W=None, hbias=None, vbias=None, numpy_rng=None):

        super(DRBM, self).__init__(input=input, n_visible=n_visible, n_hidden=n_hidden,
        W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng)

        self.pcost = 10000


        # phase of initial cost
        initial_cost = 0.
        rbm_for_test = RBM(input=input, n_visible=n_visible, n_hidden=1)
        calc_epoch = 1
        for epoch in xrange(calc_epoch):
            for j in xrange(500):
                rbm_for_test.contrastive_divergence(k=1)
            initial_cost += rbm_for_test.get_reconstruction_cross_entropy()
        initial_cost /= calc_epoch


        initial_cost = rbm_for_test.get_reconstruction_cross_entropy()
        print "initial cost", initial_cost


        # phase of bias
        starting_hidden = 10
        additional_neuron = 10
        p_ent = initial_cost
        while True:
            print "calculate:", starting_hidden, " ... ",
            ent = 0.
            rbm_for_test = RBM(input=input, n_visible=n_visible, n_hidden=starting_hidden)
            calc_epoch = 1
            for epoch in xrange(calc_epoch):
                for j in xrange(1000):
                    rbm_for_test.contrastive_divergence(k=1)
                ent += rbm_for_test.get_reconstruction_cross_entropy()
            ent /= calc_epoch
            print ent
            if (initial_cost - ent) > 1.0: break
            p_ent = ent

            starting_hidden += additional_neuron

        v_ent = (ent - p_ent)/additional_neuron
        starting_hidden += (initial_cost - ent)/v_ent
        starting_hidden = max(int(starting_hidden), 1)

        print "starting_hidden", starting_hidden

        # phase of prediction
        point_n = 5
        ents = numpy.zeros(point_n)
        self.finetune_epochs = 2000
        minimum_ent = initial_cost * 2
        # test_hid = numpy.array(range(starting_hidden, starting_hidden+11, 10/(point_n-1)))
        f = open("learning_epoch.dat", "w")
        calc_range = max(starting_hidden, 10)
        test_hid = numpy.array([starting_hidden + i*(calc_range/point_n) for i in xrange(point_n)])
        #test_hid = numpy.array(range(starting_hidden, 2*starting_hidden+1, starting_hidden/(point_n)))
        print test_hid
        calc_epoch = 1
        for epoch in xrange(calc_epoch):
            for i, hid in enumerate(test_hid):
                flag = False
                rbm_for_test = RBM(input=input, n_visible=n_visible, n_hidden=hid)
                for j in xrange(self.finetune_epochs):
                    rbm_for_test.contrastive_divergence(k=1)
                    if j % 5 == 0:
                        cost = rbm_for_test.get_reconstruction_cross_entropy()
                        if flag:
                            if cost > minimum_ent:
                                self.finetune_epochs = j

                                break
                        elif cost < minimum_ent:
                            flag = True
                        f.write(str(cost)+" ")
                print self.finetune_epochs, "times ",
                minimum_ent = cost
                ents[i] += rbm_for_test.get_reconstruction_cross_entropy()
                f.write("\n")
        ents /= calc_epoch

        print ents

        # if ents[1] - ents[0] > 1: sys.exit()

        #self.variation = -(ents[1] - ents[0])/(test_hid[1] - test_hid[0])
        self.variation = - (point_n * sum(test_hid*ents) - sum(test_hid)*sum(ents)) / \
                         (point_n * sum(test_hid*test_hid) - sum(test_hid)**2)

        print self.variation

        n_hidden = (ents[0] / self.variation) + starting_hidden
        print "n_hidden = ", n_hidden

        super(DRBM, self).__init__(input=input, n_visible=n_visible, n_hidden=n_hidden,
        W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng)

        # self.subRBM = copy.copy(self)
        # self.subRBM.add_hidden(add_n=ADD_NUM)

        # print self.subRBM.W.shape[1]
        # print self.subRBM.mom_W.shape[1]
        # print self.W.shape[1]
        # print self.mom_W.shape[1]

        # self.threshold = threshold

        print "initialize ok", self.W.shape[1], self.get_reconstruction_cross_entropy()

        self.__class__ = RBM

        # print self.input

from operator import itemgetter, attrgetter

class ObserveRBM(RBM):

    def energy_function(self, input_data):
        h = self.propup(input_data)
        return numpy.dot(numpy.dot(- h, self.W.T), input_data)

    def observe(self, dataset):
        energy_set = []
        for data in dataset:
            energy = self.energy_function(data)
            energy_set.append([energy, [d for d in data]])
        sorted_energy_set = sorted(energy_set, key=itemgetter(0))
        # for i, e_set in enumerate(sorted_energy_set):
        #     print i, e_set
        return numpy.array([data[1] for data in sorted_energy_set[int(len(dataset)*0.75):]])

    def observe_by_threshold(self, dataset, teachset, threshold):
        result_set = []
        result_teach = []
        print "threshold is ", threshold
        for data, teach in zip(dataset, teachset):
            energy = self.energy_function(data)
            if energy > threshold:
                print energy
                result_set.append(data)
                result_teach.append(teach)

        return result_set, result_teach

    def get_max_energy(self, dataset):
        max_energy = -float("inf")
        for data in dataset:
            energy = self.energy_function(data)

            if energy > max_energy:
                max_energy = energy
        return max_energy

    def synthesis(self, added_rbm):
        self.W = numpy.c_[self.W, added_rbm.W]
        self.hbias = numpy.r_[self.hbias, added_rbm.hbias]
        self.n_hidden += added_rbm.n_hidden

def get_dataset(data_n=100, prior_dataset=None, noise_rate=0.3):

    if prior_dataset is None:
        return None

    res_dataset = []

    for i in xrange(data_n):
        p_data = random.choice(prior_dataset)
        got_data = []
        for bit in p_data:
            got_data.append(bit if random.random() < noise_rate else (1 - bit))
        res_dataset.append(got_data)

    return numpy.array(res_dataset)


def rbm_script():

    #parameters
    data_n = 100
    training_epochs = 1000

    prior_dataset = numpy.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    dataset = get_dataset(data_n=data_n, prior_dataset=prior_dataset, noise_rate=0.1)
    # print dataset
    rng = numpy.random.RandomState(1234)
    rbm = DRBM(input=dataset, n_visible=len(prior_dataset[0]), n_hidden=INITIAL_NEURON, numpy_rng=rng)
    res = rbm.train_rbm(training_epochs=training_epochs)
    # print rbm.get_reconstruction_cross_entropy()

    prior_dataset2 =  numpy.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]])
    dataset2 = get_dataset(data_n=data_n, prior_dataset=prior_dataset2, noise_rate=0.1)
    outside_dataset = rbm.observe(dataset2)
    # print outside_dataset
    rbm_new = DRBM(input=outside_dataset, n_visible=len(outside_dataset[0]), n_hidden=INITIAL_NEURON,
                   numpy_rng=rng, vbias=rbm.vbias)
    rbm_new.train_rbm(training_epochs=training_epochs, vbias_renew=False)
    # print rbm_new.get_reconstruction_cross_entropy()

    rbm.synthesis(rbm_new)
    # print rbm.W.shape, rbm.n_hidden, rbm.hbias.shape, rbm.vbias.shape

    rbm.observe(dataset2)


if __name__ == "__main__":
    # test_rbm()
    rbm_script()