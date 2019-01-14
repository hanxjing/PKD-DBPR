# coding:utf-8
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from  datetime import *
import re
import warnings
import sys
import time
import os
import math

warnings.filterwarnings("ignore")
rule_type = ['exrule', 'exnorule', 'frequence']

ex_rule1 = ['tops', 'knee length skirts']
# ex_rule5 = ['blouses', 'knee length skirts']
# ex_rule8 = ['sweaters', 'knee length skirts']
ex_rule9 = ['sweaters', 'skirts']
ex_rule10 = ['t-shirts', 'skinny jeans']
ex_rule11 = ['t-shirts', 'jeans']
ex_rule12 = ['tank tops', 'skirts']


ex_norule1 = ['tank tops', 'dresses']
ex_norule2 = ['gray', 'maroon']


ex_rule13 = ['white', 'blue']
ex_rule14 = ['maroon', 'black blue']
# ex_rule15 = ['gray', 'gray']
ex_rule16 = ['black', 'black']

frule1 = ['category_', 'tag0cj33']
frule2 = ['category_', 'tag0c33']
frule3 = ['category_', 'tag0c33h']
frule4 = ['category_', 'tag3ce6']
frule5 = ['category_', 'tag2ce6']
frule6 = ['category_', 'tag4ce6']
frule7 = ['category_', 'tag1ce6']
frule8 = ['category_', 'tag5ce6']
frule9 = ['category_', 'tag4ce5']
frule10 = ['category_', 'tag4ce4']
frule11 = ['category_', 'tag4ce7']
frule12 = ['category_', 'tag2te7']
frule13 = ['category_', 'tag4te7']
frule14 = ['category_', 'tag4je6']
frule15 = ['category_', 'tag3je6']
frule16 = ['category_', 'tag4je6']

frule17 = ['category_', 'tag3be5']

frule20 = ['color_', 'tag433']
frule21 = ['color_', 'tag4e20']

frules = [[frule7, frule7, frule7]]
# color: tag4e5,
# tag4e30
def train(
        batch_size=64,
        n_epochs=40,
        rule_type=rule_type[2],
        rule1=None,
        rule2=None,
        rule3=None
        ):
    rule_lambda = [1.0, 0.0, 0.0]
    # parameters of text
    non_static = False
    filter_hs = [2, 3, 4, 5]
    hidden_units = [100, 2]
    conv_non_linear = "relu"
    img_w = 300

    print "loading w2v data...",
    x = cPickle.load(open("/home/hana/data_mm/data_mm_text/cloth.binary.p", "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    if non_static == True:
        print "model architecture: CNN-non-static"
        print "using: random vectors"
        U = W2
    elif non_static == False:
        print "model architecture: CNN-static"
        print "using: word2vec vectors, dim=%d" % W.shape[1]
        U = W

    # make text data
    datasets = make_idx_data(revs, word_idx_map, max_l=55, k=300, filter_h=filter_hs[-1])
    train_text_i, train_text_j, train_text_k = datasets[0], datasets[1], datasets[2]
    valid_text_i, valid_text_j, valid_text_k = datasets[3], datasets[4], datasets[5]
    test_text_i, test_text_j, test_text_k = datasets[6], datasets[7], datasets[8]

    # load visual data
    print 'loading visual data'
    print('now():' + str(datetime.now()))
    with open("/home/hana/data_mm/AUC_new_dataset_train_811_norm.pkl", "rb") as f:
        train_set = np.asarray(cPickle.load(f), dtype='float32')
    with open("/home/hana/data_mm/AUC_new_dataset_valid_811_norm.pkl", "rb") as f:
        valid_set = np.asarray(cPickle.load(f), dtype='float32')
    with open("/home/hana/data_mm/AUC_new_dataset_test_811_norm.pkl", "rb") as f:
        test_set = np.asarray(cPickle.load(f), dtype='float32')
    print 'visual data loaded'
    print('now():' + str(datetime.now()))

    print 'loading rule ind'
    with open("./rules/FashionVC/ind/train_"+rule_type+"_"+rule1[0]+rule1[1]+"_ind.pkl", "rb") as f:
        train_rule1_ind = np.asarray(cPickle.load(f), dtype='float32')
    with open("./rules/FashionVC/ind/train_"+rule_type+"_"+rule2[0]+rule2[1]+"_ind.pkl", "rb") as f:
        train_rule2_ind = np.asarray(cPickle.load(f), dtype='float32')
    with open("./rules/FashionVC/ind/train_"+rule_type+"_"+rule3[0]+rule3[1]+"_ind.pkl", "rb") as f:
        train_rule3_ind = np.asarray(cPickle.load(f), dtype='float32')

    with open("./rules/FashionVC/ind/valid_"+rule_type+"_"+rule1[0]+rule1[1]+"_ind.pkl", "rb") as f:
        valid_rule1_ind = np.asarray(cPickle.load(f), dtype='float32')
    with open("./rules/FashionVC/ind/valid_"+rule_type+"_"+rule2[0]+rule2[1]+"_ind.pkl", "rb") as f:
        valid_rule2_ind = np.asarray(cPickle.load(f), dtype='float32')
    with open("./rules/FashionVC/ind/valid_"+rule_type+"_"+rule3[0]+rule3[1]+"_ind.pkl", "rb") as f:
        valid_rule3_ind = np.asarray(cPickle.load(f), dtype='float32')

    with open("./rules/FashionVC/ind/test_"+rule_type+"_"+rule1[0]+rule1[1]+"_ind.pkl", "rb") as f:
        test_rule1_ind = np.asarray(cPickle.load(f), dtype='float32')
    with open("./rules/FashionVC/ind/test_"+rule_type+"_"+rule2[0]+rule2[1]+"_ind.pkl", "rb") as f:
        test_rule2_ind = np.asarray(cPickle.load(f), dtype='float32')
    with open("./rules/FashionVC/ind/test_"+rule_type+"_"+rule3[0]+rule3[1]+"_ind.pkl", "rb") as f:
        test_rule3_ind = np.asarray(cPickle.load(f), dtype='float32')
    print 'rule ind loaded'

    train_set_size = train_set[0].shape[0]
    valid_set_size = valid_set[0].shape[0]
    test_set_size = test_set[0].shape[0]

    train_set_i, train_set_j, train_set_k = train_set[0], train_set[1], train_set[2]
    valid_set_i, valid_set_j, valid_set_k = valid_set[0], valid_set[1], valid_set[2]
    test_set_i, test_set_j, test_set_k = test_set[0], test_set[1], test_set[2]

    train_rule1_ind = train_rule1_ind[0]
    valid_rule1_ind = valid_rule1_ind[0]
    test_rule1_ind = test_rule1_ind[0]
    train_rule2_ind = train_rule2_ind[0]
    valid_rule2_ind = valid_rule2_ind[0]
    test_rule2_ind = test_rule2_ind[0]
    train_rule3_ind = train_rule3_ind[0]
    valid_rule3_ind = valid_rule3_ind[0]
    test_rule3_ind = test_rule3_ind[0]

    np.random.seed(3435)
    # training data
    if train_set_size % batch_size > 0:
        extra_data_num = batch_size - train_set_size % batch_size
        permutation_order = np.random.permutation(train_set_size)
        train_set_i = train_set_i[permutation_order]
        train_set_j = train_set_j[permutation_order]
        train_set_k = train_set_k[permutation_order]
        train_text_i = train_text_i[permutation_order]
        train_text_j = train_text_j[permutation_order]
        train_text_k = train_text_k[permutation_order]
        train_rule1_ind = train_rule1_ind[permutation_order]
        train_rule2_ind = train_rule2_ind[permutation_order]
        train_rule3_ind = train_rule3_ind[permutation_order]

        extra_data_i = train_set_i[:extra_data_num]
        extra_data_j = train_set_j[:extra_data_num]
        extra_data_k = train_set_k[:extra_data_num]
        extra_text_i = train_text_i[:extra_data_num]
        extra_text_j = train_text_j[:extra_data_num]
        extra_text_k = train_text_k[:extra_data_num]
        extra_rule1_ind = train_rule1_ind[:extra_data_num]
        extra_rule2_ind = train_rule2_ind[:extra_data_num]
        extra_rule3_ind = train_rule3_ind[:extra_data_num]

        train_set_i = np.append(train_set_i, extra_data_i, axis=0)
        train_set_j = np.append(train_set_j, extra_data_j, axis=0)
        train_set_k = np.append(train_set_k, extra_data_k, axis=0)
        train_text_i = np.append(train_text_i, extra_text_i, axis=0)
        train_text_j = np.append(train_text_j, extra_text_j, axis=0)
        train_text_k = np.append(train_text_k, extra_text_k, axis=0)
        train_rule1_ind = np.append(train_rule1_ind, extra_rule1_ind, axis=0)
        train_rule2_ind = np.append(train_rule2_ind, extra_rule2_ind, axis=0)
        train_rule3_ind = np.append(train_rule3_ind, extra_rule3_ind, axis=0)

    train_set_size = train_set_i.shape[0]
    train_set_i = shared_dataset_x(train_set_i)
    train_set_j = shared_dataset_x(train_set_j)
    train_set_k = shared_dataset_x(train_set_k)
    train_text_i = shared_dataset_x(train_text_i)
    train_text_j = shared_dataset_x(train_text_j)
    train_text_k = shared_dataset_x(train_text_k)
    train_rule1_ind = theano.shared(np.asarray(train_rule1_ind,dtype=theano.config.floatX),borrow=True)
    train_rule2_ind = theano.shared(np.asarray(train_rule2_ind, dtype=theano.config.floatX), borrow=True)
    train_rule3_ind = theano.shared(np.asarray(train_rule3_ind, dtype=theano.config.floatX), borrow=True)

    # valid data
    if valid_set_size % batch_size > 0:
        extra_data_num = batch_size - valid_set_size % batch_size
        permutation_order = np.random.permutation(valid_set_size)
        valid_set_i = valid_set_i[permutation_order]
        valid_set_j = valid_set_j[permutation_order]
        valid_set_k = valid_set_k[permutation_order]
        valid_text_i = valid_text_i[permutation_order]
        valid_text_j = valid_text_j[permutation_order]
        valid_text_k = valid_text_k[permutation_order]
        valid_rule1_ind = valid_rule1_ind[permutation_order]
        valid_rule2_ind = valid_rule2_ind[permutation_order]
        valid_rule3_ind = valid_rule3_ind[permutation_order]

        extra_data_i = valid_set_i[:extra_data_num]
        extra_data_j = valid_set_j[:extra_data_num]
        extra_data_k = valid_set_k[:extra_data_num]
        extra_text_i = valid_text_i[:extra_data_num]
        extra_text_j = valid_text_j[:extra_data_num]
        extra_text_k = valid_text_k[:extra_data_num]
        extra_rule1_ind = valid_rule1_ind[:extra_data_num]
        extra_rule2_ind = valid_rule2_ind[:extra_data_num]
        extra_rule3_ind = valid_rule3_ind[:extra_data_num]

        valid_set_i = np.append(valid_set_i, extra_data_i, axis=0)
        valid_set_j = np.append(valid_set_j, extra_data_j, axis=0)
        valid_set_k = np.append(valid_set_k, extra_data_k, axis=0)
        valid_text_i = np.append(valid_text_i, extra_text_i, axis=0)
        valid_text_j = np.append(valid_text_j, extra_text_j, axis=0)
        valid_text_k = np.append(valid_text_k, extra_text_k, axis=0)
        valid_rule1_ind = np.append(valid_rule1_ind, extra_rule1_ind, axis=0)
        valid_rule2_ind = np.append(valid_rule2_ind, extra_rule2_ind, axis=0)
        valid_rule3_ind = np.append(valid_rule3_ind, extra_rule3_ind, axis=0)

    valid_set_size = valid_set_i.shape[0]
    valid_set_i = shared_dataset_x(valid_set_i)
    valid_set_j = shared_dataset_x(valid_set_j)
    valid_set_k = shared_dataset_x(valid_set_k)
    valid_text_i = shared_dataset_x(valid_text_i)
    valid_text_j = shared_dataset_x(valid_text_j)
    valid_text_k = shared_dataset_x(valid_text_k)
    valid_rule1_ind = theano.shared(np.asarray(valid_rule1_ind,dtype=theano.config.floatX),borrow=True)
    valid_rule2_ind = theano.shared(np.asarray(valid_rule2_ind, dtype=theano.config.floatX), borrow=True)
    valid_rule3_ind = theano.shared(np.asarray(valid_rule3_ind, dtype=theano.config.floatX), borrow=True)

    # test data
    if test_set_size % batch_size > 0:
        extra_data_num = batch_size - test_set_size % batch_size
        permutation_order = np.random.permutation(test_set_size)
        test_set_i = test_set_i[permutation_order]
        test_set_j = test_set_j[permutation_order]
        test_set_k = test_set_k[permutation_order]
        test_text_i = test_text_i[permutation_order]
        test_text_j = test_text_j[permutation_order]
        test_text_k = test_text_k[permutation_order]
        test_rule1_ind = test_rule1_ind[permutation_order]
        test_rule2_ind = test_rule2_ind[permutation_order]
        test_rule3_ind = test_rule3_ind[permutation_order]

        extra_data_i = test_set_i[:extra_data_num]
        extra_data_j = test_set_j[:extra_data_num]
        extra_data_k = test_set_k[:extra_data_num]
        extra_text_i = test_text_i[:extra_data_num]
        extra_text_j = test_text_j[:extra_data_num]
        extra_text_k = test_text_k[:extra_data_num]
        extra_rule1_ind = test_rule1_ind[:extra_data_num]
        extra_rule2_ind = test_rule2_ind[:extra_data_num]
        extra_rule3_ind = test_rule3_ind[:extra_data_num]

        test_set_i = np.append(test_set_i, extra_data_i, axis=0)
        test_set_j = np.append(test_set_j, extra_data_j, axis=0)
        test_set_k = np.append(test_set_k, extra_data_k, axis=0)
        test_text_i = np.append(test_text_i, extra_text_i, axis=0)
        test_text_j = np.append(test_text_j, extra_text_j, axis=0)
        test_text_k = np.append(test_text_k, extra_text_k, axis=0)
        test_rule1_ind = np.append(test_rule1_ind, extra_rule1_ind, axis=0)
        test_rule2_ind = np.append(test_rule2_ind, extra_rule2_ind, axis=0)
        test_rule3_ind = np.append(test_rule3_ind, extra_rule3_ind, axis=0)

    # shuffle testing data
    test_set_size = test_set_i.shape[0]
    test_set_i = shared_dataset_x(test_set_i)
    test_set_j = shared_dataset_x(test_set_j)
    test_set_k = shared_dataset_x(test_set_k)
    test_text_i = shared_dataset_x(test_text_i)
    test_text_j = shared_dataset_x(test_text_j)
    test_text_k = shared_dataset_x(test_text_k)
    test_rule1_ind = theano.shared(np.asarray(test_rule1_ind,dtype=theano.config.floatX),borrow=True)
    test_rule2_ind = theano.shared(np.asarray(test_rule2_ind, dtype=theano.config.floatX), borrow=True)
    test_rule3_ind = theano.shared(np.asarray(test_rule3_ind, dtype=theano.config.floatX), borrow=True)

    print 'train size:%f , valid size:%f , test size:%f'%(train_set_size,valid_set_size,test_set_size)
    n_train_batches = train_set_size / batch_size
    n_valid_batches = valid_set_size / batch_size
    n_test_batches = test_set_size / batch_size

    iteration = 0
    best_test_q_perf = 0.0
    ret_test_p_perf = 0.0
    ret_iteration = 0
    ret_dropout_rate = 0.0
    ret_mu_param = 0.0
    for _n_hidden in [1024]:
        for _mu_param in [0.01, 0.001, 0.0001]:
            for _learning_rate in [0.05]:
                # parameters of classifier
                n_hidden = _n_hidden
                n_in = 4096
                n_out = n_hidden
                n2_in = 400
                n2_out = n_hidden
                dropout_rate_v = 0.0
                dropout_rate_t = 0.4

                # parameters of logicnn
                pi_params = [0.95, 0]       #    [1.0, 0] is train P only
                learning_rate = _learning_rate
                momentum = 0.9
                rule_num = 3
                C = 1.0
                mu_param = _mu_param    #weight of Sqr

                index = T.lscalar()
                input1 = T.matrix('input1')
                input2 = T.matrix('input2')
                input3 = T.matrix('input3')
                input1_t = T.matrix('input1_t')
                input2_t = T.matrix('input2_t')
                input3_t = T.matrix('input3_t')

                # convolution setup
                rng = np.random.RandomState(3435)
                img_h = len(datasets[0][0])
                filter_w = img_w
                feature_maps = hidden_units[0]
                filter_shapes = []
                pool_sizes = []
                for filter_h in filter_hs:
                    filter_shapes.append((feature_maps, 1, filter_h, filter_w))
                    pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))
                parameters = [("image shape", img_h, img_w), ("filter shape", filter_shapes),
                              ("hidden_units", hidden_units),
                              ("conv_non_linear", conv_non_linear)]
                print parameters
                Words = theano.shared(value=U, name="Words")
                zero_vec_tensor = T.vector()
                zero_vec = np.zeros(img_w)
                set_zero = theano.function([zero_vec_tensor],
                                           updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))],
                                           allow_input_downcast=True)
                layer0_input_i = Words[T.cast(input1_t.flatten(), dtype="int32")].reshape(
                    (input1_t.shape[0], 1, input1_t.shape[1], Words.shape[1]))
                layer0_input_j = Words[T.cast(input2_t.flatten(), dtype="int32")].reshape(
                    (input2_t.shape[0], 1, input2_t.shape[1], Words.shape[1]))
                layer0_input_k = Words[T.cast(input3_t.flatten(), dtype="int32")].reshape(
                    (input3_t.shape[0], 1, input3_t.shape[1], Words.shape[1]))

                layer0_input = [layer0_input_i, layer0_input_j, layer0_input_k]

                # convolution
                conv_layers = []
                layer1_inputs_i = []
                layer1_inputs_j = []
                layer1_inputs_k = []
                for i in xrange(len(filter_hs)):
                    filter_shape = filter_shapes[i]
                    pool_size = pool_sizes[i]

                    conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                                    image_shape=(batch_size, 1, img_h, img_w),
                                                    filter_shape=filter_shape, poolsize=pool_size,
                                                    non_linear=conv_non_linear)
                    layer1_input_i = conv_layer.output_i.flatten(2)
                    layer1_input_j = conv_layer.output_j.flatten(2)
                    layer1_input_k = conv_layer.output_k.flatten(2)
                    conv_layers.append(conv_layer)
                    layer1_inputs_i.append(layer1_input_i)
                    layer1_inputs_j.append(layer1_input_j)
                    layer1_inputs_k.append(layer1_input_k)

                layer1_input_i = T.concatenate(layer1_inputs_i, 1)
                layer1_input_j = T.concatenate(layer1_inputs_j, 1)
                layer1_input_k = T.concatenate(layer1_inputs_k, 1)

                network = MLP(rng,
                              input1=input1,
                              input2=input2,
                              input3=input3,
                              input1_t=layer1_input_i,
                              input2_t=layer1_input_j,
                              input3_t=layer1_input_k,
                              dropout_rate_v=dropout_rate_v,
                              dropout_rate_t=dropout_rate_t,
                              n_in=n_in,
                              n_out=n_out,
                              n2_in=n2_in,
                              n2_out=n2_out
                                )

                rule1_ind = T.fmatrix('rule1_ind')
                rule2_ind = T.fmatrix('rule2_ind')
                rule3_ind = T.fmatrix('rule3_ind')

                rules = [Rule(rule1_ind), Rule(rule2_ind), Rule(rule3_ind)]
                new_pi = get_pi(cur_iter=0, params=pi_params)
                logic_nn = LogicNN(input1=input1,
                                   input2=input2,
                                   input3=input3,
                                   network=network,
                                   rules=rules,
                                   rule_num=rule_num,
                                   n_hidden=n_hidden,
                                   rule_lambda =rule_lambda,
                                   C=C,
                                   pi=new_pi,
                                   mu_param=mu_param
                                   )
                # parameters to update
                params = logic_nn.params
                for conv_layer in conv_layers:
                    params += conv_layer.params
                if non_static:
                    params += [Words]

                cost = logic_nn.cost()
                dropout_cost = logic_nn.dropout_cost()

                # momentum
                gparams = T.grad(dropout_cost, params)
                updates = []
                for p, g in zip(params, gparams):
                    mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
                    v = momentum * mparam_i - learning_rate * g
                    updates.append((mparam_i, v))
                    updates.append((p, p + v))

                train_model = theano.function([index], cost, updates=updates,
                                              givens={
                                                input1: train_set_i[index * batch_size:(index + 1) * batch_size],
                                                input2: train_set_j[index * batch_size:(index + 1) * batch_size],
                                                input3: train_set_k[index * batch_size:(index + 1) * batch_size],
                                                input1_t: train_text_i[index * batch_size:(index + 1) * batch_size],
                                                input2_t: train_text_j[index * batch_size:(index + 1) * batch_size],
                                                input3_t: train_text_k[index * batch_size:(index + 1) * batch_size],
                                                rule1_ind: train_rule1_ind[index * batch_size:(index + 1) * batch_size],
                                                rule2_ind: train_rule2_ind[index * batch_size:(index + 1) * batch_size],
                                                rule3_ind: train_rule3_ind[index * batch_size:(index + 1) * batch_size]},
                                              allow_input_downcast=True,
                                              on_unused_input='warn')

                train_test_model = theano.function([index], logic_nn.sup(),
                                                   givens={
                                                    input1: train_set_i[index * batch_size:(index + 1) * batch_size],
                                                    input2: train_set_j[index * batch_size:(index + 1) * batch_size],
                                                    input3: train_set_k[index * batch_size:(index + 1) * batch_size],
                                                    input1_t: train_text_i[index * batch_size:(index + 1) * batch_size],
                                                    input2_t: train_text_j[index * batch_size:(index + 1) * batch_size],
                                                    input3_t: train_text_k[index * batch_size:(index + 1) * batch_size],
                                                    rule1_ind: train_rule1_ind[index * batch_size:(index + 1) * batch_size],
                                                    rule2_ind: train_rule2_ind[index * batch_size:(index + 1) * batch_size],
                                                    rule3_ind: train_rule3_ind[index * batch_size:(index + 1) * batch_size]},
                                                   allow_input_downcast=True,
                                                   on_unused_input='warn')

                val_model = theano.function([index], logic_nn.sup(),
                                            givens={
                                                input1: valid_set_i[index * batch_size:(index + 1) * batch_size],
                                                input2: valid_set_j[index * batch_size:(index + 1) * batch_size],
                                                input3: valid_set_k[index * batch_size:(index + 1) * batch_size],
                                                input1_t: valid_text_i[index * batch_size:(index + 1) * batch_size],
                                                input2_t: valid_text_j[index * batch_size:(index + 1) * batch_size],
                                                input3_t: valid_text_k[index * batch_size:(index + 1) * batch_size],
                                                rule1_ind: valid_rule1_ind[index * batch_size:(index + 1) * batch_size],
                                                rule2_ind: valid_rule2_ind[index * batch_size:(index + 1) * batch_size],
                                                rule3_ind: valid_rule3_ind[index * batch_size:(index + 1) * batch_size]
                                            },
                                            allow_input_downcast=True,
                                            on_unused_input='warn')

                test_model = theano.function([index], logic_nn.sup(),
                                             givens={
                                                 input1: test_set_i[index * batch_size:(index + 1) * batch_size],
                                                 input2: test_set_j[index * batch_size:(index + 1) * batch_size],
                                                 input3: test_set_k[index * batch_size:(index + 1) * batch_size],
                                                 input1_t: test_text_i[index * batch_size:(index + 1) * batch_size],
                                                 input2_t: test_text_j[index * batch_size:(index + 1) * batch_size],
                                                 input3_t: test_text_k[index * batch_size:(index + 1) * batch_size],
                                                 rule1_ind: test_rule1_ind[index * batch_size:(index + 1) * batch_size],
                                                 rule2_ind: test_rule2_ind[index * batch_size:(index + 1) * batch_size],
                                                 rule3_ind: test_rule3_ind[index * batch_size:(index + 1) * batch_size]
                                             },
                                             allow_input_downcast=True,
                                             on_unused_input='warn')
                test_mijk = theano.function([index], logic_nn.mijk(),
                                             givens={
                                                 input1: test_set_i[index * batch_size:(index + 1) * batch_size],
                                                 input2: test_set_j[index * batch_size:(index + 1) * batch_size],
                                                 input3: test_set_k[index * batch_size:(index + 1) * batch_size],
                                                 input1_t: test_text_i[index * batch_size:(index + 1) * batch_size],
                                                 input2_t: test_text_j[index * batch_size:(index + 1) * batch_size],
                                                 input3_t: test_text_k[index * batch_size:(index + 1) * batch_size],
                                                 rule1_ind: test_rule1_ind[index * batch_size:(index + 1) * batch_size],
                                                 rule2_ind: test_rule2_ind[index * batch_size:(index + 1) * batch_size],
                                                 rule3_ind: test_rule3_ind[index * batch_size:(index + 1) * batch_size]
                                             },
                                             allow_input_downcast=True,
                                             on_unused_input='warn')

                print 'training...'
                fi = open('./rules/FashionVC/results/'+rule_type+"_"+rule1[0]+rule1[1]+'comb3_f.txt', 'a+')
                epoch = 0
                batch = 0
                iteration += 1
                best_test_q_iter = 0.0
                print 'iteration: %i' % iteration
                fi.write('###iteration: %f\n' % iteration)
                fi.flush()
                while (epoch < n_epochs):
                    start_time = time.time()
                    epoch = epoch + 1
                    if epoch > 5:
                        learning_rate = 0.02
                    cost = 0.0
                    L_sup = 0.0
                    L_p_q = 0.0
                    L_sqr = 0.0
                    p_m12 = []
                    p_m13 = []
                    q_m12 = []
                    q_m13 = []
                    # train
                    for minibatch_index in xrange(n_train_batches):
                        batch = batch + 1
                        new_pi = get_pi(cur_iter=batch * 1. / n_train_batches, params=pi_params)
                        logic_nn.set_pi(new_pi)
                        set_zero(zero_vec)
                        cost_batch = train_model(minibatch_index)
                        cost += cost_batch[0]
                        L_sup += cost_batch[1]
                        L_p_q += cost_batch[2]
                        L_sqr += cost_batch[3]
                        p_m12.append(cost_batch[4])
                        p_m13.append(cost_batch[5])
                        q_m12.append(cost_batch[6])
                        q_m13.append(cost_batch[7])
                    print 'epoch: %i, cost: %.4f, L_sup: %.4f, L_p_q: %.4f, L_sqr: %.4f' % (
                    epoch, cost, L_sup, L_p_q, L_sqr)
                    # training result
                    train_sup = [train_test_model(i) for i in xrange(n_train_batches)]
                    train_sup = np.array(train_sup)
                    train_q_sup = train_sup[:, 0]
                    train_p_sup = train_sup[:, 1]
                    count_q = 0.0
                    count_p = 0.0
                    for i in range(train_q_sup.shape[0]):
                        for j in range(train_q_sup.shape[1]):
                            if train_q_sup[i, j, 0] > 0.5:
                                count_q += 1
                            if train_p_sup[i, j, 0] > 0.5:
                                count_p += 1
                    train_q_perf = count_q / (train_q_sup.shape[0] * train_q_sup.shape[1])
                    train_p_perf = count_p / (train_p_sup.shape[0] * train_p_sup.shape[1])
                    print('training time: %.2f secs; q_train perf: %.4f %% ,p_train perf: %.4f %% ' % \
                          (time.time() - start_time, train_q_perf * 100., train_p_perf * 100.))

                    # valid result
                    valid_sup = [val_model(i) for i in xrange(n_valid_batches)]
                    valid_sup = np.array(valid_sup)
                    valid_q_sup = valid_sup[:, 0]
                    valid_p_sup = valid_sup[:, 1]
                    count_q = 0.0
                    count_p = 0.0
                    for i in range(valid_q_sup.shape[0]):
                        for j in range(valid_q_sup.shape[1]):
                            if valid_q_sup[i, j, 0] > 0.5:
                                count_q += 1
                            if valid_p_sup[i, j, 0] > 0.5:
                                count_p += 1
                    val_q_perf = count_q / (valid_q_sup.shape[0] * valid_q_sup.shape[1])
                    val_p_perf = count_p / (valid_p_sup.shape[0] * valid_p_sup.shape[1])

                    # testing result
                    test_sup = [test_model(i) for i in xrange(n_test_batches)]
                    test_sup = np.array(test_sup)
                    test_q_sup = test_sup[:, 0]
                    test_p_sup = test_sup[:, 1]
                    count_q = 0.0
                    count_p = 0.0
                    for i in range(test_q_sup.shape[0]):
                        for j in range(test_q_sup.shape[1]):
                            if test_q_sup[i, j, 0] > 0.5:
                                count_q += 1
                            if test_p_sup[i, j, 0] > 0.5:
                                count_p += 1
                    test_q_perf = count_q / (test_q_sup.shape[0] * test_q_sup.shape[1])
                    test_p_perf = count_p / (test_p_sup.shape[0] * test_p_sup.shape[1])

                    print 'valid perf: q %.4f %%, p %.4f %%' % (val_q_perf * 100., val_p_perf * 100.)
                    print 'test perf: q %.4f %%, p %.4f %%' % (test_q_perf * 100., test_p_perf * 100.)
                    fi.write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                        cost, L_sup, L_p_q, L_sqr, train_q_perf * 100., train_p_perf * 100., val_q_perf * 100.,
                        val_p_perf * 100., test_q_perf * 100., test_p_perf * 100.))
                    fi.flush()
                    if test_q_perf > best_test_q_iter:
                        best_test_q_iter = test_q_perf
                        

                        iter_valid_q_perf = val_q_perf
                        iter_valid_p_perf = val_p_perf
                        iter_test_p_perf = test_p_perf
                    if test_q_perf > best_test_q_perf:
                        best_test_q_perf = test_q_perf
                        ret_test_p_perf = test_p_perf
                        ret_iteration = iteration
                        ret_mu_param = mu_param
                        best_w1 = network.W1.get_value()
                        best_w2 = network.W2.get_value()
                        best_w1t = network.W1t.get_value()
                        best_w2t = network.W2t.get_value()
                        best_b1 = network.b1.get_value()
                        best_b2 = network.b2.get_value()
                        count = 0
                        for conv_layer in conv_layers:
                            if count == 0:
                                wc0 = conv_layer.W.get_value()
                                bc0 = conv_layer.b.get_value()
                            if count == 1:
                                wc1 = conv_layer.W.get_value()
                                bc1 = conv_layer.b.get_value()
                            if count == 2:
                                wc2 = conv_layer.W.get_value()
                                bc2 = conv_layer.b.get_value()
                            if count == 3:
                                wc3 = conv_layer.W.get_value()
                                bc3 = conv_layer.b.get_value()
                            count += 1

                print '###interation: %i: valid q: %.4f%%, p: %.4f%%; test q: %.4f%%, p: %.4f%%' % (
                    iteration, iter_valid_q_perf * 100, iter_valid_p_perf * 100., best_test_q_iter * 100.,iter_test_p_perf * 100.)
                fi.write('###interation: %i: valid q: %.4f%%, p: %.4f%%; test q: %.4f%%, p: %.4f%%\n' % (
                    iteration, iter_valid_q_perf * 100, iter_valid_p_perf * 100., best_test_q_iter * 100.,iter_test_p_perf * 100.))
                fi.flush()
                fi.close()
    np.savetxt('./param2/W1.csv', best_w1)
    np.savetxt('./param2/W2.csv', best_w2)
    np.savetxt('./param2/W1t.csv', best_w1t)
    np.savetxt('./param2/W2t.csv', best_w2t)
    np.savetxt('./param2/b1.csv', best_b1)
    np.savetxt('./param2/b2.csv', best_b2)
    cPickle.dump(wc0, open("./param2/Wc0.pkl", "wb"))
    cPickle.dump(wc1, open("./param2/Wc1.pkl", "wb"))
    cPickle.dump(wc2, open("./param2/Wc2.pkl", "wb"))
    cPickle.dump(wc3, open("./param2/Wc3.pkl", "wb"))
    cPickle.dump(bc0, open("./param2/bc0.pkl", "wb"))
    cPickle.dump(bc1, open("./param2/bc1.pkl", "wb"))
    cPickle.dump(bc2, open("./param2/bc2.pkl", "wb"))
    cPickle.dump(bc3, open("./param2/bc3.pkl", "wb"))

    print 'best q perf: %.4f%%, p perf: %.4f%%' % (best_test_q_perf * 100., ret_test_p_perf * 100.)
    print 'in iteration: %i, ret_mu: %.4f' % (
    ret_iteration, ret_mu_param)


def get_pi(cur_iter, params=None, pi=None):
    k, lb = params[0], params[1]
    pi = 1. - max([k ** cur_iter, lb])
    return pi

def shared_dataset_x(data_x, borrow=True):
    shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),
                            borrow=borrow)
    return shared_x

def shared_dataset_y(data_y, borrow=True):
    shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),
                            borrow=borrow)
    return T.cast(shared_y, 'int32')

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def get_idx_from_sent(sent, word_idx_map, max_l=56, k=300, filter_h=5):
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l=40, k=300, filter_h=5):
    train_i_idx, train_j_idx, train_k_idx = [], [], []
    valid_i_idx, valid_j_idx, valid_k_idx = [], [], []
    test_i_idx, test_j_idx, test_k_idx = [], [], []

    print 'loading text data'
    print('now():' + str(datetime.now()))
    train_i, train_j, train_k = cPickle.load(open("/home/hana/data_mm/data_mm_text/AUC_new_dataset_train_811.pkl", "rb"))

    valid_i, valid_j, valid_k = cPickle.load(open("/home/hana/data_mm/data_mm_text/AUC_new_dataset_valid_811.pkl", "rb"))

    test_i, test_j, test_k = cPickle.load(open("/home/hana/data_mm/data_mm_text/AUC_new_dataset_test_811.pkl", "rb"))

    for i in range(len(train_i)):
        train_sent_i_idx = get_idx_from_sent(train_i[i], word_idx_map, max_l, k, filter_h)
        train_sent_j_idx = get_idx_from_sent(train_j[i], word_idx_map, max_l, k, filter_h)
        train_sent_k_idx = get_idx_from_sent(train_k[i], word_idx_map, max_l, k, filter_h)
        train_i_idx.append(train_sent_i_idx)
        train_j_idx.append(train_sent_j_idx)
        train_k_idx.append(train_sent_k_idx)

    for i in range(len(valid_i)):
        valid_sent_i_idx = get_idx_from_sent(valid_i[i], word_idx_map, max_l, k, filter_h)
        valid_sent_j_idx = get_idx_from_sent(valid_j[i], word_idx_map, max_l, k, filter_h)
        valid_sent_k_idx = get_idx_from_sent(valid_k[i], word_idx_map, max_l, k, filter_h)
        valid_i_idx.append(valid_sent_i_idx)
        valid_j_idx.append(valid_sent_j_idx)
        valid_k_idx.append(valid_sent_k_idx)

    for i in range(len(test_i)):
        test_sent_i_idx = get_idx_from_sent(test_i[i], word_idx_map, max_l, k, filter_h)
        test_sent_j_idx = get_idx_from_sent(test_j[i], word_idx_map, max_l, k, filter_h)
        test_sent_k_idx = get_idx_from_sent(test_k[i], word_idx_map, max_l, k, filter_h)
        test_i_idx.append(test_sent_i_idx)
        test_j_idx.append(test_sent_j_idx)
        test_k_idx.append(test_sent_k_idx)

    train_i_idx = np.array(train_i_idx, dtype="int")
    train_j_idx = np.array(train_j_idx, dtype="int")
    train_k_idx = np.array(train_k_idx, dtype="int")
    valid_i_idx = np.array(valid_i_idx, dtype="int")
    valid_j_idx = np.array(valid_j_idx, dtype="int")
    valid_k_idx = np.array(valid_k_idx, dtype="int")
    test_i_idx = np.array(test_i_idx, dtype="int")
    test_j_idx = np.array(test_j_idx, dtype="int")
    test_k_idx = np.array(test_k_idx, dtype="int")

    return [train_i_idx, train_j_idx, train_k_idx, valid_i_idx, valid_j_idx, valid_k_idx, test_i_idx, test_j_idx,
            test_k_idx]

if __name__=="__main__":
    execfile("matching_classes.py")

    for rule in frules:
        train(rule1=rule[0], rule2=rule[1], rule3=rule[2])
