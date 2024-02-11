import tensorflow as tf
import math
import numpy as np
import data_init
import datetime
import os


class Config(object):
    lr = 0.001
    decay_rate = 0.9
    decay_steps = 5000

    no_tea = True      # True: stu only; False: train with tea
    kd = True           # False: tea only; True: tea and stu
    tea_trained = False  # True: tea pre-trained; False: train tea

    paper_feat_length = 289
    auth_feat_length = 37
    topic_feat_length = 130
    lp_length = 64

    q_length = 64   

    a_count = 6
    a_p_count = 60
    p_tile = 10
    coa_count = 300
    coa_tile = 50
    ref_count = 25
    ref_a_count = 150
    t_count = 12
    t_p_count = 120
    t_p_a_count = 720
    t_a_tile = 60

    epochs = 40
    buffer_size = 100
    batch_size = 10

    years = 3

    train_path = ['tfrecord/data_0.tfrecord',
                  'tfrecord/data_1.tfrecord',
                  'tfrecord/data_2.tfrecord',
                  'tfrecord/data_3.tfrecord',
                  'tfrecord/data_4.tfrecord',
                  'tfrecord/data_5.tfrecord',
                  'tfrecord/data_6.tfrecord',
                  'tfrecord/data_7.tfrecord']

    test_path = ['tfrecord/data_8.tfrecord',
                 'tfrecord/data_9.tfrecord']

    fold = 1
    # train_path = ['tfrecord/data_0.tfrecord']
    # test_path = ['tfrecord/data_8.tfrecord']

    T = 0.9


def get_init(nin, nout):
    scale = math.sqrt(6.0 / (nin + nout))
    init = tf.random_uniform_initializer(-scale, scale)
    return init


def batch_attention(keys, query, cnt, weight, tile_size):
    query_len = query.shape[-1]
    querys = tf.tile(query, [1, tile_size])
    querys = tf.reshape(querys, [-1, tile_size, query_len])
    keys_query = tf.concat([keys, querys], 2)

    weight_len = weight.shape[0]
    keys_query = tf.reshape(keys_query, [-1, weight_len])
    res = tf.matmul(keys_query, weight)
    res = tf.reshape(res, [-1, tile_size, 1])
    res = tf.reshape(res, [-1, tile_size])
    res = tf.nn.softmax(tf.nn.tanh(res))
    res = tf.reshape(res, [-1, tile_size, 1])
    res_keys = tf.multiply(keys, res)
    res_keys = tf.reduce_sum(res_keys, 1) / cnt
    res_keys = tf.nn.tanh(res_keys)
    return res_keys


def batch_pooling(keys, query, cnt, weight, tile_size):

    keys = tf.reduce_sum(keys, 1) / cnt
    keys = tf.nn.tanh(keys)
    return keys


def batch_multi_node_attention(keys, query, cnt, weight, tile_size, node_count):
    query_len = query.shape[-1]
    querys = tf.tile(query, [1, 1, tile_size])
    querys = tf.reshape(querys, [-1, node_count * tile_size, query_len])
    keys_query = tf.concat([keys, querys], 2)

    weight_len = weight.shape[0]
    keys_query = tf.reshape(keys_query, [-1, weight_len])
    res = tf.matmul(keys_query, weight)
    res = tf.reshape(res, [-1, node_count * tile_size, 1])
    res = tf.reshape(res, [-1, node_count, tile_size])
    res = tf.nn.softmax(tf.nn.tanh(res))
    res = tf.reshape(res, [-1, node_count * tile_size, 1])
    res_keys = tf.multiply(keys, res)
    res_keys = tf.reshape(res_keys, [-1, node_count, tile_size, keys.shape[-1]])

    res_keys = tf.reduce_sum(res_keys, 2)

    rk_len = res_keys.shape[-1]
    res_keys = tf.reshape(res_keys, [-1, rk_len])
    cnt = tf.reshape(cnt, [-1, 1])
    res_keys = res_keys / cnt

    res_keys = tf.reshape(res_keys, [-1, node_count, rk_len])

    res_keys = tf.nn.tanh(res_keys)
    return res_keys


def batch_multi_node_pooling(keys, query, cnt, weight, tile_size, node_count):
    k_len = keys.shape[-1]
    keys = tf.reshape(keys, [-1, node_count, tile_size, k_len])
    keys = tf.reduce_sum(keys, 2)
    keys = tf.reshape(keys, [-1, k_len])
    cnt = tf.reshape(cnt, [-1, 1])
    keys = keys / cnt

    keys = tf.reshape(keys, [-1, node_count, k_len])

    keys = tf.nn.tanh(keys)
    return keys


def batch_sem_att(keys, w, b, q, tile):
    keys_len = keys.shape[-1]
    keys = tf.reshape(keys, [-1, keys_len])
    res = tf.matmul(keys, w) + b
    res = tf.matmul(res, q)
    res = tf.reshape(res, [-1, tile])
    res = tf.nn.softmax(tf.nn.tanh(res))
    res = tf.reshape(res, [-1, tile, 1])

    keys = tf.reshape(keys, [-1, tile, keys_len])
    res_keys = tf.multiply(keys, res)
    res_keys = tf.reduce_sum(res_keys, 1)
    return res_keys


def batch_multi_node_sem_att(keys, w, b, q, tile, node_count):
    keys_len = keys.shape[-1]
    keys = tf.reshape(keys, [-1, keys_len])
    res = tf.matmul(keys, w) + b
    res = tf.matmul(res, q)
    res = tf.reshape(res, [-1, node_count, tile])
    res = tf.nn.softmax(tf.nn.tanh(res))
    res = tf.reshape(res, [-1, node_count, tile, 1])

    keys = tf.reshape(keys, [-1, node_count, tile, keys_len])
    res_keys = tf.multiply(keys, res)
    res_keys = tf.reduce_sum(res_keys, 2)
    return res_keys


def acc(x, y):
    ex = tf.exp(x) - 1
    ey = tf.exp(y) - 1
    l_bound = tf.greater(ex - 0.5 * ey, 0.0)
    h_bound = tf.greater(0.0, ex - 1.5 * ey)
    bound = tf.logical_and(l_bound, h_bound)
    return tf.reduce_sum(tf.cast(bound, dtype=tf.float32), 0)


def main():
    config = Config()

    tf.set_random_seed(19940127)

    print('building model...')
    with tf.Graph().as_default():
        paper_feat = tf.placeholder(tf.float32, [None, config.paper_feat_length], 'paper_feat')

        auth_feats = tf.placeholder(tf.float32, [None, config.a_count, config.auth_feat_length], 'auth_feats')
        topic_feats = tf.placeholder(tf.float32, [None, config.t_count, config.topic_feat_length], 'topic_feats')

        ref_feats = tf.placeholder(tf.float32, [None, config.ref_count, config.paper_feat_length], 'ref_feats')
        ref_auth_feats = tf.placeholder(tf.float32, [None, config.ref_a_count, config.auth_feat_length],
                                        'ref_auth_feats')

        auth_paper_feats = tf.placeholder(tf.float32, [None, config.a_p_count, config.paper_feat_length],
                                          'auth_paper_feats')
        coauth_feats = tf.placeholder(tf.float32, [None, config.coa_count, config.auth_feat_length], 'coauth_feats')

        topic_paper_feats = tf.placeholder(tf.float32, [None, config.t_p_count, config.paper_feat_length],
                                           'topic_paper_feats')
        topic_paper_auth_feats = tf.placeholder(tf.float32, [None, config.t_p_a_count, config.auth_feat_length],
                                                'topic_paper_auth_feats')
        n_citation = tf.placeholder(tf.float32, [None], 'n_citation')

        auth_cnt = tf.placeholder(tf.float32, [None], 'auth_cnt')
        topic_cnt = tf.placeholder(tf.float32, [None], 'topic_cnt')
        ref_cnt = tf.placeholder(tf.float32, [None], 'ref_cnt')
        ref_auth_cnt = tf.placeholder(tf.float32, [None, 25], 'ref_auth_cnt')
        auth_paper_cnt = tf.placeholder(tf.float32, [None, 6], 'auth_paper_cnt')
        coauth_cnt = tf.placeholder(tf.float32, [None, 60], 'coauth_cnt')
        topic_paper_cnt = tf.placeholder(tf.float32, [None, 12], 'topic_paper_cnt')
        topic_paper_auth_cnt = tf.placeholder(tf.float32, [None, 12], 'topic_paper_auth_cnt')

        paper_auth_cnt = tf.reshape(auth_cnt, [-1, 1]) + 1
        paper_topic_cnt = tf.reshape(topic_cnt, [-1, 1]) + 1
        paper_ref_cnt = tf.reshape(ref_cnt, [-1, 1]) + 1
        paper_ref_auth_cnt = tf.reduce_sum(ref_auth_cnt, 1)
        paper_ref_auth_cnt = tf.reshape(paper_ref_auth_cnt, [-1, 1]) + 1
        paper_auth_paper_cnt = tf.reduce_sum(auth_paper_cnt, 1)
        paper_auth_paper_cnt = tf.reshape(paper_auth_paper_cnt, [-1, 1]) + 1

        auth_coauth_cnt = tf.reshape(coauth_cnt, [-1, 6, 10])
        auth_coauth_cnt = tf.reduce_sum(auth_coauth_cnt, 2) + 1
        auth_p_cnt = tf.reshape(auth_paper_cnt, [-1, 6, 1]) + 1

        topic_p_cnt = tf.reshape(topic_paper_cnt, [-1, 12, 1]) + 1
        topic_p_a_cnt = tf.reshape(topic_paper_auth_cnt, [-1, 12, 1]) + 1

        log_citation = tf.log(n_citation + 1)

        with tf.name_scope('liner_projection'):
            w_lp_p = tf.get_variable('w_lp_p', [config.paper_feat_length, config.lp_length], tf.float32,
                                     get_init(config.paper_feat_length, config.lp_length))
            w_lp_a = tf.get_variable('w_lp_a', [config.auth_feat_length, config.lp_length], tf.float32,
                                     get_init(config.auth_feat_length, config.lp_length))
            w_lp_t = tf.get_variable('w_lp_t', [config.topic_feat_length, config.lp_length], tf.float32,
                                     get_init(config.topic_feat_length, config.lp_length))

            paper_feat_lp = tf.matmul(paper_feat, w_lp_p)

            ref_feats_reshape = ref_feats
            ref_feats_reshape = tf.reshape(ref_feats_reshape, [-1, config.paper_feat_length])
            ref_feats_lp = tf.matmul(ref_feats_reshape, w_lp_p)
            ref_feats_lp = tf.reshape(ref_feats_lp, [-1, config.ref_count, config.lp_length])

            auth_paper_feats_reshape = auth_paper_feats
            auth_paper_feats_reshape = tf.reshape(auth_paper_feats_reshape, [-1, config.paper_feat_length])
            auth_paper_feats_lp = tf.matmul(auth_paper_feats_reshape, w_lp_p)
            auth_paper_feats_lp = tf.reshape(auth_paper_feats_lp, [-1, config.a_p_count, config.lp_length])

            topic_paper_feats_reshape = topic_paper_feats
            topic_paper_feats_reshape = tf.reshape(topic_paper_feats_reshape, [-1, config.paper_feat_length])
            topic_paper_feats_lp = tf.matmul(topic_paper_feats_reshape, w_lp_p)
            topic_paper_feats_lp = tf.reshape(topic_paper_feats_lp, [-1, config.t_p_count, config.lp_length])

            auth_feats_reshape = auth_feats
            auth_feats_reshape = tf.reshape(auth_feats_reshape, [-1, config.auth_feat_length])
            auth_feats_lp = tf.matmul(auth_feats_reshape, w_lp_a)
            auth_feats_lp = tf.reshape(auth_feats_lp, [-1, config.a_count, config.lp_length])

            ref_auth_feats_reshape = ref_auth_feats
            ref_auth_feats_reshape = tf.reshape(ref_auth_feats_reshape, [-1, config.auth_feat_length])
            ref_auth_feats_lp = tf.matmul(ref_auth_feats_reshape, w_lp_a)
            ref_auth_feats_lp = tf.reshape(ref_auth_feats_lp, [-1, config.ref_a_count, config.lp_length])

            coauth_feats_reshape = coauth_feats
            coauth_feats_reshape = tf.reshape(coauth_feats_reshape, [-1, config.auth_feat_length])
            coauth_feats_lp = tf.matmul(coauth_feats_reshape, w_lp_a)
            coauth_feats_lp = tf.reshape(coauth_feats_lp, [-1, config.coa_count, config.lp_length])

            topic_paper_auth_feats_reshape = topic_paper_auth_feats
            topic_paper_auth_feats_reshape = tf.reshape(topic_paper_auth_feats_reshape, [-1, config.auth_feat_length])
            topic_paper_auth_feats_lp = tf.matmul(topic_paper_auth_feats_reshape, w_lp_a)
            topic_paper_auth_feats_lp = tf.reshape(topic_paper_auth_feats_lp,
                                                   [-1, config.t_p_a_count, config.lp_length])

            topic_feats_reshape = topic_feats
            topic_feats_reshape = tf.reshape(topic_feats_reshape, [-1, config.topic_feat_length])
            topic_feats_lp = tf.matmul(topic_feats_reshape, w_lp_t)
            topic_feats_lp = tf.reshape(topic_feats_lp, [-1, config.t_count, config.lp_length])

        with tf.name_scope('neighbor_attention'):
            w_att_p_a = tf.get_variable('w_att_p_a', [config.lp_length * 2, 1], tf.float32,
                                        get_init(config.lp_length * 2, 1))
            w_att_p_t = tf.get_variable('w_att_p_t', [config.lp_length * 2, 1], tf.float32,
                                        get_init(config.lp_length * 2, 1))
            w_att_p_ref = tf.get_variable('w_att_p_ref', [config.lp_length * 2, 1], tf.float32,
                                          get_init(config.lp_length * 2, 1))
            w_att_p_ref_a = tf.get_variable('w_att_p_ref_a', [config.lp_length * 2, 1], tf.float32,
                                            get_init(config.lp_length * 2, 1))
            w_att_p_a_p = tf.get_variable('w_att_p_a_p', [config.lp_length * 2, 1], tf.float32,
                                          get_init(config.lp_length * 2, 1))

            paper_auth_agg = batch_pooling(auth_feats_lp, paper_feat_lp, paper_auth_cnt, w_att_p_a, config.a_count)
            paper_topic_agg = batch_pooling(topic_feats_lp, paper_feat_lp, paper_topic_cnt, w_att_p_t, config.t_count)
            paper_ref_agg = batch_pooling(ref_feats_lp, paper_feat_lp, paper_ref_cnt, w_att_p_ref, config.ref_count)
            paper_ref_auth_agg = batch_pooling(ref_auth_feats_lp, paper_feat_lp, paper_ref_auth_cnt, w_att_p_ref_a,
                                                 config.ref_a_count)
            paper_auth_paper_agg = batch_pooling(auth_paper_feats_lp, paper_feat_lp, paper_auth_paper_cnt,
                                                   w_att_p_a_p, config.a_p_count)

            paper_agg = tf.concat([paper_feat_lp, paper_auth_agg, paper_topic_agg, paper_ref_agg,
                                   paper_ref_auth_agg, paper_auth_paper_agg], -1)
            paper_agg = tf.reshape(paper_agg, [-1, 6, config.lp_length])
            paper_agg = tf.reshape(paper_agg, [-1, 6, config.lp_length])

            w_att_a_p = tf.get_variable('w_att_a_p', [config.lp_length * 2, 1], tf.float32,
                                        get_init(config.lp_length * 2, 1))
            w_att_a_coa = tf.get_variable('w_att_a_coa', [config.lp_length * 2, 1], tf.float32,
                                          get_init(config.lp_length * 2, 1))

            auth_paper_agg = batch_multi_node_pooling(auth_paper_feats_lp, auth_feats_lp, auth_p_cnt, w_att_a_p,
                                                        config.p_tile, config.a_count)
            auth_coa_agg = batch_multi_node_pooling(coauth_feats_lp, auth_feats_lp, auth_coauth_cnt, w_att_a_coa,
                                                      config.coa_tile, config.a_count)

            auth_agg = tf.concat([auth_feats_lp, auth_paper_agg, auth_coa_agg], -1)
            auth_agg = tf.reshape(auth_agg, [-1, config.a_count, 3, config.lp_length])

            w_att_t_p = tf.get_variable('w_att_t_p', [config.lp_length * 2, 1], tf.float32,
                                        get_init(config.lp_length * 2, 1))
            w_att_t_p_a = tf.get_variable('w_att_t_p_a', [config.lp_length * 2, 1], tf.float32,
                                          get_init(config.lp_length * 2, 1))

            topic_paper_agg = batch_multi_node_pooling(topic_paper_feats_lp, topic_feats_lp, topic_p_cnt, w_att_t_p,
                                                         config.p_tile, config.t_count)
            topic_auth_agg = batch_multi_node_pooling(topic_paper_auth_feats_lp, topic_feats_lp, topic_p_a_cnt,
                                                        w_att_t_p_a,
                                                        config.t_a_tile, config.t_count)
            topic_agg = tf.concat([topic_feats_lp, topic_paper_agg, topic_auth_agg], -1)
            topic_agg = tf.reshape(topic_agg, [-1, config.t_count, 3, config.lp_length])

        with tf.name_scope('semantic_attention'):
            w_att_sem = tf.get_variable('w_att_sem', [config.lp_length, config.q_length], tf.float32,
                                        get_init(config.lp_length, config.q_length))
            b_att_sem = tf.get_variable('b_att_sem', [config.q_length], tf.float32)
            q_att_sem = tf.get_variable('q_att_sem', [config.q_length, 1], tf.float32, get_init(config.q_length, 1))

            paper_embed = batch_sem_att(paper_agg, w_att_sem, b_att_sem, q_att_sem, 6)  # [bs,256]
            auths_embed = batch_multi_node_sem_att(auth_agg, w_att_sem, b_att_sem, q_att_sem, 3,
                                                   config.a_count)  # [bs,6,256]
            topics_embed = batch_multi_node_sem_att(topic_agg, w_att_sem, b_att_sem, q_att_sem, 3,
                                                    config.t_count)  # [bs,12,256]

        with tf.name_scope('predict_attention'):
            topics_embed_mean = tf.reduce_sum(topics_embed, 1) / paper_topic_cnt  # [bs,256]

            w_att_a_p_pre = tf.get_variable('w_att_a_p_pre', [config.lp_length * 2, 1], tf.float32,
                                            get_init(config.lp_length * 2, 1))

            auth_paper_pre = batch_attention(auths_embed, paper_embed, paper_auth_cnt, w_att_a_p_pre, config.a_count)

            w_att_a_t_pre = tf.get_variable('w_att_a_t_pre', [config.lp_length * 2, 1], tf.float32,
                                            get_init(config.lp_length * 2, 1))

            auth_topic_pre = batch_attention(auths_embed, topics_embed_mean, paper_auth_cnt, w_att_a_t_pre,
                                             config.a_count)

        with tf.name_scope('teacher_prediction'):
            tea_pre_in = tf.concat([paper_embed, topics_embed_mean, auth_paper_pre, auth_topic_pre], -1)

            w_tea_1 = tf.get_variable('w_tea_1', [4 * config.lp_length, config.lp_length], tf.float32,
                                      get_init(4 * config.lp_length, config.lp_length))
            b_tea_1 = tf.get_variable('b_tea_1', [config.lp_length], tf.float32)
            w_tea_2 = tf.get_variable('w_tea_2', [config.lp_length, config.lp_length], tf.float32,
                                      get_init(config.lp_length, config.lp_length))
            b_tea_2 = tf.get_variable('b_tea_2', [config.lp_length], tf.float32)
            w_tea_3 = tf.get_variable('w_tea_3', [config.lp_length, config.lp_length / 2], tf.float32,
                                      get_init(config.lp_length, config.lp_length / 2))
            b_tea_3 = tf.get_variable('b_tea_3', [config.lp_length / 2], tf.float32)

            w_tea_pre = tf.get_variable('w_tea_pre', [config.lp_length / 2, 1], tf.float32,
                                        get_init(config.lp_length / 2, 1))
            b_tea_pre = tf.get_variable('b_tea_pre', [1], tf.float32,
                                        initializer=tf.constant_initializer(-1))

            tea_h1 = tf.nn.tanh(tf.matmul(tea_pre_in, w_tea_1) + b_tea_1)
            tea_h2 = tf.nn.tanh(tf.matmul(tea_h1, w_tea_2) + b_tea_2)
            tea_h3 = tf.nn.tanh(tf.matmul(tea_h2, w_tea_3) + b_tea_3)

            tea_prediction = tf.matmul(tea_h3, w_tea_pre) + b_tea_pre

        with tf.name_scope('student_prediction'):
            stu_pre_in = tf.concat([topics_embed_mean, auth_topic_pre], -1)

            w_stu_1 = tf.get_variable('w_stu_1', [2 * config.lp_length, config.lp_length], tf.float32,
                                      get_init(2 * config.lp_length, config.lp_length))
            b_stu_1 = tf.get_variable('b_stu_1', [config.lp_length], tf.float32)
            w_stu_2 = tf.get_variable('w_stu_2', [config.lp_length, config.lp_length], tf.float32,
                                      get_init(config.lp_length, config.lp_length))
            b_stu_2 = tf.get_variable('b_stu_2', [config.lp_length], tf.float32)
            w_stu_3 = tf.get_variable('w_stu_3', [config.lp_length, config.lp_length / 2], tf.float32,
                                      get_init(config.lp_length, config.lp_length / 2))
            b_stu_3 = tf.get_variable('b_stu_3', [config.lp_length / 2], tf.float32)

            w_stu_pre = tf.get_variable('w_stu_pre', [config.lp_length / 2, 1], tf.float32,
                                        get_init(config.lp_length / 2, 1))

            stu_h1 = tf.nn.tanh(tf.matmul(stu_pre_in, w_stu_1) + b_stu_1)
            stu_h2 = tf.nn.tanh(tf.matmul(stu_h1, w_stu_2) + b_stu_2)
            stu_h3 = tf.nn.tanh(tf.matmul(stu_h2, w_stu_3) + b_stu_3)

            stu_prediction = tf.matmul(stu_h3, w_stu_pre)

        with tf.name_scope('loss_tea'):
            tea_prediction = tf.reshape(tea_prediction, [-1])
            loss_tea = tf.multiply((log_citation - tea_prediction), (log_citation - tea_prediction))
            loss_tea = tf.reduce_sum(loss_tea, 0)

            acc_tea = acc(tea_prediction, log_citation)

        with tf.name_scope('train_tea'):
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(config.lr, global_step, config.decay_steps, config.decay_rate,
                                            staircase=True)
            train_step_tea = tf.train.AdamOptimizer(lr).minimize(loss_tea, global_step=global_step)

        with tf.name_scope('loss_stu'):
            stu_prediction = tf.reshape(stu_prediction, [-1])
            loss_stu_gt = tf.multiply((log_citation - stu_prediction), (log_citation - stu_prediction))
            loss_stu_tea = tf.multiply((tea_prediction - stu_prediction), (tea_prediction - stu_prediction))
            loss_stu = config.T * loss_stu_gt + (1 - config.T) * loss_stu_tea
            loss_stu = tf.reduce_sum(loss_stu, 0)

            msle_stu = tf.reduce_sum(loss_stu_gt, 0)
            acc_stu = acc(stu_prediction, log_citation)

        with tf.name_scope('train_stu'):
            global_step_stu = tf.Variable(0, trainable=False)
            lr_stu = tf.train.exponential_decay(config.lr, global_step_stu, config.decay_steps, config.decay_rate,
                                                staircase=True)
            var_list_stu = [w_stu_1, b_stu_1, w_stu_2, b_stu_2, w_stu_3, b_stu_3, w_stu_pre]
            train_step_stu = tf.train.AdamOptimizer(lr_stu).minimize(loss_stu, global_step=global_step_stu,
                                                                     var_list=var_list_stu)

            train_step_stu_no_tea = tf.train.AdamOptimizer(lr_stu).minimize(loss_stu_gt, global_step=global_step_stu)

        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        ws = tf.trainable_variables()
        for w in ws:
            tf.summary.histogram(w.name, w)

        with tf.Session(config=config_gpu) as sess:

            sess.run(tf.global_variables_initializer())

            print('training...')

            train_log = open('year'+str(config.years)+'/basic_train_log_noneigh_fold'+str(config.fold)+'.txt', 'w+')
            train_log.write(datetime.datetime.now().strftime('%F %T') + '\n')

            train_log_stu = open('year'+str(config.years)+'/basic_train_log_noneigh_stu_'+str(config.T)+'.txt', 'w+')
            train_log_stu.write(datetime.datetime.now().strftime('%F %T') + '\n')

            trainset = data_init.read(config.train_path, 1, config.buffer_size, config.batch_size, config.years)
            testset = data_init.read(config.test_path, 1, config.buffer_size, config.batch_size, config.years)
            if not config.no_tea:
                # teacher training
                saver = tf.train.Saver(max_to_keep=1)

                for i in range(config.epochs):

                    train_batch_count = 0
                    train_iter = trainset.make_one_shot_iterator()
                    train_values = train_iter.get_next()
                    try:
                        while True:
                            train_batch_values = sess.run(train_values)

                            train_feed_dict = {paper_feat: train_batch_values[0], auth_feats: train_batch_values[1],
                                               topic_feats: train_batch_values[2], ref_feats: train_batch_values[3],
                                               ref_auth_feats: train_batch_values[4],
                                               auth_paper_feats: train_batch_values[5],
                                               coauth_feats: train_batch_values[6],
                                               topic_paper_feats: train_batch_values[7],
                                               topic_paper_auth_feats: train_batch_values[8],
                                               n_citation: train_batch_values[9],
                                               auth_cnt: train_batch_values[10], topic_cnt: train_batch_values[11],
                                               ref_cnt: train_batch_values[12],
                                               ref_auth_cnt: train_batch_values[13], auth_paper_cnt: train_batch_values[14],
                                               coauth_cnt: train_batch_values[15],
                                               topic_paper_cnt: train_batch_values[16],
                                               topic_paper_auth_cnt: train_batch_values[17]}

                            sess.run(train_step_tea, feed_dict=train_feed_dict)

                            train_batch_count += 1
                            if train_batch_count % 100 == 0:
                                print(train_batch_count)
                                # loss = sess.run(loss_tea, feed_dict=train_feed_dict)
                                # batch_acc = sess.run(acc_tea, feed_dict=train_feed_dict)
                                # train_log.write(
                                #     'epoch: ' + str(i + 1) + ' batch: ' + str(train_batch_count) + ' loss: ' + str(
                                #         loss) + ' acc: ' + str(batch_acc) + '\n')

                            if train_batch_count % 1000 == 0:
                                log_cite, tea_pre = sess.run([log_citation, tea_prediction], feed_dict=train_feed_dict)
                                # train_log.write('gt: ' + str(log_cite) +'\n')
                                # train_log.write('pre: ' + str(tea_pre) +'\n')


                    except tf.errors.OutOfRangeError:
                        print('epoch ' + str(i + 1) + ' end.\n')

                    #saver.save(sess, 'year'+str(config.years)+'/ckpt/Basic_Model_noneigh_fold'+str(config.fold)+'.ckpt')

                    test_iter = testset.make_one_shot_iterator()
                    test_values = test_iter.get_next()
                    test_counts = 0
                    test_loss = 0
                    test_acc = 0
                    try:
                        while True:
                            test_batch_values = sess.run(test_values)

                            test_feed_dict = {paper_feat: test_batch_values[0], auth_feats: test_batch_values[1],
                                              topic_feats: test_batch_values[2], ref_feats: test_batch_values[3],
                                              ref_auth_feats: test_batch_values[4],
                                              auth_paper_feats: test_batch_values[5],
                                              coauth_feats: test_batch_values[6],
                                              topic_paper_feats: test_batch_values[7],
                                              topic_paper_auth_feats: test_batch_values[8],
                                              n_citation: test_batch_values[9],
                                              auth_cnt: test_batch_values[10], topic_cnt: test_batch_values[11],
                                              ref_cnt: test_batch_values[12],
                                              ref_auth_cnt: test_batch_values[13], auth_paper_cnt: test_batch_values[14],
                                              coauth_cnt: test_batch_values[15],
                                              topic_paper_cnt: test_batch_values[16],
                                              topic_paper_auth_cnt: test_batch_values[17]}

                            test_counts += test_batch_values[0].shape[0]
                            test_loss += sess.run(loss_tea, feed_dict=test_feed_dict)
                            test_acc += sess.run(acc_tea, feed_dict=test_feed_dict)

                    except tf.errors.OutOfRangeError:
                        test_loss /= test_counts
                        test_acc /= test_counts
                        train_log.write('epoch: ' + str(i + 1) + ' testing loss: ' + str(test_loss) + ' testing acc: ' + str(test_acc) + '\n')

                print('tea train end')
                train_log.write(datetime.datetime.now().strftime('%F %T') + '\n')
                train_log.close()

                # student training
                if config.kd:
                    for i in range(config.epochs):

                        train_batch_count = 0
                        train_iter = trainset.make_one_shot_iterator()
                        train_values = train_iter.get_next()
                        try:
                            while True:
                                train_batch_values = sess.run(train_values)

                                train_feed_dict = {paper_feat: train_batch_values[0], auth_feats: train_batch_values[1],
                                                   topic_feats: train_batch_values[2], ref_feats: train_batch_values[3],
                                                   ref_auth_feats: train_batch_values[4],
                                                   auth_paper_feats: train_batch_values[5],
                                                   coauth_feats: train_batch_values[6],
                                                   topic_paper_feats: train_batch_values[7],
                                                   topic_paper_auth_feats: train_batch_values[8],
                                                   n_citation: train_batch_values[9],
                                                   auth_cnt: train_batch_values[10], topic_cnt: train_batch_values[11],
                                                   ref_cnt: train_batch_values[12],
                                                   ref_auth_cnt: train_batch_values[13],
                                                   auth_paper_cnt: train_batch_values[14],
                                                   coauth_cnt: train_batch_values[15],
                                                   topic_paper_cnt: train_batch_values[16],
                                                   topic_paper_auth_cnt: train_batch_values[17]}

                                sess.run(train_step_stu, feed_dict=train_feed_dict)

                                train_batch_count += 1
                                if train_batch_count % 100 == 0:
                                    print(train_batch_count)
                                    loss_gt = sess.run(msle_stu, feed_dict=train_feed_dict)
                                    batch_acc_stu = sess.run(acc_stu, feed_dict=train_feed_dict)
                                    train_log_stu.write(
                                        'stu epoch : ' + str(i + 1) + ' batch: ' + str(train_batch_count) + ' loss: ' + str(
                                            loss_gt) + ' acc: ' + str(batch_acc_stu) + '\n')

                        except tf.errors.OutOfRangeError:
                            print('stu epoch ' + str(i + 1) + ' end.\n')

                        test_iter = testset.make_one_shot_iterator()
                        test_values = test_iter.get_next()
                        test_counts = 0
                        test_loss_gt = 0
                        test_acc_stu = 0
                        try:
                            while True:
                                test_batch_values = sess.run(test_values)

                                test_feed_dict = {paper_feat: test_batch_values[0], auth_feats: test_batch_values[1],
                                                  topic_feats: test_batch_values[2], ref_feats: test_batch_values[3],
                                                  ref_auth_feats: test_batch_values[4],
                                                  auth_paper_feats: test_batch_values[5],
                                                  coauth_feats: test_batch_values[6],
                                                  topic_paper_feats: test_batch_values[7],
                                                  topic_paper_auth_feats: test_batch_values[8],
                                                  n_citation: test_batch_values[9],
                                                  auth_cnt: test_batch_values[10], topic_cnt: test_batch_values[11],
                                                  ref_cnt: test_batch_values[12],
                                                  ref_auth_cnt: test_batch_values[13],
                                                  auth_paper_cnt: test_batch_values[14],
                                                  coauth_cnt: test_batch_values[15],
                                                  topic_paper_cnt: test_batch_values[16],
                                                  topic_paper_auth_cnt: test_batch_values[17]}

                                test_counts += test_batch_values[0].shape[0]
                                test_loss_gt += sess.run(msle_stu, feed_dict=test_feed_dict)
                                test_acc_stu += sess.run(acc_stu, feed_dict=test_feed_dict)

                        except tf.errors.OutOfRangeError:
                            test_loss_gt /= test_counts
                            test_acc_stu /= test_counts
                            train_log_stu.write('stu epoch : ' + str(i + 1) + ' testing loss: ' + str(test_loss_gt) + ' testing acc: ' + str(test_acc_stu) + '\n')

                    print('stu train end')
                    train_log_stu.write(datetime.datetime.now().strftime('%F %T') + '\n')
                    train_log_stu.close()
            else:
                # student training without teacher
                for i in range(config.epochs):

                    train_batch_count = 0
                    train_iter = trainset.make_one_shot_iterator()
                    train_values = train_iter.get_next()
                    try:
                        while True:
                            train_batch_values = sess.run(train_values)

                            train_feed_dict = {paper_feat: train_batch_values[0], auth_feats: train_batch_values[1],
                                               topic_feats: train_batch_values[2], ref_feats: train_batch_values[3],
                                               ref_auth_feats: train_batch_values[4],
                                               auth_paper_feats: train_batch_values[5],
                                               coauth_feats: train_batch_values[6],
                                               topic_paper_feats: train_batch_values[7],
                                               topic_paper_auth_feats: train_batch_values[8],
                                               n_citation: train_batch_values[9],
                                               auth_cnt: train_batch_values[10], topic_cnt: train_batch_values[11],
                                               ref_cnt: train_batch_values[12],
                                               ref_auth_cnt: train_batch_values[13],
                                               auth_paper_cnt: train_batch_values[14],
                                               coauth_cnt: train_batch_values[15],
                                               topic_paper_cnt: train_batch_values[16],
                                               topic_paper_auth_cnt: train_batch_values[17]}

                            sess.run(train_step_stu_no_tea, feed_dict=train_feed_dict)

                            train_batch_count += 1
                            if train_batch_count % 100 == 0:
                                print(train_batch_count)
                                loss_gt = sess.run(msle_stu, feed_dict=train_feed_dict)
                                batch_acc_stu = sess.run(acc_stu, feed_dict=train_feed_dict)
                                train_log_stu.write(
                                    'stu epoch : ' + str(i + 1) + ' batch: ' + str(train_batch_count) + ' loss: ' + str(
                                        loss_gt) + ' acc: ' + str(batch_acc_stu) + '\n')

                    except tf.errors.OutOfRangeError:
                        print('stu epoch ' + str(i + 1) + ' end.\n')

                    test_iter = testset.make_one_shot_iterator()
                    test_values = test_iter.get_next()
                    test_counts = 0
                    test_loss_gt = 0
                    test_acc_stu = 0
                    try:
                        while True:
                            test_batch_values = sess.run(test_values)

                            test_feed_dict = {paper_feat: test_batch_values[0], auth_feats: test_batch_values[1],
                                              topic_feats: test_batch_values[2], ref_feats: test_batch_values[3],
                                              ref_auth_feats: test_batch_values[4],
                                              auth_paper_feats: test_batch_values[5],
                                              coauth_feats: test_batch_values[6],
                                              topic_paper_feats: test_batch_values[7],
                                              topic_paper_auth_feats: test_batch_values[8],
                                              n_citation: test_batch_values[9],
                                              auth_cnt: test_batch_values[10], topic_cnt: test_batch_values[11],
                                              ref_cnt: test_batch_values[12],
                                              ref_auth_cnt: test_batch_values[13],
                                              auth_paper_cnt: test_batch_values[14],
                                              coauth_cnt: test_batch_values[15],
                                              topic_paper_cnt: test_batch_values[16],
                                              topic_paper_auth_cnt: test_batch_values[17]}

                            test_counts += test_batch_values[0].shape[0]
                            test_loss_gt += sess.run(msle_stu, feed_dict=test_feed_dict)
                            test_acc_stu += sess.run(acc_stu, feed_dict=test_feed_dict)

                    except tf.errors.OutOfRangeError:
                        test_loss_gt /= test_counts
                        test_acc_stu /= test_counts
                        train_log_stu.write('stu epoch : ' + str(i + 1) + ' testing loss: ' + str(
                            test_loss_gt) + ' testing acc: ' + str(test_acc_stu) + '\n')

                print('stu train end')
                train_log_stu.write(datetime.datetime.now().strftime('%F %T') + '\n')
                train_log_stu.close()



if __name__ == '__main__':
    main()
