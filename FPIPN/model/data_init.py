import pickle
import json
import os
import tensorflow as tf


def read_tfrecord(serialized_example, years):
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16 * 2 + 1
    context_features = {
        'n_citation0': tf.FixedLenFeature([], dtype=tf.int64),
        'n_citation1': tf.FixedLenFeature([], dtype=tf.int64),
        'n_citation2': tf.FixedLenFeature([], dtype=tf.int64),
        'n_citation3': tf.FixedLenFeature([], dtype=tf.int64),
        'year': tf.FixedLenFeature([], dtype=tf.int64),
        'paper_feat': tf.FixedLenFeature([paper_feat_len], dtype=tf.float32),
        'auth_paper_cnt': tf.FixedLenFeature([6], dtype=tf.int64),
        'coauth_feats_cnt': tf.FixedLenFeature([60], dtype=tf.int64),
        'auth_cnt': tf.FixedLenFeature([], dtype=tf.int64),
        'ref_paper_cnt': tf.FixedLenFeature([], dtype=tf.int64),
        'ref_auth_cnt': tf.FixedLenFeature([25], dtype=tf.int64),
        'topic_cnt': tf.FixedLenFeature([], dtype=tf.int64),
        'topic_paper_cnt': tf.FixedLenFeature([12], dtype=tf.int64),
        'topic_paper_auth_cnt': tf.FixedLenFeature([12], dtype=tf.int64),
    }
    sequence_features = {
        # 作者特征 [6，37]
        "author_feats": tf.VarLenFeature(dtype=tf.float32),
        # 作者代表作文章特征 [6*10, 289]
        "auth_paper_feats": tf.VarLenFeature(dtype=tf.float32),
        # 代表作的所有合作者特征 [6*(6-1)*10, 37]
        "coauth_feats": tf.VarLenFeature(dtype=tf.float32),
        # 参考文献特征：[25, 289]
        "ref_feats": tf.VarLenFeature(dtype=tf.float32),
        # 参考文献作者特征 :[25*6, 37]
        "ref_auth_feats": tf.VarLenFeature(dtype=tf.float32),
        # topic特征 : [12, 130]
        "topic_feats": tf.VarLenFeature(dtype=tf.float32),
        # topic 相关论文：[12*10, 289]
        "topic_paper_feats": tf.VarLenFeature(dtype=tf.float32),
        # topic 相关论文的作者：[12*10*6, 37]
        "topic_paper_auth_feats": tf.VarLenFeature(dtype=tf.float32),
    }
    context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    paper_feat = context['paper_feat']
    auth_feats = tf.sparse_tensor_to_dense(sequence['author_feats'])
    topic_feats = tf.sparse_tensor_to_dense(sequence['topic_feats'])
    ref_feats = tf.sparse_tensor_to_dense(sequence['ref_feats'])
    ref_auth_feats = tf.sparse_tensor_to_dense(sequence['ref_auth_feats'])
    auth_paper_feats = tf.sparse_tensor_to_dense(sequence['auth_paper_feats'])
    coauth_feats = tf.sparse_tensor_to_dense(sequence['coauth_feats'])
    topic_paper_feats = tf.sparse_tensor_to_dense(sequence['topic_paper_feats'])
    topic_paper_auth_feats = tf.sparse_tensor_to_dense(sequence['topic_paper_auth_feats'])

    n_citation = context['n_citation'+str(years)]

    auth_cnt = context['auth_cnt']
    topic_cnt = context['topic_cnt']
    ref_cnt = context['ref_paper_cnt']
    ref_auth_cnt = context['ref_auth_cnt']
    auth_paper_cnt = context['auth_paper_cnt']
    coauth_cnt = context['coauth_feats_cnt']
    topic_paper_cnt = context['topic_paper_cnt']
    topic_paper_auth_cnt = context['topic_paper_auth_cnt']

    return paper_feat, auth_feats, topic_feats, ref_feats, ref_auth_feats, auth_paper_feats, \
           coauth_feats, topic_paper_feats, topic_paper_auth_feats, n_citation, auth_cnt, topic_cnt,\
           ref_cnt, ref_auth_cnt, auth_paper_cnt, coauth_cnt, topic_paper_cnt, topic_paper_auth_cnt


def my_filter(*data):

    return tf.greater(data[9], 0)

def read(file_paths, epochs, buffer_size, batch_size, years):
    # file_paths = [file_path]  # We have only one file
    tfrecord_dataset = tf.data.TFRecordDataset(file_paths)
    parsed_dataset = tfrecord_dataset.map(lambda x: read_tfrecord(x, years))
    parsed_dataset = parsed_dataset.filter(my_filter)
    parsed_dataset = parsed_dataset.repeat(epochs).shuffle(buffer_size).batch(batch_size)

    return parsed_dataset


def main():
    train_path = ['tfrecord/data_0.tfrecord']
    tf.enable_eager_execution()
    parsed_dataset = read(train_path, 1, 100, 10, 1)
    train_batch_count = 0

    for values in parsed_dataset:
        train_batch_count += 1
        if train_batch_count % 1000 == 0:
            print(train_batch_count)
    print(train_batch_count)

if __name__ == '__main__':
    main()
