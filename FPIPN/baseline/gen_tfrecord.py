from datetime import datetime
import json
import os
import random
import sys
import threading
import tqdm
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if type(value) not in [list, np.ndarray]:
      value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if type(value) not in [list, np.ndarray]:
      value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
  
def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _int64_list_feature_list(values):
  """Wrapper for inserting an int64 list FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _float_feature_list(values):
  """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


# data = [
#     {
#         'feat1': [2,3,4,5],
#         'feat2': [[1,2], [5,6]],
#         'label': 1
#     },
#     {
#         'feat1': [1,3,4,5],
#         'feat2': [[555.2,2], [5,6],[3,3,3,3]],
#         'label': 0
#     }

# ]

def serialize_example(feature0, feature1):
    feature = {
        'feature0': _int64_feature_list(feature0),
        'feature1': _float_feature_list(feature1),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature)) 
    return example_proto.SerializeToString()

def gen_tfrecord():
    file_path = 'data/tfrecords/data.tfrecords'
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for d in data:

            serialized_example = serialize_example(tf.serialize_tensor(d['feat1']), tf.serialize_tensor(d['feat2']))

                                        # tf.io.serialize_tensor(feature3))    
            writer.write(serialized_example)

def get_paper_feat(paper_id, doc_type, pub_name, venue, abstract_embed, title_embed, doc_embed, pub_embed, venue_embed):
    if doc_type == '':
        doc_emb = np.zeros(1)
    else:
        doc_emb = doc_embed[doc_type]
    if venue == '':
        venue_emb = np.zeros(16)
    else:
        venue_emb = venue_embed[venue]
    if pub_name == '':
        pub_emb = np.zeros(16)
    else:
        pub_emb = pub_embed[pub_name]

    return np.concatenate((abstract_embed.get(paper_id, np.zeros(128)), title_embed.get(paper_id, np.zeros(128)), doc_emb, venue_emb, pub_emb), axis=0)

def get_auth_feat(auth_id, year, org_embed, auth_feats):
    if auth_id not in auth_feats:
        print('aid not in auth_feats')
        return np.zeros(37)

    if 'author_org' in auth_feats[auth_id]:
        org = auth_feats[auth_id]['author_org']
        org_emb = org_embed[org]
    else:
        org_emb = [0.0] * 32

    year = year - 2000
    nps = all_fit['n_papers'].transform([[auth_feats[auth_id]['n_papers'][year]]])[0,0]
    nc = all_fit['n_citation'].transform([[auth_feats[auth_id]['n_citation'][year]]])[0,0]
    nprs = all_fit['n_partners'].transform([[auth_feats[auth_id]['n_partners'][year]]])[0,0]
    ntp = all_fit['n_topic'].transform([[auth_feats[auth_id]['n_topic'][year]]])[0,0]
    fpt = all_fit['first_paper_time'].transform([[auth_feats[auth_id]['first_paper_time']]])[0,0]


    feats = np.concatenate((org_emb, [nps, nc, nprs, ntp, fpt]), axis=0)
    return feats

def get_topic_feat(t_name, year, topic_embed, topic_all_feat):
    year = year - 2000
    wby = all_fit['weights_by_year'].transform([[topic_all_feat[t_name]['weights_by_year'][year]]])[0,0]
    nby = all_fit['npaper_by_year'].transform([[topic_all_feat[t_name]['npaper_by_year'][year]]])[0,0]

    feat = np.concatenate((topic_embed[t_name], [wby, nby]) ,axis=0)
    return feat



def _process_dblp_files(thread_index, all_thread, train_data, 
        paper_info, abstract_embed, title_embed, topic_embed, 
        org_embed, doc_embed, pub_embed, venue_embed, references_3_map, auth_feats, topic_all_feat, writer_list):

    max_auth = 6 # 每篇文章最大个数
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16*2 + 1
    max_refs = 25 # 参考文献最大
    bs = int(len(train_data) / all_thread)

    if thread_index == all_thread-1:
        sub_train = train_data[thread_index * bs:]
    else:
        sub_train = train_data[thread_index * bs:(thread_index+1)*bs]
    for idx, data in enumerate(sub_train):  # 对于每个样
        paper_id = data['paper_id']
        year = data['year']
        if year >= 2017:
            continue
        venue = data['venue']
        doc_type = data['doc_type']
        publisher = data['publisher']
        n_citation = references_3_map[paper_id]['n_citation']
        paper_embed = get_paper_feat(paper_id, doc_type, publisher, venue, abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
        # 文章作者
        author_feats = np.zeros((max_auth, author_feat_len))
        auth_paper_feats = np.zeros((max_auth*10, paper_feat_len))
        coauth_feats = np.zeros((max_auth * (max_auth-1) * 10, author_feat_len))

        auth_paper_cnt = np.zeros(max_auth,dtype=np.int) # 也要保存
        coauth_feats_cnt = np.zeros(max_auth*10,dtype=np.int) # 每个作者代表作的合作者个数
        auth_cnt = 0 # 作者个数
        for i, authdata in enumerate(data['author_neighbours'][:max_auth]):
            auth_id = authdata[0]
            auth_feat = get_auth_feat(auth_id, year, org_embed, auth_feats)
            author_feats[i] = auth_feat
            auth_cnt += 1
            # 当前作者代表作 <=10
            # auth_paper_feat = np.zeros((10, paper_feat_len))
            for j, pid in enumerate(authdata[1][0][:10]):

                auth_paper_feats[i*10+j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                    paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)


                # 代表作的作者 除了自己最多4个
                co_cnt = 0
                for aid in paper_info[pid]['authors']:
                    if co_cnt >= max_auth-1:
                        break
                    # 去掉自己
                    if aid == auth_id:
                        continue
                    
                    coauth_feats[i*10*(max_auth-1)+j*(max_auth-1)+co_cnt] = get_auth_feat(aid, year, org_embed, auth_feats)
                    co_cnt += 1

                coauth_feats_cnt[i*10+j] = co_cnt
            auth_paper_cnt[i] = len(authdata[1][0])

            
        # ref相关特征
        ref_feats = np.zeros((max_refs, paper_feat_len))
        ref_auth_feats = np.zeros((max_refs*max_auth, author_feat_len))
        ref_paper_cnt = 0 # 参考文献个数
        ref_auth_cnt = np.zeros(max_refs,dtype=np.int) # 参考文献作者的个数
        for i, refdata in enumerate(data['paper_neighbours'][:max_refs]):
            pid = refdata[0]
            ab_embed = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                            paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
            ref_feats[ref_paper_cnt] = ab_embed
            ref_paper_cnt += 1
            # 每篇文章作者最多
            for j, aid in enumerate(paper_info[pid]['authors'][:max_auth]):
                ref_auth_feats[i*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)
            ref_auth_cnt[i] = len(data['paper_neighbours']) if len(data['paper_neighbours'])<10 else 10
        
        # topic 相关特征
        topic_feat_len = 128 + 2
        max_topic = 12
        topic_cnt = 0
        topic_paper_cnt = np.zeros(max_topic,dtype=np.int)
        topic_paper_auth_cnt = np.zeros(max_topic,dtype=np.int)
        topic_feats = np.zeros((max_topic, topic_feat_len))
        topic_paper_feats = np.zeros((max_topic*10, paper_feat_len))
        topic_paper_auth_feats = np.zeros((max_topic*10*max_auth, author_feat_len))
        for i, topicdata in enumerate(data['topic_neighbours'][:max_topic]):
            t_name = topicdata[0]
            topic_feats[topic_cnt] = get_topic_feat(t_name, year, topic_embed, topic_all_feat)
            topic_cnt += 1
            for j, pid in enumerate(topicdata[1][:10]):
                topic_paper_feats[i*10 +j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)

            topic_paper_cnt[i] = len(topicdata[1]) if len(topicdata[1])<10 else 10
            for j, aid in enumerate(topicdata[2][:10*max_auth]):
                topic_paper_auth_feats[i*10*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)

            topic_paper_auth_cnt[i] = len(topicdata[2]) if len(topicdata[2])<10*max_auth else 10*max_auth
        

        feat = {}
        feat['n_citation'] = _int64_feature(n_citation)
        feat['year'] = _int64_feature(year)
        feat['paper_feat'] = _float_feature(paper_embed)

        feat['auth_paper_cnt'] = _int64_feature(auth_paper_cnt)
        feat['coauth_feats_cnt'] = _int64_feature(coauth_feats_cnt)
        feat['auth_cnt'] = _int64_feature(auth_cnt)
        feat['ref_paper_cnt'] = _int64_feature(ref_paper_cnt)
        feat['ref_auth_cnt'] = _int64_feature(ref_auth_cnt)
        feat['topic_cnt'] = _int64_feature(topic_cnt)
        feat['topic_paper_cnt'] = _int64_feature(topic_paper_cnt)
        feat['topic_paper_auth_cnt'] = _int64_feature(topic_paper_auth_cnt)

        context = tf.train.Features(feature=feat)

        feature_lists = tf.train.FeatureLists(feature_list={
            "author_feats": _float_feature_list(author_feats),
            "auth_paper_feats": _float_feature_list(auth_paper_feats),
            "coauth_feats": _float_feature_list(coauth_feats),
            "ref_feats": _float_feature_list(ref_feats),
            "ref_auth_feats": _float_feature_list(ref_auth_feats),
            "topic_feats": _float_feature_list(topic_feats),
            "topic_paper_feats": _float_feature_list(topic_paper_feats),
            "topic_paper_auth_feats": _float_feature_list(topic_paper_auth_feats)
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

        serialized = sequence_example.SerializeToString()
        
        # file_idx = idx % 10
        writer_list[thread_index].write(serialized)  # 写入文件中
    # t = [writer_list[i].close() for i in range(10)]






def generate_tfrecords():
    # X=[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[11.0,12.0,13.0,14.0],[20.0,21.0,22.0,23.0]]
    # Y=[[1],[2],[3],[4]]
    # root = '/home/hefu/aminer/pre/embed/'
    root = 'data/embed/'
    # 加载特征数据
    abstract_embed = pickle.load( open( "data/embed/p_abstract_embed_DBOW_128.pkl", "rb" ))
    title_embed = pickle.load( open( "data/embed/p_title_embed_DBOW_128.pkl", "rb" ))
    topic_embed = pickle.load( open( "data/embed/topic_embed_fasttext_128.pkl", "rb" ))
    with open(root + 'author_orgs_embed.json', 'r', encoding='utf8') as fin:
        org_embed = json.load(fin)  
    with open(root + 'doc_types_embed.json', 'r', encoding='utf8') as fin:
        doc_embed = json.load(fin)  
    with open(root + 'publisher_names_embed.json', 'r', encoding='utf8') as fin:
        pub_embed = json.load(fin)  
    with open(root + 'venue_names_embed.json', 'r', encoding='utf8') as fin:
        venue_embed = json.load(fin)  

    train_data = pickle.load( open( "data/sample/train_data.pkl", "rb" ))
    with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
        references_3_map = json.load(fin) 
    auth_feats = {} 
    with open('data/embed/authors_feat_map.json', 'r', encoding='utf8') as fin:
        auth_feats = json.load(fin) 
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_all_feat = json.load(fin) 

    # 归一化
    cum_paper, cum_cite, cum_co, cum_topic, cum_first  = [],[],[],[],[]
    cum_weight, cum_time_paper = [], []
    for aid in auth_feats:
        cum_paper.extend(auth_feats[aid]['n_papers'])
        cum_cite.extend(auth_feats[aid]['n_citation'])
        cum_co.extend(auth_feats[aid]['n_partners'])
        cum_topic.extend(auth_feats[aid]['n_topic'])
        cum_first.append(auth_feats[aid]['first_paper_time']) # min_max
    for tid in topic_all_feat:
        cum_weight.extend(topic_all_feat[tid]['weights_by_year'])
        cum_time_paper.extend(topic_all_feat[tid]['npaper_by_year'])

    print('preprocessing ...')
    global all_fit
    all_fit = {
        'n_papers': StandardScaler().fit(np.reshape(cum_paper, (-1,1))),
        'n_citation': StandardScaler().fit(np.reshape(cum_cite, (-1,1))),
        'n_partners': StandardScaler().fit(np.reshape(cum_co, (-1,1))),
        'n_topic': StandardScaler().fit(np.reshape(cum_topic, (-1,1))),
        'first_paper_time': StandardScaler().fit(np.reshape(cum_first, (-1,1))),
        'weights_by_year': StandardScaler().fit(np.reshape(cum_weight, (-1,1))),
        'npaper_by_year': StandardScaler().fit(np.reshape(cum_time_paper, (-1,1))),

    }
    

    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    # 加载paper信息
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                paper_id = one_paper['id'] 
                
                paper_info[paper_id] = {}
                paper_info[paper_id]['year'] = one_paper.get('year', -1)
                paper_info[paper_id]['n_citation'] = one_paper.get('n_citation', -1)
                if 'authors' in one_paper:
                    auth_all = []
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                    paper_info[paper_id]['authors'] = auth_all
                paper_info[paper_id]['refs'] = one_paper.get('references', [])
                
                paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
                paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
                if 'venue' in one_paper:
                    paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
                else:
                    paper_info[paper_id]['venue'] = ''


    # threads = []
    # coord = tf.train.Coordinator()
    # all_thread = 10
    # writer_list = [tf.python_io.TFRecordWriter('tfrecord_thd/data_{}.tfrecord'.format(i)) for i in range(all_thread)]# 这是tf2.x版本
    # for thread_index in range(all_thread):
    #     print(thread_index)
    #     args = (thread_index, all_thread, train_data, 
    #         paper_info, abstract_embed, title_embed, topic_embed, 
    #         org_embed, doc_embed, pub_embed, venue_embed, references_3_map, auth_feats, topic_all_feat, writer_list)
            
    #     t = threading.Thread(target=_process_dblp_files, args=args)
    #     t.start()
    #     threads.append(t)

    # # Wait for all the threads to terminate.
    # coord.join(threads)
    # t = [writer_list[i].close() for i in range(all_thread)]



    max_auth = 6 # 每篇文章最大个数
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16*2 + 1
    max_refs = 25 # 参考文献最大
    writer_list = [tf.python_io.TFRecordWriter('tfrecord/data_{}.tfrecord'.format(i)) for i in range(10)]# 这是tf2.x版本
    for idx, data in enumerate(tqdm.tqdm(train_data)):  # 对于每个样本

        # x = data[i]['feat2']
        # y = data[i]['label']
        paper_id = data['paper_id']
        year = data['year']
        if year >= 2017:
            continue
        venue = data['venue']
        doc_type = data['doc_type']
        publisher = data['publisher']
        n_citation0 = references_3_map[paper_id]['n_citation0']
        n_citation1 = references_3_map[paper_id]['n_citation1']
        n_citation2 = references_3_map[paper_id]['n_citation2']
        n_citation3 = references_3_map[paper_id]['n_citation3']
        paper_embed = get_paper_feat(paper_id, doc_type, publisher, venue, abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
        # 文章作者
        author_feats = np.zeros((max_auth, author_feat_len))
        auth_paper_feats = np.zeros((max_auth*10, paper_feat_len))
        coauth_feats = np.zeros((max_auth * (max_auth-1) * 10, author_feat_len))

        auth_paper_cnt = np.zeros(max_auth,dtype=np.int) # 也要保存
        coauth_feats_cnt = np.zeros(max_auth*10,dtype=np.int) # 每个作者代表作的合作者个数
        auth_cnt = 0 # 作者个数
        for i, authdata in enumerate(data['author_neighbours'][:max_auth]):
            auth_id = authdata[0]
            auth_feat = get_auth_feat(auth_id, year, org_embed, auth_feats)
            author_feats[i] = auth_feat
            auth_cnt += 1
            # 当前作者代表作 <=10
            # auth_paper_feat = np.zeros((10, paper_feat_len))
            for j, pid in enumerate(authdata[1][0][:10]):

                auth_paper_feats[i*10+j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                    paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)


                # 代表作的作者 除了自己最多4个
                co_cnt = 0
                for aid in paper_info[pid]['authors']:
                    if co_cnt >= max_auth-1:
                        break
                    # 去掉自己
                    if aid == auth_id:
                        continue
                    
                    coauth_feats[i*10*(max_auth-1)+j*(max_auth-1)+co_cnt] = get_auth_feat(aid, year, org_embed, auth_feats)
                    co_cnt += 1

                coauth_feats_cnt[i*10+j] = co_cnt
            auth_paper_cnt[i] = len(authdata[1][0])

            
        # ref相关特征
        ref_feats = np.zeros((max_refs, paper_feat_len))
        ref_auth_feats = np.zeros((max_refs*max_auth, author_feat_len))
        ref_paper_cnt = 0 # 参考文献个数
        ref_auth_cnt = np.zeros(max_refs,dtype=np.int) # 参考文献作者的个数
        for i, refdata in enumerate(data['paper_neighbours'][:max_refs]):
            pid = refdata[0]
            ab_embed = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                            paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
            ref_feats[ref_paper_cnt] = ab_embed
            ref_paper_cnt += 1
            # 每篇文章作者最多
            for j, aid in enumerate(paper_info[pid]['authors'][:max_auth]):
                ref_auth_feats[i*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)
            ref_auth_cnt[i] = len(data['paper_neighbours']) if len(data['paper_neighbours'])<10 else 10
        
        # topic 相关特征
        topic_feat_len = 128 + 2
        max_topic = 12
        topic_cnt = 0
        topic_paper_cnt = np.zeros(max_topic,dtype=np.int)
        topic_paper_auth_cnt = np.zeros(max_topic,dtype=np.int)
        topic_feats = np.zeros((max_topic, topic_feat_len))
        topic_paper_feats = np.zeros((max_topic*10, paper_feat_len))
        topic_paper_auth_feats = np.zeros((max_topic*10*max_auth, author_feat_len))
        for i, topicdata in enumerate(data['topic_neighbours'][:max_topic]):
            t_name = topicdata[0]
            topic_feats[topic_cnt] = get_topic_feat(t_name, year, topic_embed, topic_all_feat)
            topic_cnt += 1
            for j, pid in enumerate(topicdata[1][:10]):
                topic_paper_feats[i*10 +j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)

            topic_paper_cnt[i] = len(topicdata[1]) if len(topicdata[1])<10 else 10
            for j, aid in enumerate(topicdata[2][:10*max_auth]):
                topic_paper_auth_feats[i*10*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)

            topic_paper_auth_cnt[i] = len(topicdata[2]) if len(topicdata[2])<10*max_auth else 10*max_auth


        feat = {}
        feat['paper_id'] = _int64_feature(int(paper_id))
        # feat['p_id_target'] = _int64_feature(-1)
        feat['n_citation0'] = _int64_feature(n_citation0)
        feat['n_citation1'] = _int64_feature(n_citation1)
        feat['n_citation2'] = _int64_feature(n_citation2)
        feat['n_citation3'] = _int64_feature(n_citation3)
        feat['year'] = _int64_feature(year)
        feat['paper_feat'] = _float_feature(paper_embed)

        feat['auth_paper_cnt'] = _int64_feature(auth_paper_cnt)
        feat['coauth_feats_cnt'] = _int64_feature(coauth_feats_cnt)
        feat['auth_cnt'] = _int64_feature(auth_cnt)
        feat['ref_paper_cnt'] = _int64_feature(ref_paper_cnt)
        feat['ref_auth_cnt'] = _int64_feature(ref_auth_cnt)
        feat['topic_cnt'] = _int64_feature(topic_cnt)
        feat['topic_paper_cnt'] = _int64_feature(topic_paper_cnt)
        feat['topic_paper_auth_cnt'] = _int64_feature(topic_paper_auth_cnt)

        context = tf.train.Features(feature=feat)

        feature_lists = tf.train.FeatureLists(feature_list={
            "author_feats": _float_feature_list(author_feats),
            "auth_paper_feats": _float_feature_list(auth_paper_feats),
            "coauth_feats": _float_feature_list(coauth_feats),
            "ref_feats": _float_feature_list(ref_feats),
            "ref_auth_feats": _float_feature_list(ref_auth_feats),
            "topic_feats": _float_feature_list(topic_feats),
            "topic_paper_feats": _float_feature_list(topic_paper_feats),
            "topic_paper_auth_feats": _float_feature_list(topic_paper_auth_feats)
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

        serialized = sequence_example.SerializeToString()
        
        file_idx = idx % 10
        writer_list[file_idx].write(serialized)  # 写入文件中
    
    t = [writer_list[i].close() for i in range(10)]
    print('records 文件保存完毕.')  # 保存结束之后会得到一个  example.tfrecord 文件
    


def generate_tfrecords_casev2():
    """
    保存学者之前的数据：多了一个target_pid
    """
    # X=[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[11.0,12.0,13.0,14.0],[20.0,21.0,22.0,23.0]]
    # Y=[[1],[2],[3],[4]]
    # root = '/home/hefu/aminer/pre/embed/'
    root = 'data/embed/'
    # 加载特征数据
    abstract_embed = pickle.load( open( "data/embed/p_abstract_embed_DBOW_128.pkl", "rb" ))
    title_embed = pickle.load( open( "data/embed/p_title_embed_DBOW_128.pkl", "rb" ))
    topic_embed = pickle.load( open( "data/embed/topic_embed_fasttext_128.pkl", "rb" ))
    with open(root + 'author_orgs_embed.json', 'r', encoding='utf8') as fin:
        org_embed = json.load(fin)  
    with open(root + 'doc_types_embed.json', 'r', encoding='utf8') as fin:
        doc_embed = json.load(fin)  
    with open(root + 'publisher_names_embed.json', 'r', encoding='utf8') as fin:
        pub_embed = json.load(fin)  
    with open(root + 'venue_names_embed.json', 'r', encoding='utf8') as fin:
        venue_embed = json.load(fin)  
    
    train_data = pickle.load( open( "data/sample/train_data_case.pkl", "rb" ))
    with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
        references_3_map = json.load(fin) 
    auth_feats = {} 
    with open('data/embed/authors_feat_map.json', 'r', encoding='utf8') as fin:
        auth_feats = json.load(fin) 
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_all_feat = json.load(fin) 

    # 归一化
    cum_paper, cum_cite, cum_co, cum_topic, cum_first  = [],[],[],[],[]
    cum_weight, cum_time_paper = [], []
    for aid in auth_feats:
        cum_paper.extend(auth_feats[aid]['n_papers'])
        cum_cite.extend(auth_feats[aid]['n_citation'])
        cum_co.extend(auth_feats[aid]['n_partners'])
        cum_topic.extend(auth_feats[aid]['n_topic'])
        cum_first.append(auth_feats[aid]['first_paper_time']) # min_max
    for tid in topic_all_feat:
        cum_weight.extend(topic_all_feat[tid]['weights_by_year'])
        cum_time_paper.extend(topic_all_feat[tid]['npaper_by_year'])

    print('preprocessing ...')
    global all_fit
    all_fit = {
        'n_papers': StandardScaler().fit(np.reshape(cum_paper, (-1,1))),
        'n_citation': StandardScaler().fit(np.reshape(cum_cite, (-1,1))),
        'n_partners': StandardScaler().fit(np.reshape(cum_co, (-1,1))),
        'n_topic': StandardScaler().fit(np.reshape(cum_topic, (-1,1))),
        'first_paper_time': StandardScaler().fit(np.reshape(cum_first, (-1,1))),
        'weights_by_year': StandardScaler().fit(np.reshape(cum_weight, (-1,1))),
        'npaper_by_year': StandardScaler().fit(np.reshape(cum_time_paper, (-1,1))),

    }
    

    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    # 加载paper信息
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                paper_id = one_paper['id'] 
                
                paper_info[paper_id] = {}
                paper_info[paper_id]['year'] = one_paper.get('year', -1)
                paper_info[paper_id]['n_citation'] = one_paper.get('n_citation', -1)
                if 'authors' in one_paper:
                    auth_all = []
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                    paper_info[paper_id]['authors'] = auth_all
                paper_info[paper_id]['refs'] = one_paper.get('references', [])
                
                paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
                paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
                if 'venue' in one_paper:
                    paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
                else:
                    paper_info[paper_id]['venue'] = ''



    max_auth = 6 # 每篇文章最大个数
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16*2 + 1
    max_refs = 25 # 参考文献最大
    writer_list = [tf.python_io.TFRecordWriter('tfrecord/case_data_{}.tfrecord'.format(i)) for i in range(10)]# 这是tf2.x版本
    for idx, data in enumerate(tqdm.tqdm(train_data)):  # 对于每个样本

        # x = data[i]['feat2']
        # y = data[i]['label']
        paper_id = data['paper_id']
        year = data['year']
        if year >= 2017:
            continue
        venue = data['venue']
        doc_type = data['doc_type']
        publisher = data['publisher']
        n_citation0 = references_3_map[paper_id]['n_citation0']
        n_citation1 = references_3_map[paper_id]['n_citation1']
        n_citation2 = references_3_map[paper_id]['n_citation2']
        n_citation3 = references_3_map[paper_id]['n_citation3']
        paper_embed = get_paper_feat(paper_id, doc_type, publisher, venue, abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
        # 文章作者
        author_feats = np.zeros((max_auth, author_feat_len))
        auth_paper_feats = np.zeros((max_auth*10, paper_feat_len))
        coauth_feats = np.zeros((max_auth * (max_auth-1) * 10, author_feat_len))

        auth_paper_cnt = np.zeros(max_auth,dtype=np.int) # 也要保存
        coauth_feats_cnt = np.zeros(max_auth*10,dtype=np.int) # 每个作者代表作的合作者个数
        auth_cnt = 0 # 作者个数
        for i, authdata in enumerate(data['author_neighbours'][:max_auth]):
            auth_id = authdata[0]
            auth_feat = get_auth_feat(auth_id, year, org_embed, auth_feats)
            author_feats[i] = auth_feat
            auth_cnt += 1
            # 当前作者代表作 <=10
            # auth_paper_feat = np.zeros((10, paper_feat_len))
            for j, pid in enumerate(authdata[1][0][:10]):

                auth_paper_feats[i*10+j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                    paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)


                # 代表作的作者 除了自己最多4个
                co_cnt = 0
                for aid in paper_info[pid]['authors']:
                    if co_cnt >= max_auth-1:
                        break
                    # 去掉自己
                    if aid == auth_id:
                        continue
                    
                    coauth_feats[i*10*(max_auth-1)+j*(max_auth-1)+co_cnt] = get_auth_feat(aid, year, org_embed, auth_feats)
                    co_cnt += 1

                coauth_feats_cnt[i*10+j] = co_cnt
            auth_paper_cnt[i] = len(authdata[1][0])

            
        # ref相关特征
        ref_feats = np.zeros((max_refs, paper_feat_len))
        ref_auth_feats = np.zeros((max_refs*max_auth, author_feat_len))
        ref_paper_cnt = 0 # 参考文献个数
        ref_auth_cnt = np.zeros(max_refs,dtype=np.int) # 参考文献作者的个数
        for i, refdata in enumerate(data['paper_neighbours'][:max_refs]):
            pid = refdata[0]
            ab_embed = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                            paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
            ref_feats[ref_paper_cnt] = ab_embed
            ref_paper_cnt += 1
            # 每篇文章作者最多
            for j, aid in enumerate(paper_info[pid]['authors'][:max_auth]):
                ref_auth_feats[i*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)
            ref_auth_cnt[i] = len(data['paper_neighbours']) if len(data['paper_neighbours'])<10 else 10
        
        # topic 相关特征
        topic_feat_len = 128 + 2
        max_topic = 12
        topic_cnt = 0
        topic_paper_cnt = np.zeros(max_topic,dtype=np.int)
        topic_paper_auth_cnt = np.zeros(max_topic,dtype=np.int)
        topic_feats = np.zeros((max_topic, topic_feat_len))
        topic_paper_feats = np.zeros((max_topic*10, paper_feat_len))
        topic_paper_auth_feats = np.zeros((max_topic*10*max_auth, author_feat_len))
        for i, topicdata in enumerate(data['topic_neighbours'][:max_topic]):
            t_name = topicdata[0]
            topic_feats[topic_cnt] = get_topic_feat(t_name, year, topic_embed, topic_all_feat)
            topic_cnt += 1
            for j, pid in enumerate(topicdata[1][:10]):
                topic_paper_feats[i*10 +j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)

            topic_paper_cnt[i] = len(topicdata[1]) if len(topicdata[1])<10 else 10
            for j, aid in enumerate(topicdata[2][:10*max_auth]):
                topic_paper_auth_feats[i*10*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)

            topic_paper_auth_cnt[i] = len(topicdata[2]) if len(topicdata[2])<10*max_auth else 10*max_auth


        feat = {}
        feat['n_citation0'] = _int64_feature(n_citation0)
        feat['n_citation1'] = _int64_feature(n_citation1)
        feat['n_citation2'] = _int64_feature(n_citation2)
        feat['n_citation3'] = _int64_feature(n_citation3)
        feat['year'] = _int64_feature(year)
        feat['p_id_target'] = _int64_feature(int(data['p_id_target']))
        feat['paper_id'] = _int64_feature(int(paper_id))
        feat['paper_feat'] = _float_feature(paper_embed)


        feat['auth_paper_cnt'] = _int64_feature(auth_paper_cnt)
        feat['coauth_feats_cnt'] = _int64_feature(coauth_feats_cnt)
        feat['auth_cnt'] = _int64_feature(auth_cnt)
        feat['ref_paper_cnt'] = _int64_feature(ref_paper_cnt)
        feat['ref_auth_cnt'] = _int64_feature(ref_auth_cnt)
        feat['topic_cnt'] = _int64_feature(topic_cnt)
        feat['topic_paper_cnt'] = _int64_feature(topic_paper_cnt)
        feat['topic_paper_auth_cnt'] = _int64_feature(topic_paper_auth_cnt)

        context = tf.train.Features(feature=feat)

        feature_lists = tf.train.FeatureLists(feature_list={
            "author_feats": _float_feature_list(author_feats),
            "auth_paper_feats": _float_feature_list(auth_paper_feats),
            "coauth_feats": _float_feature_list(coauth_feats),
            "ref_feats": _float_feature_list(ref_feats),
            "ref_auth_feats": _float_feature_list(ref_auth_feats),
            "topic_feats": _float_feature_list(topic_feats),
            "topic_paper_feats": _float_feature_list(topic_paper_feats),
            "topic_paper_auth_feats": _float_feature_list(topic_paper_auth_feats)
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

        serialized = sequence_example.SerializeToString()
        
        file_idx = idx % 10
        writer_list[file_idx].write(serialized)  # 写入文件中
    
    t = [writer_list[i].close() for i in range(10)]
    print('records 文件保存完毕.')  # 保存结束之后会得到一个  example.tfrecord 文件
 

def generate_tfrecords_case():
    # X=[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[11.0,12.0,13.0,14.0],[20.0,21.0,22.0,23.0]]
    # Y=[[1],[2],[3],[4]]
    # root = '/home/hefu/aminer/pre/embed/'
    root = 'data/embed/'
    # 加载特征数据
    abstract_embed = pickle.load( open( "data/embed/p_abstract_embed_DBOW_128.pkl", "rb" ))
    title_embed = pickle.load( open( "data/embed/p_title_embed_DBOW_128.pkl", "rb" ))
    topic_embed = pickle.load( open( "data/embed/topic_embed_fasttext_128.pkl", "rb" ))
    with open(root + 'author_orgs_embed.json', 'r', encoding='utf8') as fin:
        org_embed = json.load(fin)  
    with open(root + 'doc_types_embed.json', 'r', encoding='utf8') as fin:
        doc_embed = json.load(fin)  
    with open(root + 'publisher_names_embed.json', 'r', encoding='utf8') as fin:
        pub_embed = json.load(fin)  
    with open(root + 'venue_names_embed.json', 'r', encoding='utf8') as fin:
        venue_embed = json.load(fin)  

    train_data = pickle.load( open( "data/sample/train_data_case_1w.pkl", "rb" ))
    with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
        references_3_map = json.load(fin) 
    auth_feats = {} 
    with open('data/embed/authors_feat_map.json', 'r', encoding='utf8') as fin:
        auth_feats = json.load(fin) 
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_all_feat = json.load(fin) 

    # 归一化
    cum_paper, cum_cite, cum_co, cum_topic, cum_first  = [],[],[],[],[]
    cum_weight, cum_time_paper = [], []
    for aid in auth_feats:
        cum_paper.extend(auth_feats[aid]['n_papers'])
        cum_cite.extend(auth_feats[aid]['n_citation'])
        cum_co.extend(auth_feats[aid]['n_partners'])
        cum_topic.extend(auth_feats[aid]['n_topic'])
        cum_first.append(auth_feats[aid]['first_paper_time']) # min_max
    for tid in topic_all_feat:
        cum_weight.extend(topic_all_feat[tid]['weights_by_year'])
        cum_time_paper.extend(topic_all_feat[tid]['npaper_by_year'])

    print('preprocessing ...')
    global all_fit
    all_fit = {
        'n_papers': StandardScaler().fit(np.reshape(cum_paper, (-1,1))),
        'n_citation': StandardScaler().fit(np.reshape(cum_cite, (-1,1))),
        'n_partners': StandardScaler().fit(np.reshape(cum_co, (-1,1))),
        'n_topic': StandardScaler().fit(np.reshape(cum_topic, (-1,1))),
        'first_paper_time': StandardScaler().fit(np.reshape(cum_first, (-1,1))),
        'weights_by_year': StandardScaler().fit(np.reshape(cum_weight, (-1,1))),
        'npaper_by_year': StandardScaler().fit(np.reshape(cum_time_paper, (-1,1))),

    }
    

    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    # 加载paper信息
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                paper_id = one_paper['id'] 
                
                paper_info[paper_id] = {}
                paper_info[paper_id]['year'] = one_paper.get('year', -1)
                paper_info[paper_id]['n_citation'] = one_paper.get('n_citation', -1)
                if 'authors' in one_paper:
                    auth_all = []
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                    paper_info[paper_id]['authors'] = auth_all
                paper_info[paper_id]['refs'] = one_paper.get('references', [])
                
                paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
                paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
                if 'venue' in one_paper:
                    paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
                else:
                    paper_info[paper_id]['venue'] = ''




    max_auth = 6 # 每篇文章最大个数
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16*2 + 1
    max_refs = 25 # 参考文献最大
    writer_list = [tf.python_io.TFRecordWriter('case_tfrecord/data_{}.tfrecord'.format(i)) for i in range(10)]# 这是tf2.x版本
    for idx, data in enumerate(tqdm.tqdm(train_data)):  # 对于每个样本

        # x = data[i]['feat2']
        # y = data[i]['label']
        paper_id = data['paper_id']
        year = data['year']
        if year >= 2017:
            continue
        venue = data['venue']
        doc_type = data['doc_type']
        publisher = data['publisher']
        n_citation0 = references_3_map[paper_id]['n_citation0']
        n_citation1 = references_3_map[paper_id]['n_citation1']
        n_citation2 = references_3_map[paper_id]['n_citation2']
        n_citation3 = references_3_map[paper_id]['n_citation3']
        paper_embed = get_paper_feat(paper_id, doc_type, publisher, venue, abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
        # 文章作者
        author_feats = np.zeros((max_auth, author_feat_len))
        auth_paper_feats = np.zeros((max_auth*10, paper_feat_len))
        coauth_feats = np.zeros((max_auth * (max_auth-1) * 10, author_feat_len))

        auth_paper_cnt = np.zeros(max_auth,dtype=np.int) # 也要保存
        coauth_feats_cnt = np.zeros(max_auth*10,dtype=np.int) # 每个作者代表作的合作者个数
        auth_cnt = 0 # 作者个数
        for i, authdata in enumerate(data['author_neighbours'][:max_auth]):
            auth_id = authdata[0]
            auth_feat = get_auth_feat(auth_id, year, org_embed, auth_feats)
            author_feats[i] = auth_feat
            auth_cnt += 1
            # 当前作者代表作 <=10
            # auth_paper_feat = np.zeros((10, paper_feat_len))
            for j, pid in enumerate(authdata[1][0][:10]):

                auth_paper_feats[i*10+j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                    paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)


                # 代表作的作者 除了自己最多4个
                co_cnt = 0
                for aid in paper_info[pid]['authors']:
                    if co_cnt >= max_auth-1:
                        break
                    # 去掉自己
                    if aid == auth_id:
                        continue
                    
                    coauth_feats[i*10*(max_auth-1)+j*(max_auth-1)+co_cnt] = get_auth_feat(aid, year, org_embed, auth_feats)
                    co_cnt += 1

                coauth_feats_cnt[i*10+j] = co_cnt
            auth_paper_cnt[i] = len(authdata[1][0])

            
        # ref相关特征
        ref_feats = np.zeros((max_refs, paper_feat_len))
        ref_auth_feats = np.zeros((max_refs*max_auth, author_feat_len))
        ref_paper_cnt = 0 # 参考文献个数
        ref_auth_cnt = np.zeros(max_refs,dtype=np.int) # 参考文献作者的个数
        for i, refdata in enumerate(data['paper_neighbours'][:max_refs]):
            pid = refdata[0]
            ab_embed = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                            paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
            ref_feats[ref_paper_cnt] = ab_embed
            ref_paper_cnt += 1
            # 每篇文章作者最多
            for j, aid in enumerate(paper_info[pid]['authors'][:max_auth]):
                ref_auth_feats[i*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)
            ref_auth_cnt[i] = len(data['paper_neighbours']) if len(data['paper_neighbours'])<10 else 10
        
        # topic 相关特征
        topic_feat_len = 128 + 2
        max_topic = 12
        topic_cnt = 0
        topic_paper_cnt = np.zeros(max_topic,dtype=np.int)
        topic_paper_auth_cnt = np.zeros(max_topic,dtype=np.int)
        topic_feats = np.zeros((max_topic, topic_feat_len))
        topic_paper_feats = np.zeros((max_topic*10, paper_feat_len))
        topic_paper_auth_feats = np.zeros((max_topic*10*max_auth, author_feat_len))
        for i, topicdata in enumerate(data['topic_neighbours'][:max_topic]):
            t_name = topicdata[0]
            topic_feats[topic_cnt] = get_topic_feat(t_name, year, topic_embed, topic_all_feat)
            topic_cnt += 1
            for j, pid in enumerate(topicdata[1][:10]):
                topic_paper_feats[i*10 +j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)

            topic_paper_cnt[i] = len(topicdata[1]) if len(topicdata[1])<10 else 10
            for j, aid in enumerate(topicdata[2][:10*max_auth]):
                topic_paper_auth_feats[i*10*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)

            topic_paper_auth_cnt[i] = len(topicdata[2]) if len(topicdata[2])<10*max_auth else 10*max_auth


        feat = {}
        feat['n_citation0'] = _int64_feature(n_citation0)
        feat['n_citation1'] = _int64_feature(n_citation1)
        feat['n_citation2'] = _int64_feature(n_citation2)
        feat['n_citation3'] = _int64_feature(n_citation3)
        feat['year'] = _int64_feature(year)
        feat['paper_feat'] = _float_feature(paper_embed)

        feat['auth_paper_cnt'] = _int64_feature(auth_paper_cnt)
        feat['coauth_feats_cnt'] = _int64_feature(coauth_feats_cnt)
        feat['auth_cnt'] = _int64_feature(auth_cnt)
        feat['ref_paper_cnt'] = _int64_feature(ref_paper_cnt)
        feat['ref_auth_cnt'] = _int64_feature(ref_auth_cnt)
        feat['topic_cnt'] = _int64_feature(topic_cnt)
        feat['topic_paper_cnt'] = _int64_feature(topic_paper_cnt)
        feat['topic_paper_auth_cnt'] = _int64_feature(topic_paper_auth_cnt)

        context = tf.train.Features(feature=feat)

        feature_lists = tf.train.FeatureLists(feature_list={
            "author_feats": _float_feature_list(author_feats),
            "auth_paper_feats": _float_feature_list(auth_paper_feats),
            "coauth_feats": _float_feature_list(coauth_feats),
            "ref_feats": _float_feature_list(ref_feats),
            "ref_auth_feats": _float_feature_list(ref_auth_feats),
            "topic_feats": _float_feature_list(topic_feats),
            "topic_paper_feats": _float_feature_list(topic_paper_feats),
            "topic_paper_auth_feats": _float_feature_list(topic_paper_auth_feats)
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

        serialized = sequence_example.SerializeToString()
        
        file_idx = idx % 10
        writer_list[file_idx].write(serialized)  # 写入文件中
    
    t = [writer_list[i].close() for i in range(10)]
    print('records 文件保存完毕.')  # 保存结束之后会得到一个  example.tfrecord 文件
 


def generate_json():
    # X=[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[11.0,12.0,13.0,14.0],[20.0,21.0,22.0,23.0]]
    # Y=[[1],[2],[3],[4]]
    # root = '/home/hefu/aminer/pre/embed/'
    root = 'data/embed/'
    # 加载特征数据
    abstract_embed = pickle.load( open( "data/embed/p_abstract_embed_DBOW_128.pkl", "rb" ))
    title_embed = pickle.load( open( "data/embed/p_title_embed_DBOW_128.pkl", "rb" ))
    topic_embed = pickle.load( open( "data/embed/topic_embed_fasttext_128.pkl", "rb" ))
    with open(root + 'author_orgs_embed.json', 'r', encoding='utf8') as fin:
        org_embed = json.load(fin)  
    with open(root + 'doc_types_embed.json', 'r', encoding='utf8') as fin:
        doc_embed = json.load(fin)  
    with open(root + 'publisher_names_embed.json', 'r', encoding='utf8') as fin:
        pub_embed = json.load(fin)  
    with open(root + 'venue_names_embed.json', 'r', encoding='utf8') as fin:
        venue_embed = json.load(fin)  

    train_data = pickle.load( open( "data/sample/train_data.pkl", "rb" ))
    with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
        references_3_map = json.load(fin) 
    auth_feats = {} 
    with open('data/embed/authors_feat_map.json', 'r', encoding='utf8') as fin:
        auth_feats = json.load(fin) 
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_all_feat = json.load(fin) 

    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    # 加载paper信息
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                paper_id = one_paper['id'] 
                
                paper_info[paper_id] = {}
                paper_info[paper_id]['year'] = one_paper.get('year', -1)
                paper_info[paper_id]['n_citation'] = one_paper.get('n_citation', -1)
                if 'authors' in one_paper:
                    auth_all = []
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                    paper_info[paper_id]['authors'] = auth_all
                paper_info[paper_id]['refs'] = one_paper.get('references', [])
                
                paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
                paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
                if 'venue' in one_paper:
                    paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
                else:
                    paper_info[paper_id]['venue'] = ''


    max_auth = 6 # 每篇文章最大个数
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16*2 + 1
    max_refs = 25 # 参考文献最大
    # writer_list = [tf.python_io.TFRecordWriter('tfrecord/data_{}.tfrecord'.format(i)) for i in range(10)]# 这是tf2.x版本
    new_data = [[]for _ in range(10)]
    for idx, data in enumerate(tqdm.tqdm(train_data)):  # 对于每个样本

        # x = data[i]['feat2']
        # y = data[i]['label']
        paper_id = data['paper_id']
        year = data['year']
        if year >= 2017:
            continue
        venue = data['venue']
        doc_type = data['doc_type']
        publisher = data['publisher']
        n_citation = references_3_map[paper_id]['n_citation']
        paper_embed = get_paper_feat(paper_id, doc_type, publisher, venue, abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
        # 文章作者
        author_feats = np.zeros((max_auth, author_feat_len))
        auth_paper_feats = np.zeros((max_auth*10, paper_feat_len))
        coauth_feats = np.zeros((max_auth * (max_auth-1) * 10, author_feat_len))

        auth_paper_cnt = np.zeros(max_auth,dtype=np.int) # 也要保存
        coauth_feats_cnt = np.zeros(max_auth*10,dtype=np.int) # 每个作者代表作的合作者个数
        auth_cnt = 0 # 作者个数
        for i, authdata in enumerate(data['author_neighbours'][:max_auth]):
            auth_id = authdata[0]
            auth_feat = get_auth_feat(auth_id, year, org_embed, auth_feats)
            author_feats[i] = auth_feat
            auth_cnt += 1
            # 当前作者代表作 <=10
            # auth_paper_feat = np.zeros((10, paper_feat_len))
            for j, pid in enumerate(authdata[1][0][:10]):

                auth_paper_feats[i*10+j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                    paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)


                # 代表作的作者 除了自己最多4个
                co_cnt = 0
                for aid in paper_info[pid]['authors']:
                    if co_cnt >= max_auth-1:
                        break
                    # 去掉自己
                    if aid == auth_id:
                        continue
                    
                    coauth_feats[i*10*(max_auth-1)+j*(max_auth-1)+co_cnt] = get_auth_feat(aid, year, org_embed, auth_feats)
                    co_cnt += 1

                coauth_feats_cnt[i*10+j] = co_cnt
            auth_paper_cnt[i] = len(authdata[1][0])

            
        # ref相关特征
        ref_feats = np.zeros((max_refs, paper_feat_len))
        ref_auth_feats = np.zeros((max_refs*max_auth, author_feat_len))
        ref_paper_cnt = 0 # 参考文献个数
        ref_auth_cnt = np.zeros(max_refs,dtype=np.int) # 参考文献作者的个数
        for i, refdata in enumerate(data['paper_neighbours'][:max_refs]):
            pid = refdata[0]
            ab_embed = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                            paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
            ref_feats[ref_paper_cnt] = ab_embed
            ref_paper_cnt += 1
            # 每篇文章作者最多
            for j, aid in enumerate(paper_info[pid]['authors'][:max_auth]):
                ref_auth_feats[i*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)
            ref_auth_cnt[i] = len(data['paper_neighbours']) if len(data['paper_neighbours'])<10 else 10
        
        # topic 相关特征
        topic_feat_len = 128 + 2
        max_topic = 12
        topic_cnt = 0
        topic_paper_cnt = np.zeros(max_topic,dtype=np.int)
        topic_paper_auth_cnt = np.zeros(max_topic,dtype=np.int)
        topic_feats = np.zeros((max_topic, topic_feat_len))
        topic_paper_feats = np.zeros((max_topic*10, paper_feat_len))
        topic_paper_auth_feats = np.zeros((max_topic*10*max_auth, author_feat_len))
        for i, topicdata in enumerate(data['topic_neighbours'][:max_topic]):
            t_name = topicdata[0]
            topic_feats[topic_cnt] = get_topic_feat(t_name, year, topic_embed, topic_all_feat)
            topic_cnt += 1
            for j, pid in enumerate(topicdata[1][:10]):
                topic_paper_feats[i*10 +j] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)

            topic_paper_cnt[i] = len(topicdata[1]) if len(topicdata[1])<10 else 10
            for j, aid in enumerate(topicdata[2][:10*max_auth]):
                topic_paper_auth_feats[i*10*max_auth+j] = get_auth_feat(aid, year, org_embed, auth_feats)

            topic_paper_auth_cnt[i] = len(topicdata[2]) if len(topicdata[2])<10*max_auth else 10*max_auth
        

        one_data ={
            'paper_feat':paper_embed,
            'auth_feats': author_feats,
            'topic_feats': topic_feats,
            'ref_feats': ref_feats,
            'ref_auth_feats':ref_auth_feats, 
            'auth_paper_feats': auth_paper_feats, 
            'coauth_feats': coauth_feats, 
            'topic_paper_feats': topic_paper_feats,
            'topic_paper_auth_feats': topic_paper_auth_feats,
            'n_citation':n_citation

        }

        file_idx = idx % 10
        if file_idx == 0:
            new_data[file_idx].append(one_data)  # 写入文件中
    pickle.dump(new_data[0], open("data/train_json/data_{}.pkl".format(0), "wb" ))
    # for i in range(10):
    #     pickle.dump(new_data[i], open("data/train_json/data_{}.pkl".format(i), "wb" ))
    print('records 文件保存完毕.')  # 保存结束之后会得到一个  example.tfrecord 文件
    


# def my_filter(data):
#     return False
    # return tf.math.equal(data, 0)
    # # print(a)
    # # t = tf.reshape(tf.equal(b, 0), [])
    # # t = tf.unstack(b)
    # # print(t)
    # # res = tf.reshape(tf.equal(t, 0), [])
    
    # # print(tf.reshape(tf.equal(b, 0)[0], []))
    
    # return a==0


def read_tfrecord(serialized_example):
    # print(serialized_example)
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16*2 + 1
    context_features = { 
        # ***_cnt是sequence_features想对应的实际个数（因为已经pad过0）
        # 'topic_cnt': tf.FixedLenFeature([], dtype=tf.int64),
        # target
        'paper_id': tf.FixedLenFeature([], dtype=tf.int64),
        'p_id_target': tf.FixedLenFeature([], dtype=tf.int64),
        'n_citation0': tf.FixedLenFeature([], dtype=tf.int64),
        'n_citation1': tf.FixedLenFeature([], dtype=tf.int64),
        'n_citation2': tf.FixedLenFeature([], dtype=tf.int64),
        'n_citation3': tf.FixedLenFeature([], dtype=tf.int64),
        'year':tf.FixedLenFeature([], dtype=tf.int64),
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
        # topic特征 : [12, 128]
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
    # print('???', context['n_citation'])
    # print(sequence['topic_feats'])
    # 对于sequence里的特征需要用以下函数转换成tensor
    tensor_value = tf.sparse_tensor_to_dense(sequence['topic_feats'])
    # context里的特征直接取
    # if np.random.rand(1)>0.5:
    #     return None
    # else:

    return context['paper_id'], context['p_id_target'], context['n_citation1'], context['n_citation2'], context['n_citation3']

    # print(tf.decode_raw(sequence['feat1'], tf.float32))
    # print(context['label'])
    # print(sequence['feat1'])
    # return context['label'], sequence['feat1']
      


    # feature_description = {
    #     'feature0': tf.io.FixedLenFeature((), tf.int64),
    #     'feature1': tf.io.FixedLenFeature((), tf.float32),
    #     'feature2': tf.io.FixedLenFeature((), tf.string),
    #     'feature3': tf.io.FixedLenFeature((), tf.string),
    # }
    # example = tf.io.parse_single_example(serialized_example, feature_description)
    
    # feature0 = example['feature0']
    # feature1 = example['feature1']
    # feature2 = example['feature2']
    # feature3 = tf.io.parse_tensor(example['feature3'], out_type = tf.float64)
  
    # return feature0, feature1, feature2, feature3

def my_filter(*data):
    return tf.greater(data[0], 0)
    return tf.math.equal(data[0], 0)
    return False

def read(file_path):
    file_paths = [file_path] # We have only one file
    tfrecord_dataset = tf.data.TFRecordDataset(file_paths)   
    
    parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    # parsed_dataset = parsed_dataset.filter(my_filter)
    
    # 这么写就是为了打印值
    # sparse_tensor = sess.run(x, feed_dict={
    #     x: tf.SparseTensorValue(indices, values, dense_shape)})
    # print('tensor', sparse_tensor)
    # tensor_value = tf.sparse_tensor_to_dense(sparse_tensor)
    # print('tensor表示的稀疏矩阵:\n', sess.run(tensor_value))


    epochs = 1
    buffer_size = 100
    batch_size = 10

    parsed_dataset = parsed_dataset.repeat(epochs).shuffle(buffer_size).batch(batch_size)
    # parsed_dataset.filter(my_filter)
    # whole_dataset_tensors = tf.contrib.data.get_single_element(parsed_dataset)
    # g=tf.Graph()
    # # Create a session and evaluate `whole_dataset_tensors` to get arrays.
    # with tf.Session(graph=g) as sess:
    #     whole_dataset_arrays = sess.run(whole_dataset_tensors)
    #     print(whole_dataset_arrays)
    #     break
    for data in parsed_dataset:
        # print(data[0][0][0])
        print(data[0])
        print(data[1])
        print(data[2])
        print(data[3])
        # print(data[2].shape)
        break

def case_get_info_by_id():
    path = 'data/case/case.txt'
    with open(path) as f:
        lines = f.readlines()
        # print(lines)
        # line_idx = 0
        for line in lines:
            print(line)
            # if line_idx % 2 == 0:
            #     pass
            # elif line_idx % 2 == 1:
            #     see = [float(s) for s in line.split('\t')[1:]]
            #     print(see, len(see))
            #     datas.append(see)
            # line_idx += 1


if __name__ == "__main__":
    # generate_tfrecords()
    # gen_tfrecord()
    # generate_json()
    # generate_tfrecords_casev2()
    # generate_tfrecords_case()
    # read('/share/lab502/zy/tfrecord/case_data_0.tfrecord')
    # read('tfrecord/case_data_0.tfrecord')
    case_get_info_by_id()