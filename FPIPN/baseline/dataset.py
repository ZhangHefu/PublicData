import os
import torch
import numpy as np
import scipy.sparse as sp 
from torch.utils.data import Dataset
import yaml
from datetime import datetime
import json
import os
import random
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
import tqdm
import torch
from  torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
# random.seed(12345)
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch_geometric.data import Data
def get_paper_feat(paper_id, doc_type, pub_name, venue, abstract_embed, title_embed, doc_embed, pub_embed, venue_embed):
    # 289
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

def gen_dataset():
    train_data = pickle.load( open( "data/sample/train_data.pkl", "rb" ))
    new_adj = []
    set_id = set()
    for data in tqdm.tqdm(train_data):
        # a-a a-p

        for auths in data['author_neighbours']:
            aid = 'a' + auths[0]
            new_adj.extend([(aid, 'p'+p) for p in auths[1][0]])
            new_adj.extend([(aid, 'a'+a) for a in auths[1][1]])

            new_adj.extend([('p'+p, aid) for p in auths[1][0]])
            new_adj.extend([('a'+a, aid) for a in auths[1][1]])
            set_id |= set(['p'+p for p in auths[1][0]])
            set_id |= set(['a'+a for a in auths[1][1]])
            set_id.add(aid)
            
        # t-a t-p
        for topics in data['topic_neighbours']:
            # print(topics[1])
            t_name = topics[0]
            new_adj.extend([('p'+p, t_name) for p in topics[1]])
            new_adj.extend([('a'+a, t_name) for a in topics[2]])

            new_adj.extend([(t_name, 'p'+p) for p in topics[1]])
            new_adj.extend([(t_name, 'a'+a) for a in topics[2]])
            set_id |= set(['p'+p for p in topics[1]])
            set_id |= set(['a'+a for a in topics[2]])
            set_id.add(t_name)

    new_adj = np.array(list(set(new_adj))).T.tolist() # 转置去重复
    
    # map  id/name -> num
    print('edge len:', len(new_adj[0]))
    id_map = {j:int(i+1) for i,j in enumerate(set_id)}
    for i in range(len(new_adj)):
        for j in range(len(new_adj[0])):
            new_adj[i][j] = id_map[new_adj[i][j]]
        
    pickle.dump(id_map, open("graph/id_map.pkl", "wb" ))
    pickle.dump(new_adj, open("graph/sub_adj.pkl", "wb" ))





def generate_feat():

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



    train_au, train_topic, train_paper = set(), set(), set()
    for data in tqdm.tqdm(train_data):
        # a-a a-p
        for auths in data['author_neighbours']:
            aid = auths[0]
            train_paper |= set([p for p in auths[1][0]])
            train_au |= set([a for a in auths[1][1]])
            train_au.add(aid)
        # t-a t-p
        for topics in data['topic_neighbours']:
            t_name = topics[0]
            train_paper |= set([p for p in topics[1]])
            train_au |= set([a for a in topics[2]])
            train_topic.add(t_name)

    topic_dim = 128
    au_dim = 33
    paper_dim = 289
    topic_emb = np.zeros((len(train_topic), topic_dim))
    au_emb = np.zeros((len(train_au), au_dim))
    paper_emb = np.zeros((len(train_paper), paper_dim))
    for idx, t_name in enumerate(train_topic):
        topic_emb[idx] = get_topic_feat(t_name, topic_embed)
    
    for idx, aid in enumerate(train_au):
        au_emb[idx] = get_auth_feat(aid, org_embed, auth_feats)

    for idx, pid in enumerate(train_paper):
        paper_emb[idx] = get_paper_feat(pid, paper_info[pid]['doc_type'], paper_info[pid]['publisher'], 
                            paper_info[pid]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
    
    # 保存
    pickle.dump(train_topic, open("graph/train_topic.pkl", "wb" ))
    pickle.dump(train_au, open("graph/train_au.pkl", "wb" ))
    pickle.dump(train_paper, open("graph/train_paper.pkl", "wb" ))
    
    torch.save(torch.Tensor(topic_emb), 'graph/topic_emb.pt')
    torch.save(torch.Tensor(au_emb), 'graph/au_emb.pt')
    torch.save(torch.Tensor(paper_emb), 'graph/paper_emb.pt')
    print(topic_emb.shape)
    print(au_emb.shape)
    print(paper_emb.shape)

def generate_pt():
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


    max_auth = 6 # 每篇文章最大个数
    author_feat_len = 37
    paper_feat_len = 128 + 128 + 16*2 + 1
    max_refs = 25 # 参考文献最大

    data_dict = {}
    for data in tqdm.tqdm(train_data):
        train_au, train_topic, train_paper = set(), set(), set()
        if references_3_map[data['paper_id']] == 0:
            continue
        # a-a a-p
        for auths in data['author_neighbours']:
            aid = auths[0]
            train_paper |= set([p for p in auths[1][0]])
            train_au |= set([a for a in auths[1][1]])
            train_au.add(aid)
        # t-a t-p
        for topics in data['topic_neighbours']:
            t_name = topics[0]
            train_paper |= set([p for p in topics[1]])
            train_au |= set([a for a in topics[2]])
            train_topic.add(t_name)
        data_dict[data['paper_id']] = {
            'year': data['year'],
            'train_paper': train_paper,
            'train_au': train_au,
            'train_topic': train_topic
        }
    
    new_feats = {}
    for pid in data_dict:
        year = data_dict[pid]['year']
        new_feats[pid] = {}
        for idx in data_dict[pid]['train_paper']:
            new_feats[pid]['p'+idx] = get_paper_feat(idx, paper_info[idx]['doc_type'], paper_info[idx]['publisher'], 
                            paper_info[idx]['venue'], abstract_embed, title_embed, doc_embed, pub_embed, venue_embed)
        for aid in data_dict[pid]['train_au']:
            new_feats[pid]['a'+aid] = get_auth_feat(aid, year, org_embed, auth_feats)
        for name in data_dict[pid]['train_topic']:
            new_feats[pid][name] = get_topic_feat(name, year, topic_embed, topic_all_feat)
    pickle.dump(new_feats, open("graph/new_feats.pkl", "wb" ))
    # torch.save(new_feats, 'graph/new_feats.pt')

class train_dataset(Dataset):
    def __init__(self, max_author, max_topic, cite_year, task):
        """
        推荐任务数据集
        label_file: 数据的json文件
        """
        self.data_dict = {}
        self.raw_data = pickle.load(open("data/sample/train_data.pkl", "rb" ))
        with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
            self.references_3_map = json.load(fin) 
        self.id_map = pickle.load(open("graph/id_map.pkl", "rb" ))
        self.max_author = max_author
        self.max_topic = max_topic
        self.cite_year = cite_year
        data = self.prepross(self.raw_data, self.cite_year)
        print('data_len:', len(data))
        print('loading feats ...')
        self.load_feats()

        self.num_nodes = len(self.id_map)
        self.test_data = data[int(len(data)*0.7):]
        self.train_data = data[:int(len(data)*0.7)]
        self.task = task


    def switch_year(self, year):
        """
        根据year 选择数据的label
        """
        self.data_dict = {}
        data = self.prepross(self.raw_data, year)
        print('data_len:', len(data))
        self.test_data = data[int(len(data)*0.7):]
        self.train_data = data[:int(len(data)*0.7)]


    def load_feats(self):
        """
        加载特征数据
        加载并拼接特征（论文 作者 topic)

        """
        self.new_feats = pickle.load(open("graph/new_feats.pkl", "rb" ))
        self.topic_emb = torch.load('graph/topic_emb.pt') # 
        self.au_emb = torch.load('graph/au_emb.pt')
        self.paper_emb = torch.load('graph/paper_emb.pt')
        self.train_topic = pickle.load( open( "graph/train_topic.pkl", "rb" ))
        self.train_au = pickle.load( open( "graph/train_au.pkl", "rb" ))
        self.train_paper = pickle.load( open( "graph/train_paper.pkl", "rb" ))
        
        self.topic_id = [self.id_map.get(t, 0) for t in self.train_topic]
        self.au_id = [self.id_map.get('a'+aid, 0) for aid in self.train_au]
        self.paper_id = [self.id_map.get('p'+pid, 0) for pid in self.train_paper]

    def prepross(self, raw_data, year):
        new_data = []
        for data in tqdm.tqdm(raw_data):
            paper_id = data['paper_id']
            if self.references_3_map[paper_id]['n_citation{}'.format(year)] == 0:
                continue
            # 获得未来n年的引用 （target）
            data['n_citation'] = self.references_3_map[paper_id]['n_citation{}'.format(year)]
            new_data.append(data)
            self.data_dict[paper_id] = {
                'year': data['year'],
                'author_neighbours': data['author_neighbours'],
                'topic_neighbours': data['topic_neighbours']
            }
        return new_data

    def build_sub_graph(self, paper_id):
        """
        根据author和topic建立一个子图
        返回一个data（graph
        """
        new_adj = []
        set_id, paper_set, author_set, topic_set = set(), set(), set(), set()

        one_data = self.data_dict[paper_id]
        for auths in one_data['author_neighbours']:
            aid = 'a' + auths[0]
            new_adj.extend([(aid, 'p'+p) for p in auths[1][0]])
            new_adj.extend([(aid, 'a'+a) for a in auths[1][1]])

            new_adj.extend([('p'+p, aid) for p in auths[1][0]])
            new_adj.extend([('a'+a, aid) for a in auths[1][1]])
            set_id |= set(['p'+p for p in auths[1][0]])
            set_id |= set(['a'+a for a in auths[1][1]])
            set_id.add(aid)

            paper_set |= set(['p'+p for p in auths[1][0]])
            author_set |= set(['a'+a for a in auths[1][1]])
            author_set.add(aid)
            
        # t-a t-p
        for topics in one_data['topic_neighbours']:
            # print(topics[1])
            t_name = topics[0]
            new_adj.extend([('p'+p, t_name) for p in topics[1]])
            new_adj.extend([('a'+a, t_name) for a in topics[2]])

            new_adj.extend([(t_name, 'p'+p) for p in topics[1]])
            new_adj.extend([(t_name, 'a'+a) for a in topics[2]])
            set_id |= set(['p'+p for p in topics[1]])
            set_id |= set(['a'+a for a in topics[2]])
            set_id.add(t_name)
            
            paper_set |= set(['p'+p for p in topics[1]])
            author_set |= set(['a'+a for a in topics[2]])
            topic_set.add(t_name)

        
        set_id = list(set_id)
        new_adj = np.array(list(set(new_adj))).T.tolist() # 转置去重复
        
        # map  id/name -> num
        id_map = {j:int(i+1) for i,j in enumerate(set_id)}
        for i in range(len(new_adj)):
            for j in range(len(new_adj[0])):
                new_adj[i][j] = id_map[new_adj[i][j]]
        
        edge_index = torch.LongTensor(new_adj)
        # 返回相应特征
        paper_set = list(paper_set)
        author_set = list(author_set)
        topic_set = list(topic_set)

        topic_dim = 130
        au_dim = 37
        paper_dim = 289
        topic_emb = np.zeros((len(topic_set), topic_dim))
        au_emb = np.zeros((len(author_set), au_dim))
        paper_emb = np.zeros((len(paper_set), paper_dim))
        for i, idx in enumerate(topic_set):
            topic_emb[i] = self.new_feats[paper_id][idx]
        for i, idx in enumerate(author_set):
            au_emb[i] = self.new_feats[paper_id][idx]
        for i, idx in enumerate(paper_set):
            paper_emb[i] = self.new_feats[paper_id][idx]
        paper_set = [id_map[idx] for idx in paper_set]
        author_set = [id_map[idx] for idx in author_set]
        topic_set = [id_map[idx] for idx in topic_set]

        # data_graph = Data(x=x, edge_index=edge_index)
        return id_map, edge_index, paper_set, author_set, topic_set, torch.Tensor(topic_emb), torch.Tensor(au_emb), torch.Tensor(paper_emb)
        
    def __getitem__(self, idx):
        """
        """
        if self.task == 'train':
            self.data = self.train_data
        elif self.task == 'test':
            self.data = self.test_data
        id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb = self.build_sub_graph(self.data[idx]['paper_id'])
        authors = [id_map['a'+a[0]] for a in self.data[idx]['author_neighbours'][:self.max_author]]
        authors_np = np.array(authors)

        topics = [id_map[t[0]] for t in self.data[idx]['topic_neighbours'][:self.max_topic]]
        topics_np = np.array(topics)

        auth_cnt, topic_cnt = len(authors), len(topics)
        target = self.data[idx]['n_citation']+1

        # label = self.cite_to_label(target)
        return authors_np, topics_np, target, int(auth_cnt), int(topic_cnt), self.data[idx]['paper_id'], id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb

    def __len__(self):

        if self.task == 'train':
            return len(self.train_data)
        elif self.task == 'test':
            return len(self.test_data)

    def clear(self):

        self.data = []

    def switch_task(self, task):
        
        if self.task == task:
            return
        # print('{} ===> {}'.format(self.task, task))
        self.task = task


class train_dataset_deepwalk(Dataset):
    def __init__(self, max_author, max_topic, cite_year, task):
        """
        推荐任务数据集
        label_file: 数据的json文件
        """
        self.raw_data = pickle.load(open("data/sample/train_data.pkl", "rb" ))
        with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
            self.references_3_map = json.load(fin) 
        self.id_map = pickle.load(open("graph/id_map.pkl", "rb" ))
        self.max_author = max_author
        self.max_topic = max_topic
        self.cite_year = cite_year
        data = self.prepross(self.raw_data, self.cite_year)
        print('data_len:', len(data))
        print('loading feats ...')
        self.load_feats()

        self.num_nodes = len(self.id_map)
        self.test_data = data[int(len(data)*0.7):]
        self.train_data = data[:int(len(data)*0.7)]
        self.task = task


    def switch_year(self, year):
        """
        根据year 选择数据的label
        """
        data = self.prepross(self.raw_data, year)
        print('data_len:', len(data))
        self.test_data = data[int(len(data)*0.7):]
        self.train_data = data[:int(len(data)*0.7)]


    def load_feats(self):
        """
        加载特征数据
        加载并拼接特征（论文 作者 topic)
        """
        self.node2id = pickle.load(open("data/graph/node2id.pkl", "rb" )) # node: id
        self.node_emb = np.fromfile('data/embedding/deepwalk.emb', np.float32)
        self.node_emb = self.node_emb.reshape(int(self.node_emb.shape[0] / 128), 128)
        print(self.node_emb.shape)
        # self.new_feats = pickle.load(open("graph/new_feats.pkl", "rb" ))
        # self.topic_emb = torch.load('graph/topic_emb.pt') # 
        # self.au_emb = torch.load('graph/au_emb.pt')
        # self.paper_emb = torch.load('graph/paper_emb.pt')
        # self.train_topic = pickle.load( open( "graph/train_topic.pkl", "rb" ))
        # self.train_au = pickle.load( open( "graph/train_au.pkl", "rb" ))
        # self.train_paper = pickle.load( open( "graph/train_paper.pkl", "rb" ))
        
        # self.topic_id = [self.id_map.get(t, 0) for t in self.train_topic]
        # self.au_id = [self.id_map.get('a'+aid, 0) for aid in self.train_au]
        # self.paper_id = [self.id_map.get('p'+pid, 0) for pid in self.train_paper]

    def prepross(self, raw_data, year):
        new_data = []
        for data in tqdm.tqdm(raw_data):
            paper_id = data['paper_id']
            if self.references_3_map[paper_id]['n_citation{}'.format(year)] == 0:
                continue
            # 获得未来n年的引用 （target）
            data['n_citation'] = self.references_3_map[paper_id]['n_citation{}'.format(year)]
            new_data.append(data)
        return new_data
    
    def __getitem__(self, idx):
        """
        """
        if self.task == 'train':
            self.data = self.train_data
        elif self.task == 'test':
            self.data = self.test_data

        authors_np = self.node_emb[[self.node2id[a[0]] for a in self.data[idx]['author_neighbours'][:self.max_author]]]
        topics_np = self.node_emb[[self.node2id[t[0]] for t in self.data[idx]['topic_neighbours'][:self.max_topic]]]
        # id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb = self.build_sub_graph(self.data[idx]['paper_id'])
        # authors = [id_map['a'+a[0]] for a in self.data[idx]['author_neighbours'][:self.max_author]]
        # authors_np = np.array(authors)

        # topics = [id_map[t[0]] for t in self.data[idx]['topic_neighbours'][:self.max_topic]]
        # topics_np = np.array(topics)
        auth_cnt, topic_cnt = len(authors_np), len(topics_np)
        target = self.data[idx]['n_citation']+1

        # label = self.cite_to_label(target)
        return torch.Tensor(authors_np), torch.Tensor(topics_np), target, int(auth_cnt), int(topic_cnt), self.data[idx]['paper_id']

    def __len__(self):

        if self.task == 'train':
            return len(self.train_data)
        elif self.task == 'test':
            return len(self.test_data)

    def clear(self):

        self.data = []

    def switch_task(self, task):
        
        if self.task == task:
            return
        # print('{} ===> {}'.format(self.task, task))
        self.task = task




class CustomBatch:

    def __init__(self):
        pass

    def process(self, batch):
        # print(batch)
        # return 
        # batchsize == len(data)

        # authors_np, topics_np, target, auth_cnt, topic_cnt, label, paper_ids,  id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb = batch
        # authors_np, topics_np, target, auth_cnt, topic_cnt, label, paper_ids,  id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb
        
        res = []
        for i in range(len(batch[0])):
            res.append([data[i] for data in batch])
        return res
        # for idx, data in enumerate(batch):
        #     for d in data[0]
        
        # batch_queue_ids, batch_cpu_usage, batch_launch_jobs = [], [], []
        # input_mask = np.zeros([len(batch), max_seq])
        # for idx, data in enumerate(batch):
        #     assert len(data[0]) == len(data[1])# , data[2]
        #     seq_length = len(data[0])
        #     assert seq_length == self.max_seq_len
        #     # 序列mask
        #     input_mask[idx, :seq_length] = 1
        #     # pad处理数据
        #     queue_ids, cpu_usage, launch_jobs = data[0], data[1], data[2]
        #     queue_ids = np.pad(queue_ids, (0, max_seq - len(queue_ids)))
        #     cpu_usage = np.pad(cpu_usage, (0, max_seq - len(cpu_usage)))
        #     launch_jobs = np.pad(launch_jobs, (0, max_seq - len(launch_jobs)))
        #     batch_queue_ids.append(queue_ids)
        #     batch_cpu_usage.append(cpu_usage)
        #     batch_launch_jobs.append(launch_jobs)
            
        # return torch.LongTensor(batch_queue_ids), torch.FloatTensor(batch_cpu_usage), torch.FloatTensor(batch_launch_jobs)

    def pad_vec(self, vec, size_to_pad, dim):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad
        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.shape)
        pad_size[dim] = size_to_pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    
    def __call__(self, batch):
        return self.process(batch)


if __name__ == '__main__':
    # generate_pt()
    # generate_feat()
    # gen_dataset()

    # dataset = train_dataset(max_author=6, max_topic=12, task='train')
    # sub_adj = pickle.load(open("graph/sub_adj.pkl", "rb" ))
    # edge_index = torch.LongTensor(sub_adj)

    # x = torch.zeros(dataset.num_nodes+1, 32)
    # data = Data(x=x, edge_index=edge_index)
    # print(data.is_undirected())

    # dataset = train_dataset(max_author=6, max_topic=12, task='train')
    dataset = train_dataset_deepwalk(max_author=6, max_topic=12,cite_year=3, task='train')
    loader = DataLoader(dataset, batch_size=1, num_workers=5, collate_fn=CustomBatch())
    print(len(dataset))
    # for author_ids, topic_ids, targets, auth_cnts, topic_cnts, labels, paper_ids,\
    #             id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, \
    #             au_embs, paper_embs in tqdm.tqdm(loader):
    for authors_emb, topics_emb, targets, auth_cnts, topic_cnts, paper_ids in tqdm.tqdm(loader):
        print(targets)
        print(authors_emb)
        # print(authors_nps[0].shape)
        # print(len(author_ids))
        break
