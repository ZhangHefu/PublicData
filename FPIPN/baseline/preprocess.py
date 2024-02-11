import json
import random
import numpy as np
import tqdm
import gensim
min_log = 10
import collections
import pickle
import unicodedata
from collections import defaultdict
# 训练前对各个数据集进行处理
# 根据学生划分出一部分作为测试集合 测试自适应模块
# all = train data + valid data(调参) + test data(用作自适应测试，学生id不能和前面有重合)
import matplotlib.pyplot as plt
def preprocess1():
    """
    we filter the questions that are answered by less than 50 students and the students that answer less than 10 questions.
{"id": "100001334", "title": "Ontologies in HYDRA - Middleware for Ambient Intelligent Devices.", "authors": 
[{"name": "Peter Kostelnik", "id": "2702511795"}, {"name": "Martin Sarnovsky", "id": "2041014688"}, 
{"name": "Jan Hreno", "id": "2398560122"}], "venue": {"raw": "AMIF"}, "year": 2009, "n_citation": 2, "page_start": "43", "page_end": "46", "doc_type": "", "publisher": "", "volume": "", "issue": "", "fos": [{"name": "Lernaean Hydra", "w": 0.4178039}, {"name": "Database", "w": 0.4269269}, {"name": "World Wide Web", "w": 0.415332377}, {"name": "Ontology (information science)", "w": 0.459045082}, {"name": "Computer science", "w": 0.399807781}, {"name": "Middleware", "w": 0.5905041}, {"name": "Ambient intelligence", "w": 0.5440575}]}
    """
    total_file = 42
    all_data = {
        'paper_ids' : set(),
        'authors_ids': set(),
        'venue_names': set(),
        'doc_types': set(),
        'langs': set(),
        'author_orgs': set(),
        'fos_names': set(),
        'publisher_names': set()
    }
    abstract_raw = {}
    title_raw = {}
    cnt = 0
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        paper_ids = all_data['paper_ids']
        authors_ids = all_data['authors_ids']
        venue_names = all_data['venue_names']
        doc_types = all_data['doc_types']
        langs = all_data['langs']
        author_orgs = all_data['author_orgs']
        fos_names = all_data['fos_names']
        publisher_names = all_data['publisher_names']
        
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                one_paper = json.loads(line) 
                if 'id' in one_paper:
                    paper_ids.add(one_paper['id'])
                if 'authors' in one_paper:
                    for auth in one_paper['authors']:
                        if 'org' in auth:    
                            author_orgs.add(auth['org'])
                        authors_ids.add(auth['id'])
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        fos_names.add(fos['name'])
                if 'publisher' in one_paper and one_paper['publisher'] != "":
                    publisher_names.add(one_paper['publisher'])
                # "venue": {"raw": "international conference on trusted systems", "id": "2760518661"}
                # 有些venueid缺省 但是有venue_raw
                if 'venue' in one_paper:
                    if 'raw' in one_paper['venue']:
                        venue_names.add(one_paper['venue']['raw'])
                    else:
                        print(one_paper)
                # 可能是空字符串
                if 'doc_type' in one_paper and one_paper['doc_type'] != '':
                    doc_types.add(one_paper['doc_type'])
                if 'lang' in one_paper:
                    langs.add(one_paper['lang'])
                # 重构indexed_abstract
                if 'indexed_abstract' in one_paper:
                    ab = one_paper['indexed_abstract']
                    word_list = [''] * ab['IndexLength']
                    for w in ab['InvertedIndex']:
                        idxs = ab['InvertedIndex'][w]
                        for idx in idxs:
                            word_list[idx] = w
                    abstract_raw[one_paper['id']] = ' '.join(word_list)
                
                if 'title' in one_paper:
                    title_raw[one_paper['id']] = one_paper['title']

                    
        for name in all_data:
            if "" in all_data[name]:
                print("{} has empty str !!!".format(name))
        print(cnt)
        print("" in authors_ids, "" in venue_names, "" in fos_names)
        print('papers: {}, authors: {}, venue: {}, doc_types: {}, langs: {}, author_orgs: {}, fos_names: {}, publisher_names: {}'.format(
                    len(paper_ids), len(authors_ids), len(venue_names), len(doc_types), 
                    len(langs), len(author_orgs), len(fos_names), len(publisher_names)))

        # break

    data_map = {}
    for k in all_data:
        data_map[k] = {x:i for i, x in enumerate(all_data[k])}
    # 保存向量map文件
    for name in tqdm.tqdm(data_map):
        with open('data/embed/%s_map.json' % name, 'w', encoding='utf8') as fout:
            map_len = len(data_map[name])
            json.dump(data_map[name], fout, ensure_ascii=False)
    
    
    with open('data/embed/abstracts_raw.txt', 'w', encoding='utf8') as fout:
        for ab in abstract_raw:
            fout.write(abstract_raw[ab] + '\n')
    with open('data/embed/titles_raw.txt', 'w', encoding='utf8') as fout:
        for title in title_raw:
            fout.write(title_raw[title] + '\n')


def read_corpus(ab_or_title):
    assert ab_or_title in ['ab', 'title', 'all']
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                # 重构indexed_abstract
                if ab_or_title == 'ab':
                    if 'indexed_abstract' in one_paper:
                        ab = one_paper['indexed_abstract']
                        word_list = [''] * ab['IndexLength']
                        for w in ab['InvertedIndex']:
                            idxs = ab['InvertedIndex'][w]
                            for idx in idxs:
                                word_list[idx] = w
                        abstract_raw = ' '.join(word_list)
                        tokens = gensim.utils.simple_preprocess(abstract_raw)
                        yield gensim.models.doc2vec.TaggedDocument(tokens, ['ab_%s' % paper_id])
                    if 'abstract' in one_paper:
                        print(one_paper)
                elif ab_or_title == 'title':
                    if 'title' in one_paper:
                        title_raw = one_paper['title']
                        tokens = gensim.utils.simple_preprocess(title_raw)
                        yield gensim.models.doc2vec.TaggedDocument(tokens, ['title_%s' % paper_id])
                elif ab_or_title == 'all':
                    if 'indexed_abstract' in one_paper:
                        ab = one_paper['indexed_abstract']
                        word_list = [''] * ab['IndexLength']
                        for w in ab['InvertedIndex']:
                            idxs = ab['InvertedIndex'][w]
                            for idx in idxs:
                                word_list[idx] = w
                        abstract_raw = ' '.join(word_list)
                        tokens = gensim.utils.simple_preprocess(abstract_raw)
                        yield gensim.models.doc2vec.TaggedDocument(tokens, ['ab_%s' % paper_id])
                    if 'abstract' in one_paper:
                        print(one_paper)
                    if 'title' in one_paper:
                        title_raw = one_paper['title']
                        tokens = gensim.utils.simple_preprocess(title_raw)
                        yield gensim.models.doc2vec.TaggedDocument(tokens, ['title_%s' % paper_id])
                    


        # break



def train_model(algo):
    # 训abstract
    train_corpus = list(read_corpus('all'))
    print(train_corpus[:2])
    if algo == "DM":
        model_DM = gensim.models.doc2vec.Doc2Vec(train_corpus, vector_size = 128, window = 10, min_count=1, workers=10, epochs = 10,  dm = 1, negative=10)
        print("Model paragraphs_DM Trained")
        model_DM.save("data/Para2Vec_Models/paragraph_DM_ab_title_128.doc2vec")
        print("Model paragraphs_DM saved")
    elif algo == "DBOW":
        model_DBOW = gensim.models.doc2vec.Doc2Vec(train_corpus, vector_size = 128, window = 10, min_count=1, workers=10, epochs = 10,  dm = 0, negative=10)
        print("Model paragraph_DBOW Trained")
        model_DBOW.save("data/Para2Vec_Models/paragraph_DBOW_ab_title_128.doc2vec")
        print("Model paragraphs_DBOW saved")

    # # title 加上
    # train_corpus = list(read_corpus('all'))

    # if algo == "DM":
    #     fname = "data/Para2Vec_Models/paragraph_"+algo+"_ab.doc2vec"
    #     model_DM = gensim.models.doc2vec.Doc2Vec.load(fname)
    #     print("abs para DM loaded")
    #     # model_DM.train(train_corpus)
    #     model_DM.train(train_corpus, total_examples=len(train_corpus), epochs=model_DM.epochs)
    #     print("abs para DM trained")
    #     model_DM.save("data/Para2Vec_Models/paragraph_DM_ab_title.doc2vec")
    #     print("abs para DM saved")
    # elif algo == "DBOW":
    #     fname = "data/Para2Vec_Models/paragraph_"+algo+"_ab.doc2vec"
    #     model_DBOW = gensim.models.doc2vec.Doc2Vec.load(fname)
    #     print("abs para DBOW loaded")
    #     # model_DBOW.train(train_corpus)
    #     model_DBOW.train(train_corpus, total_examples=len(train_corpus), epochs=model_DBOW.epochs)
    #     print("abs para DBOW trained")
    #     model_DBOW.save("data/Para2Vec_Models/paragraph_DBOW_ab_title.doc2vec")
    #     print("abs para DBOW saved")

def search_by_id(p_id):
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                if p_id == paper_id:
                    if 'indexed_abstract' in one_paper:
                        ab = one_paper['indexed_abstract']
                        word_list = [''] * ab['IndexLength']
                        for w in ab['InvertedIndex']:
                            idxs = ab['InvertedIndex'][w]
                            for idx in idxs:
                                word_list[idx] = w
                        abstract_raw = ' '.join(word_list)
                        one_paper['ab_raw'] = abstract_raw
                        one_paper['indexed_abstract'] = - 1
                        return one_paper
                    return 


                    # if 'indexed_abstract' in one_paper:
                    #     ab = one_paper['indexed_abstract']
                    #     word_list = [''] * ab['IndexLength']
                    #     for w in ab['InvertedIndex']:
                    #         idxs = ab['InvertedIndex'][w]
                    #         for idx in idxs:
                    #             word_list[idx] = w
                    #     abstract_raw = ' '.join(word_list)
                    #     one_paper['ab_raw'] = abstract_raw
                    #     one_paper['indexed_abstract'] = - 1
                    #     return one_paper


def assess_model(doc_str, fname, ab_or_title):

    assert ab_or_title in ['ab', 'title']
    
    model = gensim.models.doc2vec.Doc2Vec.load(fname)
    tokens = gensim.utils.simple_preprocess(doc_str)
    inferred_vector = model.infer_vector(tokens)
    sims = model.docvecs.most_similar([inferred_vector], topn=10)
    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document: «{}»\n'.format(doc_str))
    print('SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        tag = sims[index][0]
        print(label, tag)
        if ab_or_title == 'ab':
            if 'ab_' in tag:
                print(search_by_id(tag[3:]))
        elif ab_or_title == 'title':
            if 'title_' in tag:
                print(search_by_id(tag[6:]))

def gen_vec_file(fname, ab_or_title):
    assert ab_or_title in ['ab', 'title']
    vec_map = {}
    model = gensim.models.doc2vec.Doc2Vec.load(fname)
    for doc_tag in tqdm.tqdm(model.docvecs.index2entity):
        doc_vec = model.docvecs[doc_tag]
        if ab_or_title == 'ab':
            if 'ab_' in doc_tag:
                vec_map[doc_tag[3:]] = doc_vec
        elif ab_or_title == 'title':
            if 'title_' in doc_tag:
                vec_map[doc_tag[6:]] = doc_vec

    print(list(vec_map.keys())[:2])
    if model.dm:
        model_type = 'DM'
    else:
        model_type = 'DBOW'
    if ab_or_title == 'ab':
        print('abstract: ', len(vec_map))
        pickle.dump(vec_map, open("data/Para2Vec_Models/p_abstract_embed_{}_{}.pkl".format(model_type, model.vector_size), "wb" ))
    elif ab_or_title == 'title':
        print('title: ', len(vec_map))
        pickle.dump(vec_map, open("data/Para2Vec_Models/p_title_embed_{}_{}.pkl".format(model_type, model.vector_size), "wb" ))
        

def re_cal_cite():
    """
    重新计算3年内应用量
    """
    total_file = 42
    year_offset = 3 # 最近几年
    data = {}
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                year = one_paper.get('year', 0)
                if year == 0:
                    print(one_paper)
                references = one_paper.get('references', [])
                data[paper_id] = {'year':year, 'references':references}
        # break

    new_json = {}
    for paper_id in data:
        # year缺失
        if data[paper_id]['year'] == 0:
            new_json[paper_id] = {'cited_by0':[], 'cited_by1':[], 'cited_by2':[], 'cited_by3':[]}
            continue
        for ref in data[paper_id]['references']:
            if data[ref]['year'] == 0:
                continue
            delt_time = data[paper_id]['year'] - data[ref]['year']
            if 0 <= delt_time <= year_offset:
                if ref not in new_json:
                    new_json[ref] = {'cited_by0':[], 'cited_by1':[], 'cited_by2':[], 'cited_by3':[]}
                # 相应的年份都要保存
                for t in range(delt_time, year_offset+1):
                    new_json[ref]['cited_by%s'% t].append(paper_id)
            
    for paper_id in data:
        # 不在new_data的id也要添加
        if paper_id not in new_json:
            new_json[paper_id] = {'cited_by0':[], 'cited_by1':[], 'cited_by2':[], 'cited_by3':[]}
        for t in range(year_offset+1):
            new_json[paper_id]['n_citation%s'% t] = len(new_json[paper_id]['cited_by%s'% t])
        # new_json[paper_id]['n_citation'] = len(new_json[paper_id]['cited_by'])
        if data[paper_id]['year'] != 0:
            new_json[paper_id]['year'] = data[paper_id]['year']

        # new_ref = []
        # if data[paper_id]['year'] == 0:
        #     new_json[paper_id] = {'references':[], 'n_citation':0}
        #     continue
        # for ref in data[paper_id]['references']:
        #     if data[ref]['year'] == 0:
        #         continue
        #     if 0 <= data[ref]['year'] - data[paper_id]['year'] <= year_offset:
        #         new_ref.append(ref)
        # new_json[paper_id] = {'references':new_ref, 'n_citation':len(new_ref)}
    
    with open('data/embed/%s_map.json' % 'references_3', 'w', encoding='utf8') as fout:
        json.dump(new_json, fout, ensure_ascii=False)
    

def get_topic_vector(fname):
    model = gensim.models.doc2vec.Doc2Vec.load(fname)
    vec_map = {}
    total_file = 42
    """
    "fos": [{"name": "Multi-swarm optimization", "w": 0.653890431}, 
    {"name": "Genetic algorithm", "w": 0.654717743}, 
    {"name": "Evolutionary algorithm", "w": 0.663817763},
     {"name": "Meta-optimization", "w": 0.65959996}, 
     {"name": "Metaheuristic", "w": 0.653632}, {"name": "Mathematical optimization", "w": 0.465393662}, {"name": "Evolutionary programming", "w": 0.650162756}, {"name": "Imperialist competitive algorithm", "w": 0.625040531}, {"name": "Memetic algorithm", "w": 0.6148229}, {"name": "Computer science", "w": 0.4417319}]}
    """
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        if fos['name'] in vec_map:
                            continue
                        tokens = gensim.utils.simple_preprocess(fos['name'])
                        topic_embed = np.zeros(model.vector_size)
                        cnt = 0
                        for tok in tokens:
                            if tok in model.wv:
                                topic_embed += model.wv.get_vector(tok)  
                                cnt += 1
                            else:
                                pass
                                # print(fos['name']) 
                        if cnt != 0:
                            topic_embed /= cnt
                            vec_map[fos['name']] = topic_embed
    if model.dm:
        model_type = 'DM'
    else:
        model_type = 'DBOW'
    pickle.dump(vec_map, open("data/embed/topic_embed_{}_{}.pkl".format(model_type, model.vector_size), "wb" ))
        
    
                                                
def get_author_feat():
    # 只考虑近20年 2000~2019
    total_file = 42
    abstract_raw = {}
    title_raw = {}
    cnt = 0
    """
    {2009: 204488, 2013: 248384, 2012: 237343, 2008: 183787, 2006: 164598, 2010: 209706, 1991: 25941, 2011: 226693, 2004: 123467, 1998: 55609, 2015: 260990, 1993: 34385, 1994: 38895, 2001: 76090, 2016: 274032, 2014: 258347, 1984: 9842, 1969: 1628, 1987: 13975, 1995: 40339, 1990: 22444, 1997: 48848, 2000: 70892, 1981: 6734, 1985: 10596, 2007: 173268, 1989: 18956, 1999: 61409, 1983: 8255, 1986: 12594, 2003: 102332, 2002: 85562, 2005: 144006, 1992: 28290, 1978: 5044, 1977: 4714, 1959: 417, 
    1968: 1710, 1988: 16530, 2018: 218821, 1973: 3186, 1982: 7249, 1974: 3632, 1980: 6023, 1996: 44621, 1972: 2594, 1979: 5339, 1962: 889, 1967: 1331, 1975: 3761, 2017: 282528, 1976: 4377, 1971: 2288, 1960: 375, 1946: 26, 1970: 1661, 1961: 585, 1963: 722, 1966: 1033, 1964: 739, 1965: 879, 1937: 13, 2019: 2255, 1936: 11, 1958: 249, 1956: 194, 1957: 217, 1953: 101, 1935: 1, 1952: 33, 1938: 10, 1948: 12, 1947: 9, 1954: 151, 1949: 20, 1800: 1, 1955: 141, 1944: 5, 1941: 10, 1951: 18, 1945: 7, 
    1939: 18, 1950: 28, 1942: 13, 1943: 8, 1940: 9, 1899: 5}
    """
    # 去掉了1800 和 1899
    year_list = [1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 
    1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 
    1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
    1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    # author_map = {
    #     'author_id':{
    #         'author_org':'ustc',
    #         'papers':{[], }, # 文章id
    #         'n_papers':[0,0,1..], # 累计统计
    #         'n_citation': [0,0,2,4...],
    #         'partners':[[431453,] ]# 合作者id 去重
    #         'n_partners': []
    #         'topic':[] # topic列表 字符串 
    #         'n_topic': [] # 累计
    #     }
    author_map = {}

    
    year_set = {}
    year_s = 2000
    last_year = 20
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                year = one_paper.get('year', 0)
                n_citation = one_paper.get('n_citation', 0)
                fos_set = set()
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        fos_set.add(fos['name'])
                    
                if 'authors' in one_paper:
                    auth_all = set()
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.add(auth_id)

                        if auth_id not in author_map:
                            author_map[auth_id] = {}
                            author_map[auth_id]['papers'] = [[] for i in range(last_year)]
                            author_map[auth_id]['n_papers'] = [0] * 20
                            author_map[auth_id]['n_citation'] = [0] * 20
                            author_map[auth_id]['partners'] = [set() for i in range(last_year)]
                            author_map[auth_id]['n_partners'] = [0] * 20
                            author_map[auth_id]['topic'] = [set() for i in range(last_year)]
                            author_map[auth_id]['n_topic'] = [0] * 20
                            author_map[auth_id]['author_org_list'] = [set() for i in range(last_year)]
                            author_map[auth_id]['first_paper_time'] = 9999
                        # 统计第一次发文时间
                        if year != 0 and year < author_map[auth_id].get('first_paper_time', 9999):
                            author_map[auth_id]['first_paper_time'] = year

                        if 'org' in auth:  
                            author_map[auth_id]['author_org'] = auth['org']
                            
                    
                        if year < year_s:
                            continue
                        author_map[auth_id]['papers'][year - year_s].append(one_paper['id'])
                        author_map[auth_id]['n_papers'][year - year_s] += 1
                        author_map[auth_id]['n_citation'][year - year_s] += n_citation
                        author_map[auth_id]['topic'][year - year_s] |= fos_set
                        author_map[auth_id]['n_topic'][year - year_s] = len(author_map[auth_id]['topic'][year - year_s])
                        if 'org' in auth:  
                            author_map[auth_id]['author_org_list'][year - year_s].add(auth['org'])

                    if year < year_s:
                        continue
                    for auth in auth_all:
                        author_map[auth]['partners'][year - year_s] |= auth_all
                        author_map[auth]['partners'][year - year_s].remove(auth)
                        author_map[auth]['n_partners'][year - year_s] = len(author_map[auth]['partners'][year - year_s])
            # break
    
    # 统计累计
    for auth_id in author_map:
        auth = author_map[auth_id]
        auth['n_papers'] = np.array(auth['n_papers']).cumsum().tolist()
        auth['n_citation'] = np.array(auth['n_citation']).cumsum().tolist()
        auth['n_partners'] = np.array(auth['n_partners']).cumsum().tolist()
        auth['n_topic'] = np.array(auth['n_topic']).cumsum().tolist()
        for k in range(len(auth['topic'])):
            auth['topic'][k] = list(auth['topic'][k])
            auth['partners'][k] = list(auth['partners'][k])
            auth['author_org_list'][k] = list(auth['author_org_list'][k])
            
    # with open('data/embed/%s_map.json' % 'authors_feat', 'w', encoding='utf8') as fout:
    #     json.dump(author_map, fout, indent=4, ensure_ascii=False)   
    pickle.dump(author_map, open('data/embed/%s_map.pkl' % 'authors_feat', "wb" ))         


def cal_fos():
    year_s = 2000
    last_year = 20
    fos_cnt = {}
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                year = one_paper.get('year', 0)
                if year < year_s:
                    continue
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        fos_name = fos['name']
                        if fos_name not in fos_cnt:
                            fos_cnt[fos_name] = {}
                            fos_cnt[fos_name]['weights_by_year'] = [[] for i in range(last_year)]
                            fos_cnt[fos_name]['npaper_by_year'] = [0] * last_year
                        fos_cnt[fos_name]['weights_by_year'][year - year_s].append(fos['w'])
                        fos_cnt[fos_name]['npaper_by_year'][year - year_s] += 1


    for fos_name in fos_cnt:
        fos_cnt[fos_name]['n_paper'] = sum(fos_cnt[fos_name]['npaper_by_year'])
        for idx in range(last_year):
            if len(fos_cnt[fos_name]['weights_by_year'][idx]) == 0:
                fos_cnt[fos_name]['weights_by_year'][idx] = 0
            else:
                fos_cnt[fos_name]['weights_by_year'][idx] = sum(fos_cnt[fos_name]['weights_by_year'][idx]) / len(fos_cnt[fos_name]['weights_by_year'][idx])
    with open('data/embed/fos_feat_map.json', 'w', encoding='utf8') as fout:
        json.dump(fos_cnt, fout, ensure_ascii=False)    

def hist_info_author():
    n_paper = []
    cnt = 0
    with open('data/embed/%s_map.json' % 'authors_feat', 'r', encoding='utf8') as fin:
        author_map = json.load(fin)  
    # for a_id in tqdm.tqdm(author_map):
    #     n_paper.append(author_map[a_id]['n_papers'][-1])
    
    for a_id in tqdm.tqdm(author_map):
        if author_map[a_id]['n_papers'][-1] > 1 and author_map[a_id]['first_paper_time']>2009 and author_map[a_id]['first_paper_time']!=9999:
            cnt+=1
    print(cnt)

        # first_paper_time
        # n_paper.append(author_map[a_id]['n_papers'][-1])
    
    
    
    # plt.hist(x = n_paper, # 指定绘图数据
    #     bins = 20, # 指定直方图中条块的个数
    #     color = 'steelblue', # 指定直方图的填充色
    #     edgecolor = 'black' # 指定直方图的边框色
    # )
    # plt.savefig('data/image/n_paper.png')

def hist_info_topic():
    n_paper = []
    with open('data/embed/%s_map.json' % 'fos_feat', 'r', encoding='utf8') as fin:
        fos_feat = json.load(fin)  
    for fos_name in tqdm.tqdm(fos_feat):
        n_paper.append(fos_feat[fos_name]['n_paper'])
    data = {'idx': range(len(n_paper)), 'value':n_paper}
    import pandas as pd
    df = pd.Dateframe(data)

    
    plt.hist(x = n_paper, # 指定绘图数据
        bins = 20, # 指定直方图中条块的个数
        color = 'steelblue', # 指定直方图的填充色
        edgecolor = 'black' # 指定直方图的边框色
    )
    plt.savefig('data/image/n_paper.png')

def sample_author():
    # 用元组 每年作者保存最高引用的10篇
    
    year_s = 2000
    last_year = 20
    fos_cnt = {}
    total_file = 42
    paper_cite = {}
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                year = one_paper.get('year', 0)
                paper_cite[one_paper['id']] = {'n_citation': one_paper.get('n_citation', 0), 'year':year}    
        
    with open('data/embed/%s_map.json' % 'authors_feat', 'r', encoding='utf8') as fin:
        author_map = json.load(fin)  
    author_sample = {}
    author_p = {} 
    top_k = 10
    for year in tqdm.tqdm(range(year_s, year_s+last_year)):
        for a_id in tqdm.tqdm(author_map):
            # 过滤
            if author_map[a_id]['n_papers'][-1] <= 1:
                continue

            if a_id not in author_sample:
                author_sample[a_id] = [[] for i in range(last_year)]
                author_p[a_id] = []
            
            for p_id in author_map[a_id]['papers'][year-year_s]:
                n_cite = paper_cite[p_id]['n_citation']
                author_p[a_id].append((p_id, n_cite))
            author_p[a_id] = sorted(author_p[a_id], key=lambda x: x[1], reverse=True)
                # insert_idx = 0
                # for idx in range(len(author_p[a_id])):
                #     if n_cite >= author_p[a_id][idx][1]:
                #         insert_idx = idx
                #         break
                # author_p[a_id].insert(insert_idx, (p_id, n_cite))

            author_sample[a_id][year - year_s] = [p_id for p_id, _ in author_p[a_id][:top_k]]

    with open('data/sample/author_sample.json', 'w', encoding='utf8') as fout:
        json.dump(author_sample, fout, ensure_ascii=False)    

def sample_topic():
    
    # assert mode in ['weight', 'avg']
    year_s = 2000
    last_year = 20
    total_file = 42
    paper_cite = {}
    topic_sample = {}
    top_k = 10
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                year = one_paper.get('year', 0)
                n_cite = one_paper.get('n_citation', 0)
                paper_id = one_paper['id']
                if year < year_s:
                    continue
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        fos_name = fos['name']
                        if fos_name not in topic_sample:
                            topic_sample[fos_name] = [[] for i in range(last_year)]
                        topic_sample[fos_name][year-year_s].append((paper_id, n_cite))
                        # # 维持topk
                        # insert_idx = 0
                        
                        # for idx in range(len(topic_sample[fos_name]['top_k'][year-year_s])):
                        #     if n_cite >= topic_sample[fos_name]['top_k'][year-year_s][idx][1]:
                        #         insert_idx = idx
                        #         break
                        
                        # topic_sample[fos_name]['top_k'][year-year_s].insert(insert_idx, (paper_id, n_cite))  
                        # topic_sample[fos_name]['top_k'][year-year_s] = topic_sample[fos_name]['top_k'][year-year_s][:top_k]
    topic_weight = {}
    topic_random = {}
    for t_name in topic_sample:
        # topic_weight[t_name] = topic_sample[fos_name]['top_k']
        topic_weight[t_name] = [[] for i in range(last_year)]
        topic_random[t_name] = [[] for i in range(last_year)]
        for year in range(last_year):
            topk_paper = sorted(topic_sample[fos_name][year], key=lambda x: x[1], reverse=True)[:top_k]
            topic_weight[t_name][year] = [p_id for p_id, _ in topk_paper]
            if len(topic_sample[fos_name][year]) > top_k:
                temp = random.sample(topic_sample[fos_name][year], top_k)
            else:
                temp = topic_sample[fos_name][year]
            topic_random[t_name][year] = [p_id for p_id, _ in temp]
    with open('data/sample/topic_sample_weight.json', 'w', encoding='utf8') as fout:
        json.dump(topic_weight, fout, ensure_ascii=False)    
    with open('data/sample/topic_sample_rand.json', 'w', encoding='utf8') as fout:
        json.dump(topic_random, fout, ensure_ascii=False)    
                 

def filter_young():
    """
    寻找青年学者，机构变动

    """
    year_s = 2000
    last_year = 20
    total_file = 42
    paper_org = {} # paper-作者-年-机构
    with open('data/embed/%s_map.json' % 'authors_feat', 'r', encoding='utf8') as fin:
        author_map = json.load(fin)  
    # author_map = pickle.load( open( 'data/embed/%s_map.pkl' % 'authors_feat', "rb" ) )

    
    # 统计今年以前该作者所有机构，之前合作过的作者id
    author_orgs = {}
    for a_id in author_map:
        # 过滤发文小于2
        if author_map[a_id]['n_papers'][-1] <= 1:
            continue  
        if 'author_org' not in author_map[a_id]:
            continue
        if author_map[a_id]['n_partners'][-1] == 0:
            continue

        author_orgs[a_id] = {'cum_orgs':[set() for i in range(last_year)], 'cum_partners':[set() for i in range(last_year)]}

        author_orgs[a_id]['cum_partners'][1:] = np.array(author_map[a_id]['partners']).cumsum().tolist()
        author_orgs[a_id]['cum_orgs'][1:] = np.array(author_map[a_id]['author_org_list']).cumsum().tolist() 
        author_orgs[a_id]['cum_partners'] = [set(k) for k in author_orgs[a_id]['cum_partners']]
        author_orgs[a_id]['cum_orgs'] = [set(k) for k in author_orgs[a_id]['cum_orgs']]

    res = {}
    cnt = 0
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                paper_org[paper_id] = {}
                year = one_paper.get('year', 0) 
                if year < year_s:
                    continue    
                paper_org[paper_id]['year'] = year

                if 'authors' in one_paper:
                    for auth in one_paper['authors']:
                        a_id = auth['id'] 
                        if 'org' not in auth:
                            continue
                        # 之前没有他的机构信息
                        if a_id not in author_orgs:
                            continue
                        if len(author_orgs[a_id]['cum_orgs'][year-year_s]) == 0:
                            continue
                        # 认为他换机构了
                        if sim_org(author_orgs[a_id]['cum_orgs'][year-year_s], auth['org']) == 0:
                        # if auth['org'] not in author_orgs[a_id]['cum_orgs'][year-year_s]:
                            if cnt < 20:
                                print(author_orgs[a_id]['cum_orgs'][year-year_s])
                                print(auth['org'])

                            partners = [auth['id'] for auth in one_paper['authors'] if auth['id'] != a_id]
                            # 集合相似度？
                            # print(author_orgs[a_id]['cum_partners'])
                            # 以前合作过
                            if len(author_orgs[a_id]['cum_partners'][year-year_s] & set(partners)) >=2:
                                continue
                            if a_id not in res:
                                res[a_id] = []
                            res[a_id].append((paper_id, partners))
                            cnt += 1
    
    pickle.dump(res, open("data/sample/log_data.pkl", "wb" ))     
    # with open('data/sample/log_data.json', 'w', encoding='utf8') as fout:
    #     json.dump(res, fout, ensure_ascii=False)   
    print(cnt)

def filter_young_case():
    """
    寻找青年学者，机构变动
    把变动前后都列出来 
    a_id: [{p_id:xxx, year:2001, org:xxx, cite: xx}, [{p_id:123, year:2001, org:'xxxx', cite:xxx}, {}]]
    a_id :[paper_id, [paper_id, paper_id]]
    """
    year_s = 2000
    last_year = 20
    total_file = 42
    paper_org = {} # paper-作者-年-机构
    with open('data/embed/%s_map.json' % 'authors_feat', 'r', encoding='utf8') as fin:
        author_map = json.load(fin)  
    # author_map = pickle.load( open( 'data/embed/%s_map.pkl' % 'authors_feat', "rb" ) )
    author_cnt = defaultdict(int)
    
    # 统计今年以前该作者所有机构，之前合作过的作者id
    author_orgs = {}
    
    for a_id in author_map:
        # 过滤发文小于2
        if author_map[a_id]['n_papers'][-1] <= 1:
            continue  
        if 'author_org' not in author_map[a_id]:
            continue
        if author_map[a_id]['n_partners'][-1] == 0:
            continue

        author_orgs[a_id] = {'cum_orgs':[set() for i in range(last_year)], 
                                    'cum_partners':[set() for i in range(last_year)], 'cum_papers':[[] for i in range(last_year)]}

        author_orgs[a_id]['cum_partners'][1:] = np.array(author_map[a_id]['partners']).cumsum().tolist()
        author_orgs[a_id]['cum_orgs'][1:] = np.array(author_map[a_id]['author_org_list']).cumsum().tolist() 
        author_orgs[a_id]['cum_papers'][1:] = np.array(author_map[a_id]['papers']).cumsum().tolist() 

        author_orgs[a_id]['cum_partners'] = [set(k) for k in author_orgs[a_id]['cum_partners']]
        author_orgs[a_id]['cum_orgs'] = [set(k) for k in author_orgs[a_id]['cum_orgs']]
        # author_orgs[a_id]['cum_papers'][1:] = np.array(author_map[a_id]['papers']).cumsum().tolist()

    # 保存所有paper_id : auth_id_list
    paper_authors = {}
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                if 'authors' in one_paper:
                    author_list = [auth['id'] for auth in one_paper['authors']]
                    paper_authors[paper_id] = author_list
                    


    res = {}
    cnt = 0
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                paper_org[paper_id] = {}
                year = one_paper.get('year', 0) 
                if year < year_s:
                    continue    
                paper_org[paper_id]['year'] = year

                if 'authors' in one_paper:
                    for auth in one_paper['authors']:
                        a_id = auth['id']
                        if 'org' not in auth:
                            continue
                        if a_id in res:
                            continue
                        # 之前没有他的机构信息
                        if a_id not in author_orgs:
                            continue
                        if len(author_orgs[a_id]['cum_orgs'][year-year_s]) == 0:
                            continue
                        # 认为他换机构了
                        if sim_org(author_orgs[a_id]['cum_orgs'][year-year_s], auth['org']) == 0:
                        # if auth['org'] not in author_orgs[a_id]['cum_orgs'][year-year_s]:
                            if cnt < 20:
                                print(author_orgs[a_id]['cum_orgs'][year-year_s])
                                print(auth['org'])

                            partners = [auth['id'] for auth in one_paper['authors'] if auth['id'] != a_id]
                            # 集合相似度？
                            # print(author_orgs[a_id]['cum_partners'])
                            # 以前合作过
                            if len(author_orgs[a_id]['cum_partners'][year-year_s] & set(partners)) >=2:
                                continue
                            
                            if a_id not in res:
                                res[a_id] = []

                            paper_before = author_orgs[a_id]['cum_papers'][year-year_s][-3:]
                            # 次数统计
                            cnt_before = len(author_orgs[a_id]['cum_papers'][year-year_s])
                            
                            if cnt_before > 3:
                                author_cnt[4] += 1
                            else:
                                author_cnt[cnt_before] += 1
                            # 把换后的加入
                            res[a_id].append([-1, paper_id, partners])
                            for p in paper_before:
                                partners = [aid for aid in paper_authors[p] if aid != a_id]
                                res[a_id].append([paper_id, p, partners])
                            if cnt < 2:
                                print('################')
                                print(res[a_id])
                            # res[a_id].append((paper_id, partners))
                            cnt += 1
    
    
    pickle.dump(res, open("data/sample/log_data_case_before.pkl", "wb" ))     
    # with open('data/sample/log_data.json', 'w', encoding='utf8') as fout:
    #     json.dump(res, fout, ensure_ascii=False)   
    print(author_cnt)
    print(cnt)


def pro_str(input_str):
    input_str = input_str.lower()
    input_str = str(unicodedata.normalize('NFKD', input_str).encode('ascii','ignore'), encoding = "utf-8") 
    remove_char = [',', '(', ')', '.', '-', ':']
    ignore_word = ['of', 'the']
    for c in remove_char:
        input_str = input_str.replace(c, ' ')
    res_set = set(input_str.strip().split(' '))
    temp = []
    for w in res_set:
        if len(w) <= 1:
            temp.append(w)
    for w in temp:
        res_set.remove(w)
    for w in ignore_word:
        if w in res_set:
            res_set.remove(w)
    return res_set
    
        
    

def sim_org(org_set, org_str):
    if org_str in org_set:
        return 1
    org_split_set = set()
    org_str = org_str.lower()
    for org in org_set:
        org_split_set |= pro_str(org)
    org_set = pro_str(org_str)
    if len(org_set & org_split_set) >= 2:
        # print('q ', org_set & org_split_set)
        return 1
    elif len(org_set & org_split_set) == 1:
        if len(list(org_set & org_split_set)[0]) >=4 and 'university' not in org_set & org_split_set:
            return 1
        else:
            return 0
    else:
        return 0

# def gen_train_data(case):
#     # author_map = pickle.load( open( 'data/embed/%s_map.pkl' % 'authors_feat', "rb" ))
#     with open('data/sample/author_sample.json', 'r', encoding='utf8') as fin:
#         author_sample = json.load(fin)  
#     with open('data/sample/topic_sample_weight.json', 'r', encoding='utf8') as fin:
#         topic_sample = json.load(fin)
#     with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
#         topic_feat = json.load(fin)  
#     # 统计发文大于100的topic
#     topic_set100 = set()
#     for t_name in topic_feat:
#         if topic_feat[t_name]['n_paper'] < 100:
#             continue
#         topic_set100.add(t_name)
#     print('topic len:', len(topic_set100))

            
#     if case:
#         log_data = pickle.load(open("data/sample/log_data_case_before.pkl", "rb" ))
#     else:
#         log_data = pickle.load(open("data/sample/log_data.pkl", "rb" ))

#     train_data = []
#     paper_set = set()
#     paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
#     year_s = 2000
#     # 加载paper信息

#     total_file = 42
#     for fidx in tqdm.tqdm(range(1, total_file+1)):
#         file_name = 'data/split/dblp_papers_%s.txt' % fidx
#         with open(file_name, encoding='utf8') as fin:
#             for line in fin:
#                 # 遍历
#                 one_paper = json.loads(line) 
#                 paper_id = one_paper['id'] 
                
#                 paper_info[paper_id] = {}
#                 paper_info[paper_id]['year'] = one_paper.get('year', -1)
#                 paper_info[paper_id]['n_citation'] = one_paper.get('n_citation', -1)
#                 if 'authors' in one_paper:
#                     auth_all = []
#                     for auth in one_paper['authors']:
#                         auth_id = auth['id']
#                         auth_all.append(auth_id)
#                     paper_info[paper_id]['authors'] = auth_all
#                 paper_info[paper_id]['refs'] = one_paper.get('references', [])

#                 fos_set = []
#                 if 'fos' in one_paper:
#                     for fos in one_paper['fos']:
#                         if fos['name'] in topic_set100: 
#                             fos_set.append(fos['name'])
#                 paper_info[paper_id]['topics'] = fos_set
#                 paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
#                 paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
#                 if 'venue' in one_paper:
#                     paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
#                 else:
#                     paper_info[paper_id]['venue'] = ''

            
#     # 构建训练数据
#     cnt = 0
#     if case:
#         a_list = list(log_data.keys())[:5000]
#     else:
#         a_list = list(log_data.keys())
#     for a_id in a_list:
#         for paper in log_data[a_id]:
#             p_id = paper[0]
#             if p_id in paper_set:
#                 continue
#             if paper_info[p_id]['year'] < year_s:
#                 continue
#             paper_set.add(p_id)
#             partners = paper[1]
#             year = paper_info[p_id]['year']
#             one_data = {
#                 'paper_id': p_id, 
#                 'year': year,
#                 'n_citation': paper_info[p_id]['n_citation'],
#                 'author_neighbours': [],
#                 'paper_neighbours': [],
#                 'topic_neighbours': []
#             }
#             # 作者相关邻居
#             author_neighbours = []

#             for auth in [a_id] + partners:
#                 if auth not in author_sample:
#                     continue
#                 # 作者代表作
#                 p_list = author_sample[auth][year-year_s]
#                 # 代表作的合作者
#                 co_authors = []
#                 for p in p_list:
#                     co_authors.extend(paper_info[p]['authors'])
#                 author_neighbours.append([auth, [p_list, co_authors]])
#             one_data['author_neighbours'] = author_neighbours
#             # paper相关邻居 只取近20年？
#             paper_neighbours = []
#             for p in paper_info[p_id]['refs']:
                
#                 if paper_info[p]['year'] < year_s:
#                     continue
#                 auth_list = paper_info[p]['authors']
#                 topics = paper_info[p]['topics']
#                 # 近20年
#                 refs = [_ref for _ref in paper_info[p]['refs'] if paper_info[_ref]['year'] >= year_s]
#                 # refs = paper_info[p]['refs']
#                 ref_authors = []
#                 for p_ref in refs:
#                     ref_authors.extend(paper_info[p_ref]['authors'])
#                 # 历史发表文章 
#                 pre_papers = []
#                 for auth in auth_list:
#                     if auth not in author_sample:
#                         continue
#                     pre_papers.extend(author_sample[auth][paper_info[p]['year']-year_s])
#                 paper_neighbours.append([p, auth_list, topics, refs, ref_authors, pre_papers])
#             one_data['paper_neighbours'] = paper_neighbours
#             # topic 相关邻居
#             topic_neighbours = []
#             for topic in paper_info[p_id]['topics']:
#                 related_papers = topic_sample[topic][year-year_s]
#                 auth_list = []
#                 for p in related_papers:
#                     auth_list.extend(paper_info[p]['authors'])
#                 topic_neighbours.append([topic, related_papers, auth_list])
#             one_data['topic_neighbours'] = topic_neighbours

#             one_data['doc_type'] = paper_info[p_id]['doc_type']
#             one_data['venue'] = paper_info[p_id]['venue']
#             one_data['publisher'] = paper_info[p_id]['publisher']

#             train_data.append(one_data)
#             cnt += 1
#     print(len(train_data))
#     print(train_data[12])
#     if case:
#         pickle.dump(train_data, open("data/sample/train_data_case.pkl", "wb" ))   
#     else:
#         pickle.dump(train_data, open("data/sample/train_data.pkl", "wb" ))   



def gen_train_data_casev2():
    # author_map = pickle.load( open( 'data/embed/%s_map.pkl' % 'authors_feat', "rb" ))
    with open('data/sample/author_sample.json', 'r', encoding='utf8') as fin:
        author_sample = json.load(fin)  
    with open('data/sample/topic_sample_weight.json', 'r', encoding='utf8') as fin:
        topic_sample = json.load(fin)
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_feat = json.load(fin)  
    # 统计发文大于100的topic
    topic_set100 = set()
    for t_name in topic_feat:
        if topic_feat[t_name]['n_paper'] < 100:
            continue
        topic_set100.add(t_name)
    print('topic len:', len(topic_set100))

        
    log_data = pickle.load(open("data/sample/log_data_case_before.pkl", "rb" ))


    train_data = []
    paper_set = set()
    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    year_s = 2000
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

                fos_set = []
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        if fos['name'] in topic_set100: 
                            fos_set.append(fos['name'])
                paper_info[paper_id]['topics'] = fos_set
                paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
                paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
                if 'venue' in one_paper:
                    paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
                else:
                    paper_info[paper_id]['venue'] = ''

            
    # 构建训练数据
    cnt = 0
    for a_id in list(log_data.keys())[:5000]:
        for paper in log_data[a_id]:
            p_id = paper[1]
            p_id_target = paper[0]
            if p_id in paper_set:
                continue
            if paper_info[p_id]['year'] < year_s:
                continue
            paper_set.add(p_id)
            partners = paper[2]
            year = paper_info[p_id]['year']
            one_data = {
                'paper_id': p_id, 
                'year': year,
                'p_id_target':p_id_target,
                'n_citation': paper_info[p_id]['n_citation'],
                'author_neighbours': [],
                'paper_neighbours': [],
                'topic_neighbours': []
            }
            # 作者相关邻居
            author_neighbours = []

            for auth in [a_id] + partners:
                if auth not in author_sample:
                    continue
                # 作者代表作
                p_list = author_sample[auth][year-year_s]
                # 代表作的合作者
                co_authors = []
                for p in p_list:
                    co_authors.extend(paper_info[p]['authors'])
                author_neighbours.append([auth, [p_list, co_authors]])
            one_data['author_neighbours'] = author_neighbours
            # paper相关邻居 只取近20年？
            paper_neighbours = []
            for p in paper_info[p_id]['refs']:
                
                if paper_info[p]['year'] < year_s:
                    continue
                auth_list = paper_info[p]['authors']
                topics = paper_info[p]['topics']
                # 近20年
                refs = [_ref for _ref in paper_info[p]['refs'] if paper_info[_ref]['year'] >= year_s]
                # refs = paper_info[p]['refs']
                ref_authors = []
                for p_ref in refs:
                    ref_authors.extend(paper_info[p_ref]['authors'])
                # 历史发表文章 
                pre_papers = []
                for auth in auth_list:
                    if auth not in author_sample:
                        continue
                    pre_papers.extend(author_sample[auth][paper_info[p]['year']-year_s])
                paper_neighbours.append([p, auth_list, topics, refs, ref_authors, pre_papers])
            one_data['paper_neighbours'] = paper_neighbours
            # topic 相关邻居
            topic_neighbours = []
            for topic in paper_info[p_id]['topics']:
                related_papers = topic_sample[topic][year-year_s]
                auth_list = []
                for p in related_papers:
                    auth_list.extend(paper_info[p]['authors'])
                topic_neighbours.append([topic, related_papers, auth_list])
            one_data['topic_neighbours'] = topic_neighbours

            one_data['doc_type'] = paper_info[p_id]['doc_type']
            one_data['venue'] = paper_info[p_id]['venue']
            one_data['publisher'] = paper_info[p_id]['publisher']

            train_data.append(one_data)
            cnt += 1
    print(len(train_data))
    print(train_data[12])

    pickle.dump(train_data, open("data/sample/train_data_case.pkl", "wb" ))   


def gen_train_data_case():
    """
    根据topic相似度+cite3找到 1w*5 条数据
    用青年学者替换其他4条中的一作
    最好包含大牛或者 合作者里（第一条）有大牛
    是否需要同一年？在过滤时候
    """
    with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
        references_3_map = json.load(fin) 
    # author_map = pickle.load( open( 'data/embed/%s_map.pkl' % 'authors_feat', "rb" ))
    with open('data/sample/author_sample.json', 'r', encoding='utf8') as fin:
        author_sample = json.load(fin)  
    with open('data/sample/topic_sample_weight.json', 'r', encoding='utf8') as fin:
        topic_sample = json.load(fin)
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_feat = json.load(fin)  
    # 统计发文大于100的topic
    topic_set100 = set()
    for t_name in topic_feat:
        if topic_feat[t_name]['n_paper'] < 100:
            continue
        topic_set100.add(t_name)
    print('topic len:', len(topic_set100))

        
    log_data = pickle.load(open("data/sample/log_data.pkl", "rb" ))
    train_data = {}
    paper_set = set()
    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    year_s = 2000
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
                paper_info[paper_id]['n_citation'] = references_3_map[paper_id].get('n_citation3', -1)
                # paper_info[paper_id]['n_citation'] = one_paper.get('n_citation', -1)
                if 'authors' in one_paper:
                    auth_all = []
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                    paper_info[paper_id]['authors'] = auth_all
                paper_info[paper_id]['refs'] = one_paper.get('references', [])

                fos_set = []
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        if fos['name'] in topic_set100: 
                            fos_set.append(fos['name'])
                paper_info[paper_id]['topics'] = fos_set
                paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
                paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
                if 'venue' in one_paper:
                    paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
                else:
                    paper_info[paper_id]['venue'] = ''

    topic_paper = {} # topic_list_str: [pid1, pid2]
    # 构建训练数据
    cnt = 0
    for a_id in log_data:
        for paper in log_data[a_id]:
            p_id = paper[0]
            if p_id in paper_set:
                continue
            if paper_info[p_id]['year'] < year_s:
                continue
            paper_set.add(p_id)
            partners = paper[1]
            year = paper_info[p_id]['year']
            one_data = {
                'paper_id': p_id, 
                'year': year,
                'n_citation': paper_info[p_id]['n_citation'],
                'author_neighbours': [],
                'paper_neighbours': [],
                'topic_neighbours': [],
                'topic_str':""
            }
            # 作者相关邻居
            author_neighbours = []

            for auth in [a_id] + partners:
                if auth not in author_sample:
                    continue
                # 作者代表作
                p_list = author_sample[auth][year-year_s]
                # 代表作的合作者
                co_authors = []
                for p in p_list:
                    co_authors.extend(paper_info[p]['authors'])
                author_neighbours.append([auth, [p_list, co_authors]])
            one_data['author_neighbours'] = author_neighbours
            # paper相关邻居 只取近20年？
            paper_neighbours = []
            for p in paper_info[p_id]['refs']:
                
                if paper_info[p]['year'] < year_s:
                    continue
                auth_list = paper_info[p]['authors']
                topics = paper_info[p]['topics']
                # 近20年
                refs = [_ref for _ref in paper_info[p]['refs'] if paper_info[_ref]['year'] >= year_s]
                # refs = paper_info[p]['refs']
                ref_authors = []
                for p_ref in refs:
                    ref_authors.extend(paper_info[p_ref]['authors'])
                # 历史发表文章 
                pre_papers = []
                for auth in auth_list:
                    if auth not in author_sample:
                        continue
                    pre_papers.extend(author_sample[auth][paper_info[p]['year']-year_s])
                paper_neighbours.append([p, auth_list, topics, refs, ref_authors, pre_papers])
            one_data['paper_neighbours'] = paper_neighbours
            # topic 相关邻居
            topic_neighbours = []
            for topic in paper_info[p_id]['topics']:
                related_papers = topic_sample[topic][year-year_s]
                auth_list = []
                for p in related_papers:
                    auth_list.extend(paper_info[p]['authors'])
                topic_neighbours.append([topic, related_papers, auth_list])
            one_data['topic_neighbours'] = topic_neighbours

            one_data['doc_type'] = paper_info[p_id]['doc_type']
            one_data['venue'] = paper_info[p_id]['venue']
            one_data['publisher'] = paper_info[p_id]['publisher']

            
            topic_str = '_'.join(sorted([str(topic) for topic in paper_info[p_id]['topics']]))
            if topic_str not in topic_paper:
                topic_paper[topic_str] = []
            topic_paper[topic_str].append(p_id)
            one_data['topic_str'] = topic_str
            train_data[p_id] = one_data

            cnt += 1
    print(len(train_data))
    print('topic_paper: ', len(topic_paper))
    # print(train_data[12])
    # 寻找1w *5
    cnt_case = 0
    case_data = []
    
    for paper_id in tqdm.tqdm(train_data):
        # topic 过滤
        paper = train_data[paper_id]
        topic_str = paper['topic_str']
        if '_' not in topic_str:
            # 只有一个topic
            continue
        # cite 过滤
        sim_paper = [p_id for p_id in topic_paper[topic_str] if p_id != paper_id]
        sim_paper = [p_id for p_id in sim_paper if \
            (paper_info[p_id]['n_citation'] >=  0.5*paper_info[paper_id]['n_citation'] and \
                paper_info[p_id]['n_citation'] <=  1.5*paper_info[paper_id]['n_citation'])]
        if len(sim_paper) >= 4:
            sim_paper = sim_paper[:4]
        else:
            continue    
        case_data.append(train_data[paper_id])
        # 替换一作
        for p_id in sim_paper:
            s_paper = train_data[p_id]
            s_paper['author_neighbours'][0] =  train_data[paper_id]['author_neighbours'][0]
            case_data.append(s_paper)
        cnt_case += 1
        print(cnt_case)
        if cnt_case == 10000:
            break
    print(cnt_case, 'the same topic !!!!!!!!!!')
    print('other ...')

    for paper_id in tqdm.tqdm(train_data):
        # topic 过滤
        paper = train_data[paper_id]
        # paper_id = paper['paper_id']
        topic_str = paper['topic_str']
        if '_' not in topic_str:
            # 只有一个topic
            continue
        # sim_paper = [p_id for p_id in topic_paper[topic_str][:5] if p_id != paper_id][:4]
        # cite 过滤
        sim_paper = [p_id for p_id in topic_paper[topic_str] if p_id != paper_id]
        sim_paper = [p_id for p_id in sim_paper if \
            (paper_info[p_id]['n_citation'] >=  0.5*paper_info[paper_id]['n_citation'] and \
                paper_info[p_id]['n_citation'] <=  1.5*paper_info[paper_id]['n_citation'])]
        if len(sim_paper) >= 4:
            continue
            # sim_paper = sim_paper[:4]
        else:
            # continue
            # 相差一个》
            sim_topic = [t_str for t_str in topic_paper \
                    if len(set(topic_str.split('_')) - set(t_str.split('_'))) == 1]
            all_sim = []
            for t_str in sim_topic:
                all_sim.extend([p_id for p_id in topic_paper[t_str] if \
                    (paper_info[p_id]['n_citation'] >=  0.5*paper_info[paper_id]['n_citation'] and \
                        paper_info[p_id]['n_citation'] <=  1.5*paper_info[paper_id]['n_citation'])])
            
            random.shuffle(all_sim)
            sim_paper += all_sim[:4-len(sim_paper)]
            
        if len(sim_paper) != 4:
            continue        
        case_data.append(train_data[paper_id])
        for p_id in sim_paper:
            s_paper = train_data[p_id]
            s_paper['author_neighbours'][0] =  train_data[paper_id]['author_neighbours'][0]
            case_data.append(s_paper)
        # to_replace = [train_data[p_id] for p_id in sim_paper]
        # case_data.extend([train_data[p_id] for p_id in sim_paper])
        cnt_case += 1
        print(cnt_case)
        if cnt_case == 10000:
            break

    print(len(case_data) / 5)
    # print(case_data[30:35])

    pickle.dump(case_data, open("data/sample/train_data_case_1w.pkl", "wb" ))   


def gen_train_data_case_first_eq():
    """
    根据topic相似度+cite3找到 1w*5 条数据
    用青年学者替换其他4条中的一作
    最好包含大牛或者 合作者里（第一条）有大牛
    是否需要同一年？在过滤时候
    """
    with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
        references_3_map = json.load(fin) 
    # author_map = pickle.load( open( 'data/embed/%s_map.pkl' % 'authors_feat', "rb" ))
    with open('data/sample/author_sample.json', 'r', encoding='utf8') as fin:
        author_sample = json.load(fin)  
    with open('data/sample/topic_sample_weight.json', 'r', encoding='utf8') as fin:
        topic_sample = json.load(fin)
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_feat = json.load(fin)  
    # 统计发文大于100的topic
    topic_set100 = set()
    for t_name in topic_feat:
        if topic_feat[t_name]['n_paper'] < 100:
            continue
        topic_set100.add(t_name)
    print('topic len:', len(topic_set100))

        
    log_data = pickle.load(open("data/sample/log_data.pkl", "rb" ))
    train_data = {}
    paper_set = set()
    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    year_s = 2000
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
                paper_info[paper_id]['n_citation'] = references_3_map[paper_id].get('n_citation3', -1)
                # paper_info[paper_id]['n_citation'] = one_paper.get('n_citation', -1)
                if 'authors' in one_paper:
                    auth_all = []
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                    paper_info[paper_id]['authors'] = auth_all
                paper_info[paper_id]['refs'] = one_paper.get('references', [])

                fos_set = []
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        if fos['name'] in topic_set100: 
                            fos_set.append(fos['name'])
                paper_info[paper_id]['topics'] = fos_set
                paper_info[paper_id]['doc_type'] = one_paper.get('doc_type', '')
                paper_info[paper_id]['publisher'] = one_paper.get('publisher', '')
                if 'venue' in one_paper:
                    paper_info[paper_id]['venue'] = one_paper['venue'].get('raw', '')
                else:
                    paper_info[paper_id]['venue'] = ''

    topic_paper = {} # topic_list_str: [pid1, pid2]
    # 构建训练数据
    cnt = 0
    for a_id in log_data:
        for paper in log_data[a_id]:
            p_id = paper[0]
            if p_id in paper_set:
                continue
            if paper_info[p_id]['year'] < year_s:
                continue
            paper_set.add(p_id)
            partners = paper[1]
            year = paper_info[p_id]['year']
            one_data = {
                'paper_id': p_id, 
                'year': year,
                'n_citation': paper_info[p_id]['n_citation'],
                'author_neighbours': [],
                'paper_neighbours': [],
                'topic_neighbours': [],
                # 'topic_str':""
            }
            # 作者相关邻居
            author_neighbours = []

            for auth in [a_id] + partners:
                if auth not in author_sample:
                    continue
                # 作者代表作
                p_list = author_sample[auth][year-year_s]
                # 代表作的合作者
                co_authors = []
                for p in p_list:
                    co_authors.extend(paper_info[p]['authors'])
                author_neighbours.append([auth, [p_list, co_authors]])
            one_data['author_neighbours'] = author_neighbours
            # paper相关邻居 只取近20年？
            paper_neighbours = []
            for p in paper_info[p_id]['refs']:
                
                if paper_info[p]['year'] < year_s:
                    continue
                auth_list = paper_info[p]['authors']
                topics = paper_info[p]['topics']
                # 近20年
                refs = [_ref for _ref in paper_info[p]['refs'] if paper_info[_ref]['year'] >= year_s]
                # refs = paper_info[p]['refs']
                ref_authors = []
                for p_ref in refs:
                    ref_authors.extend(paper_info[p_ref]['authors'])
                # 历史发表文章 
                pre_papers = []
                for auth in auth_list:
                    if auth not in author_sample:
                        continue
                    pre_papers.extend(author_sample[auth][paper_info[p]['year']-year_s])
                paper_neighbours.append([p, auth_list, topics, refs, ref_authors, pre_papers])
            one_data['paper_neighbours'] = paper_neighbours
            # topic 相关邻居
            topic_neighbours = []
            for topic in paper_info[p_id]['topics']:
                related_papers = topic_sample[topic][year-year_s]
                auth_list = []
                for p in related_papers:
                    auth_list.extend(paper_info[p]['authors'])
                topic_neighbours.append([topic, related_papers, auth_list])
            one_data['topic_neighbours'] = topic_neighbours

            one_data['doc_type'] = paper_info[p_id]['doc_type']
            one_data['venue'] = paper_info[p_id]['venue']
            one_data['publisher'] = paper_info[p_id]['publisher']

            if len(paper_info[p_id]['topics']) > 0:
                # topic_str = '_'.join(sorted([str(topic) for topic in paper_info[p_id]['topics']]))
                topic_str = paper_info[p_id]['topics'][0]
                if topic_str not in topic_paper:
                    topic_paper[topic_str] = []
                topic_paper[topic_str].append(p_id)
                one_data['topic_str'] = topic_str
            train_data[p_id] = one_data

            cnt += 1
    print(len(train_data))
    print('topic_paper: ', len(topic_paper))
    # print(train_data[12])
    # 寻找1w *5
    cnt_case = 0
    case_data = []
    case_paper = set() # 保存所有case pid
    for paper_id in tqdm.tqdm(train_data):
        # topic 过滤
        if paper_id in case_paper:
            continue

        paper = train_data[paper_id]
        if 'topic_str' not in paper:
            continue
        topic_str = paper['topic_str']
        # cite 过滤
        sim_paper = [p_id for p_id in topic_paper[topic_str] if p_id != paper_id and p_id not in case_paper]
        sim_paper = [p_id for p_id in sim_paper if \
            (paper_info[p_id]['n_citation'] >=  0.5*paper_info[paper_id]['n_citation'] and \
                paper_info[p_id]['n_citation'] <=  1.5*paper_info[paper_id]['n_citation'])]
        if len(sim_paper) >= 4:
            sim_paper = sim_paper[:4]
        else:
            continue    
        case_data.append(train_data[paper_id])
        # 替换一作
        for p_id in sim_paper:
            s_paper = train_data[p_id]
            s_paper['author_neighbours'][0] =  train_data[paper_id]['author_neighbours'][0]
            case_data.append(s_paper)
            case_paper.add(p_id)
        
        case_paper.add(paper_id)
        cnt_case += 1
        # print(cnt_case)
        # if cnt_case == 10000:
        #     break
    print(cnt_case, 'the same topic first !!!!!!!!!!')
    print('other ...')

    print(len(case_data) / 5)
    # print(case_data[30:35])

    pickle.dump(case_data, open("data/sample/train_data_case_1w.pkl", "wb" ))   





def format_train_data():
    # {paper_id,n_citation,[author_ids],[topic_names]}
    train_data = pickle.load(open("data/sample/train_data.pkl", "rb" ))  
    with open('data/embed/references_3_map.json', 'r', encoding='utf8') as fin:
        references_3_map = json.load(fin) 
    new_data_year0, new_data_year1, new_data_year2, new_data_year3 = [], [], [], []
    for data in tqdm.tqdm(train_data):
        if data['year'] >= 2017:
            continue
        paper_id = data['paper_id']
        n_citation0 = references_3_map[paper_id]['n_citation0']
        n_citation1 = references_3_map[paper_id]['n_citation1']
        n_citation2 = references_3_map[paper_id]['n_citation2']
        n_citation3 = references_3_map[paper_id]['n_citation3']

        auth_list = [a[0] for a in data['author_neighbours']]
        topic_names = [a[0] for a in data['topic_neighbours']]
        new_data_year0.append([paper_id, n_citation0, auth_list, topic_names])
        new_data_year1.append([paper_id, n_citation1, auth_list, topic_names])
        new_data_year2.append([paper_id, n_citation2, auth_list, topic_names])
        new_data_year3.append([paper_id, n_citation3, auth_list, topic_names])

    print(new_data_year1[1:10])
    print(new_data_year2[1:10])
    print('len: ', len(new_data_year1))
    pickle.dump(new_data_year0, open("data/format/train_data2017_year0.pkl", "wb" )) 
    pickle.dump(new_data_year1, open("data/format/train_data2017_year1.pkl", "wb" )) 
    pickle.dump(new_data_year2, open("data/format/train_data2017_year2.pkl", "wb" )) 
    pickle.dump(new_data_year3, open("data/format/train_data2017_year3.pkl", "wb" )) 

    

def gen_join_mat():
    # paper/author/topic节点的邻接关系
        # 格式：
        # {paper_id p/a/t node_id
        # ...}
    
    # 统计发文大于100的topic
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_feat = json.load(fin)  
    topic_set100 = set()
    for t_name in topic_feat:
        if topic_feat[t_name]['n_paper'] < 100:
            continue
        topic_set100.add(t_name)
    print('topic len:', len(topic_set100))
    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    # 加载paper信息
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                if one_paper.get('year', -1) < 2000:
                    continue
                paper_id = one_paper['id'] 
                
                paper_info[paper_id] = {}
                paper_info[paper_id]['year'] = one_paper.get('year', -1)
                auth_all = []
                if 'authors' in one_paper:
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                paper_info[paper_id]['authors'] = auth_all
                # 还没过滤近20年
                paper_info[paper_id]['refs'] = one_paper.get('references', [])
                
                fos_set = []
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        if fos['name'] in topic_set100: 
                            fos_set.append(fos['name'])
                paper_info[paper_id]['topics'] = fos_set

    new_data = []
    for pid in paper_info:
        for p in paper_info[pid]['refs']:
            # 近20年
            if p not in paper_info:
                continue
            new_data.append([pid, 'p', p])
        

        for a in paper_info[pid]['authors']:
            new_data.append([pid, 'a', a])

        for t in paper_info[pid]['topics']:
            new_data.append([pid, 't', t])
    print(new_data[1:10])
    print(len(new_data))
    pickle.dump(new_data, open("data/format/adjmat_a_p_t.pkl", "wb" )) 
        
def filter_adjmat():
    """
    统计：所有被测的paper、青年学者id、被测paper的所有引文
    再生成adjmat
    """
    train_data = pickle.load(open("data/sample/train_data.pkl", "rb" ))   
    predicted_paper, young_author, prepapers_ref = set(), set(), set()
    for one_paper in train_data:
        predicted_paper.add(one_paper['paper_id'])
        # 被预测paper的所有作者都算上了 TODO
        young_author.add(one_paper['author_neighbours'][0][0])
        # young_author |= set([auth[0] for auth in one_paper['author_neighbours']])
        prepapers_ref |= set([ref[0] for ref in one_paper['paper_neighbours']])
    

    # paper/author/topic节点的邻接关系
        # 格式：
        # {paper_id p/a/t node_id
        # ...}
    
    # 统计发文大于100的topic
    with open('data/embed/fos_feat_map.json', 'r', encoding='utf8') as fin:
        topic_feat = json.load(fin)  
    topic_set100 = set()
    for t_name in topic_feat:
        if topic_feat[t_name]['n_paper'] < 100:
            continue
        topic_set100.add(t_name)
    print('topic len:', len(topic_set100))
    paper_info = {} # year  # 作者 authors # 引用的论文 refs # 文章topics
    # 加载paper信息
    total_file = 42
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                # 遍历
                one_paper = json.loads(line) 
                if one_paper.get('year', -1) < 2000:
                    continue
                paper_id = one_paper['id'] 
                
                paper_info[paper_id] = {}
                paper_info[paper_id]['year'] = one_paper.get('year', -1)
                auth_all = []
                if 'authors' in one_paper:
                    for auth in one_paper['authors']:
                        auth_id = auth['id']
                        auth_all.append(auth_id)
                paper_info[paper_id]['authors'] = auth_all
                # 还没过滤近20年
                paper_info[paper_id]['refs'] = one_paper.get('references', [])
                
                fos_set = []
                if 'fos' in one_paper:
                    for fos in one_paper['fos']:
                        if fos['name'] in topic_set100: 
                            fos_set.append(fos['name'])
                paper_info[paper_id]['topics'] = fos_set

    new_data = []
    cnt2 = 0
    for pid in paper_info:
        if pid not in predicted_paper and pid not in prepapers_ref:
            cnt2 += 1
            has_young = False
            for a in paper_info[pid]['authors']:
                if a in young_author:
                    has_young = True
                    break
            if not has_young:
                continue

        for p in paper_info[pid]['refs']:
            # 近20年
            if p not in paper_info:
                continue
            new_data.append([pid, 'p', p])
        

        for a in paper_info[pid]['authors']:
            new_data.append([pid, 'a', a])

        for t in paper_info[pid]['topics']:
            new_data.append([pid, 't', t])
    print(new_data[1:10])
    print(len(new_data))
    print('cnt2: ', cnt2)
    pickle.dump(new_data, open("data/format/adjmat_a_p_t_filter.pkl", "wb" )) 

    




def cal_cnt():
    # 统计traindata里论文、作者数目
    """
    文章的作者数量统计
    80%的文章，作者不多于____人
    90%的文章，作者不多于____人
    85%的文章，作者不多于____人

    文章的参考文献数量统计
    80%的文章，参考文献不多于____篇
    90%的文章，参考文献不多于____篇
    85%的文章，参考文献不多于____篇

    """
    train_data = pickle.load(open("data/sample/train_data.pkl", "rb" ))   
    author_cnt = []
    paper_cnt = []
    topic_cnt = []
    for data in tqdm.tqdm(train_data):
        author_cnt.append(len(data['author_neighbours']))
        paper_cnt.append(len(data['paper_neighbours']))
        topic_cnt.append(len(data['topic_neighbours']))
        if len(data['author_neighbours']) >40:
            print('40: ', data['paper_id'])
        if len(data['paper_neighbours']) >380:
            print('380: ', data['paper_id'])
    author_cnt = sorted(author_cnt)
    paper_cnt = sorted(paper_cnt)
    topic_cnt = sorted(topic_cnt)

    percent = [0.8, 0.85, 0.9, 0.95]
    for per in percent:
        au = author_cnt[int(len(author_cnt)*per)-1]
        pa = paper_cnt[int(len(paper_cnt)*per)-1]
        top = topic_cnt[int(len(topic_cnt)*per)-1]
        print("{} percent: paper: {}, author: {}, topic {}".format(per, pa, au, top))


    










                        


def cal_cnt_0508():
    # 统计留下多少学者和topic
    path = '/home/zhuangyan/repos/gcn/data/sample/train_data.pkl'
    train_data = pickle.load( open(path, "rb" ))
    set_topic = set()
    set_author = set()
    set_paper = set()
    for data in tqdm.tqdm(train_data):
        for a in data['author_neighbours']:
            set_author.add(a[0])
            set_author |= set(a[1][1])

        for a in data['paper_neighbours']:
            set_author |= set(a[1])
            set_author |= set(a[4])

        for a in data['topic_neighbours']:
            set_author |= set(a[2])
        # topic
        for t in data['paper_neighbours']:
            set_topic |= set(t[2])
        set_topic |= set([t[0] for t in data['topic_neighbours']])
        # paper
        set_paper.add(data['paper_id'])
        for p in data['author_neighbours']:
            set_paper |= set(p[1][0])
        for p in data['paper_neighbours']:
            set_paper.add(p[0])
            set_paper |= set(p[3])       
            set_paper |= set(p[5])  
        for p in data['topic_neighbours']:
            set_paper |= set(p[1])       
    print('paper:', len(set_paper))
    print('author:', len(set_author))
    print('topic:', len(set_topic))



def search_info_by_id(p_id):
    total_file = 42
    res = {'title': 'xxx', 'authors': 'xxx', 'year':'xxx'}
    for fidx in tqdm.tqdm(range(1, total_file+1)):
        file_name = 'data/split/dblp_papers_%s.txt' % fidx
        with open(file_name, encoding='utf8') as fin:
            for line in fin:
                one_paper = json.loads(line) 
                paper_id = one_paper['id']
                if p_id == paper_id:
                    res['authors'] = one_paper['authors']
                    res['title'] = one_paper['title']
                    res['year'] = one_paper['year']
                    return res
    return None

def case_get_info_by_id():
    path = 'data/case/case.txt'
    res = {}
    with open(path) as f:
        lines = f.readlines()
        # print(lines)
        # line_idx = 0
        for line in tqdm.tqdm(lines):
            line = line.strip()
            if len(line) == 0:
                continue
            paper_id = line.split(',')[2].strip()
            paper_target = line.split(',')[3].strip() # 可能是-1
            if paper_id != '-1':
                res[paper_id] = search_info_by_id(paper_id)
            if paper_target != '-1':
                res[paper_target] = search_info_by_id(paper_target)
            # res[paper_id] = search_info_by_id(paper_id)
            # res[paper_target] = search_info_by_id(paper_target)
            # print(res)
            # print(paper_id, paper_target)
    with open('data/case/case_info.json', 'w', encoding='utf8') as fout:
        json.dump(res, fout, ensure_ascii=False)
    
    



    
            



if __name__ == '__main__':
    # divide_data('assist0910')
    # preprocess1()
    # train_model('DM')
    # fname_ab = 'data/Para2Vec_Models/paragraph_DBOW_ab.doc2vec'
    # fname_title = 'data/Para2Vec_Models/paragraph_DBOW_ab_title_128.doc2vec'
    # filter_adjmat()
    # re_cal_cite()
    # print(search_by_id('2561119724'))
    # print(search_by_id("2046621340"))
    # get_topic_vector(fname_ab)
    # get_author_feat()
    # cal_fos()
    # hist_info_author()
    # sample_author()
    # sample_topic()
    # filter_young_case()
    # gen_train_data(True)
    # gen_train_data_casev2()
    case_get_info_by_id()
    # cal_cnt()
    # format_train_data()
    # gen_join_mat()
    # gen_vec_file(fname_ab, 'ab')
    # doc_str = "Representation learning in heterogeneous graphs aims to pursue a meaningful vector representation for each node so as to facili- tate downstream applications such as link prediction, personalized recommendation, node classification, etc. This task, however, is challenging not only because of the demand to incorporate het- erogeneous structural (graph) information consisting of multiple types of nodes and edges, but also due to the need for considering heterogeneous attributes or contents (e.д., text or image) associ- ated with each node. Despite a substantial amount of effort has been made to homogeneous (or heterogeneous) graph embedding, attributed graph embedding as well as graph neural networks, few of them can jointly consider heterogeneous structural (graph) infor- mation as well as heterogeneous contents information of each node effectively. In this paper, we propose HetGNN, a heterogeneous graph neural network model, to resolve this issue. Specifically, we first introduce a random walk with restart strategy to sample a fixed size of strongly correlated heterogeneous neighbors for each node and group them based upon node types. Next, we design a neural network architecture with two modules to aggregate feature information of those sampled neighboring nodes. The first module encodes “deep” feature interactions of heterogeneous contents and generates content embedding for each node. The second module aggregates content (attribute) embeddings of different neighboring groups (types) and further combines them by considering the im- pacts of different groups to obtain the ultimate node embedding. Finally, we leverage a graph context loss and a mini-batch gradient descent procedure to train the model in an end-to-end manner. Ex- tensive experiments on several datasets demonstrate that HetGNN can outperform state-of-the-art baselines in various graph mining tasks, i .e ., link prediction, recommendation, node classification & clustering and inductive node classification & clustering."
    # assess_model(doc_str, fname_ab, 'ab')
    # cal_cnt_0508()
    # gen_train_data_case_first_eq()