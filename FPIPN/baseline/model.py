import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import pickle
import tqdm
from dataset import train_dataset, CustomBatch, train_dataset_deepwalk
import os
import yaml
from  torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error
torch.multiprocessing.set_sharing_strategy('file_system')
#数据集加载
def tensor_to_numpy(tensor):
    """
    将torch.tensor 转换为numpy
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        return tensor
#网络定义
class gcn_net(torch.nn.Module):
    def __init__(self, num_node_features, dropout, conv_dim, device):
        # num_node + 1
        super(gcn_net, self).__init__()
        self.device = device
        self.dropout = dropout
        self.num_node_features = num_node_features
        # self.node_feats = nn.Embedding(num_nodes, num_node_features)
        topic_dim = 130
        au_dim = 37
        paper_dim = 289
        self.topic_fc = nn.Linear(topic_dim, num_node_features)
        self.au_fc = nn.Linear(au_dim, num_node_features)
        self.paper_fc = nn.Linear(paper_dim, num_node_features)
        # 卷积
        self.conv1_dim = conv_dim
        
        self.conv1 = GCNConv(num_node_features, self.conv1_dim)

        # mlp
        fc_dim1 = 2*self.conv1_dim
        fc_dim2 = int(fc_dim1 / 2)
        fc_dim3 = 1 # 1
        self.prednet1 = nn.Linear(3*self.conv1_dim, fc_dim1)
        self.prednet2 = nn.Linear(fc_dim1, fc_dim2)
        self.prednet3 = nn.Linear(fc_dim2, fc_dim3)
        for name, param in self.named_parameters():
            if 'weight' in name:
                print(name)
                nn.init.xavier_normal_(param)

    def forward(self, author_ids, topic_ids, auth_cnts, topic_cnts, paper_ids, id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, au_embs, paper_embs):
        bs = len(author_ids)
        pred_logits = []
        for idx in range(bs):
            author_id, topic_id, auth_cnt, topic_cnt, paper_id, id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb = author_ids[idx], topic_ids[idx], auth_cnts[idx], topic_cnts[idx], paper_ids[idx], id_maps[idx], edge_indexs[idx], paper_sets[idx], author_sets[idx], topic_sets[idx], topic_embs[idx], au_embs[idx], paper_embs[idx]
            # feat, edge_index = data.x, data.edge_index
            feat = torch.zeros(len(id_map)+1, self.num_node_features).to(self.device)
            topic_f = self.topic_fc(topic_emb.to(self.device))
            au_f = self.au_fc(au_emb.to(self.device))
            paper_f = self.paper_fc(paper_emb.to(self.device))
            topic_f = F.dropout(torch.tanh(topic_f), p=self.dropout, training=self.training)
            au_f = F.dropout(torch.tanh(au_f), p=self.dropout, training=self.training)
            paper_f = F.dropout(torch.tanh(paper_f), p=self.dropout, training=self.training)
            
            feat[topic_set] = topic_f
            feat[author_set] = au_f
            feat[paper_set] = paper_f
            # data_graph = Data(x=feat, edge_index=edge_index)
            # author_ids [bs, 6]
            # topic_ids [bs, 12]
            feat = self.conv1(feat, edge_index.to(self.device))
            feat[0] = 0.00001
            # mlp 
            topic_emb = feat[topic_id] # bs, 12, emb
            author_emb = feat[author_id] # bs, 6, emb
            topic_emb = topic_emb.sum(0) / (0.00001+topic_cnt)# avg bs, emb

            # author_emb = author_emb.sum(0) / (0.00001+auth_cnt)# avg bs, emb
            author_emb = torch.cat([author_emb[0,:], author_emb[1:,:].sum(0)/(-1.00001+auth_cnt)], 0)

            # author_emb = author_emb.view(author_emb.shape[0], -1) # bs ,6*emb
            # print(author_emb.shape, topic_emb.shape)
            cat_emb = torch.cat([author_emb, topic_emb], 0)
            pred_logits.append(cat_emb)

        cat_emb = torch.stack(pred_logits, 0)
        cat_emb = torch.tanh(cat_emb)
        cat_emb = F.dropout(cat_emb, p=self.dropout, training=self.training)
        output = self.prednet1(cat_emb)
        output = torch.tanh(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.prednet2(output) # [bs, fc_dim2]
        output = torch.tanh(output)
        # output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.prednet3(output) # [bs, fc_dim2]
        # output = F.relu(output)

        # 分类
        # return output
        # output = F.dropout(output, p=self.dropout, training=self.training)

        # output = torch.argmax(output, dim=1)
        # output = torch.relu(output).view(-1)
        # output = torch.log(output)
        
        # output = torch.log(output)
        # output = torch.sigmoid(output)
        
        # output = F.dropout(output, p=self.dropout, training=self.training)

        return output.view(-1)
    def save_model(self, ep, note):
        torch.save(self.state_dict(), 'model/gcn_parameter_{}_ep{}.pkl'.format(note, ep))
        # print('model saved')

    def load_model(self, path):
        state_dict = torch.load(path)
        load_dict = {k:v for k, v in state_dict.items()}
        self.load_state_dict(load_dict)

        
    
class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = GATConv(1, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return x


def label_to_score(labels, targets):
    loss_mse = torch.nn.MSELoss(reduction='mean')
    return loss_mse(labels.float(), targets.float())
    return mean_squared_error(targets, labels)




class gat_net(torch.nn.Module):
    def __init__(self, num_node_features, dropout, conv_dim, device):
        # num_node + 1
        super(gat_net, self).__init__()
        self.device = device
        self.dropout = dropout
        self.num_node_features = num_node_features
        # self.node_feats = nn.Embedding(num_nodes, num_node_features)
        topic_dim = 130
        au_dim = 37
        paper_dim = 289
        self.topic_fc = nn.Linear(topic_dim, num_node_features)
        self.au_fc = nn.Linear(au_dim, num_node_features)
        self.paper_fc = nn.Linear(paper_dim, num_node_features)
        # 卷积
        self.conv1_dim = conv_dim
        head = 1
        self.conv1 = GATConv(num_node_features, self.conv1_dim, heads=head, dropout=self.dropout)
        # self.conv2 = GATConv(self.conv1_dim, self.conv2_dim, heads=head, concat=False, dropout=self.dropout)

        # mlp
        fc_dim1 = 2*self.conv1_dim*head
        fc_dim2 = int(fc_dim1 / 2)
        fc_dim3 = 1 # 1
        self.prednet1 = nn.Linear(3*self.conv1_dim*head, fc_dim1)
        self.prednet2 = nn.Linear(fc_dim1, fc_dim2)
        self.prednet3 = nn.Linear(fc_dim2, fc_dim3)
        for name, param in self.named_parameters():
            if 'weight' in name:
                print(name)
                nn.init.xavier_normal_(param)

    def forward(self, author_ids, topic_ids, auth_cnts, topic_cnts, paper_ids, id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, au_embs, paper_embs):
        bs = len(author_ids)
        pred_logits = []
        for idx in range(bs):
            author_id, topic_id, auth_cnt, topic_cnt, paper_id, id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb = author_ids[idx], topic_ids[idx], auth_cnts[idx], topic_cnts[idx], paper_ids[idx], id_maps[idx], edge_indexs[idx], paper_sets[idx], author_sets[idx], topic_sets[idx], topic_embs[idx], au_embs[idx], paper_embs[idx]
            # feat, edge_index = data.x, data.edge_index
            feat = torch.zeros(len(id_map)+1, self.num_node_features).to(self.device)
            topic_f = self.topic_fc(topic_emb.to(self.device))
            au_f = self.au_fc(au_emb.to(self.device))
            paper_f = self.paper_fc(paper_emb.to(self.device))
            feat[topic_set] = topic_f
            feat[author_set] = au_f
            feat[paper_set] = paper_f
            # data_graph = Data(x=feat, edge_index=edge_index)
            # author_ids [bs, 6]
            # topic_ids [bs, 12]
            feat = self.conv1(feat, edge_index.to(self.device))
            feat = F.relu(feat)
            feat = F.dropout(feat, p=self.dropout, training=self.training)
            feat[0] = 0.00001
            # mlp 
            topic_emb = feat[topic_id] # bs, 12, emb
            author_emb = feat[author_id] # bs, 6, emb
            topic_emb = topic_emb.sum(0) / (0.00001+topic_cnt)# avg bs, emb

            # author_emb = author_emb.sum(0) / (0.00001+auth_cnt)# avg bs, emb
            author_emb = torch.cat([author_emb[0,:], author_emb[1:,:].sum(0)/(-1.00001+auth_cnt)], 0)

            # author_emb = author_emb.view(author_emb.shape[0], -1) # bs ,6*emb
            # print(author_emb.shape, topic_emb.shape)
            cat_emb = torch.cat([author_emb, topic_emb], 0)
            pred_logits.append(cat_emb)

        cat_emb = torch.stack(pred_logits, 0)
        output = self.prednet1(cat_emb)
        output = torch.tanh(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.prednet2(output) # [bs, fc_dim2]
        output = torch.tanh(output)
        output = self.prednet3(output) # [bs, fc_dim2]

        # 分类
        # return output
        # output = F.dropout(output, p=self.dropout, training=self.training)

        # output = torch.argmax(output, dim=1)
        # output = torch.relu(output).view(-1)
        # output = torch.log(output)
        
        # output = torch.log(output)
        # output = torch.sigmoid(output)
        
        # output = F.dropout(output, p=self.dropout, training=self.training)

        return output.view(-1)
    def save_model(self, ep, note):
        torch.save(self.state_dict(), 'model/gat_parameter_{}_ep{}.pkl'.format(note, ep))
        print('model saved')
    def load_model(self, path):
        state_dict = torch.load(path)
        load_dict = {k:v for k, v in state_dict.items()}
        self.load_state_dict(load_dict)
        

class sage_net(torch.nn.Module):
    def __init__(self, num_node_features, dropout, conv_dim, device):
        # num_node + 1
        super(sage_net, self).__init__()
        self.device = device
        self.dropout = dropout
        self.num_node_features = num_node_features
        # self.node_feats = nn.Embedding(num_nodes, num_node_features)
        topic_dim = 130
        au_dim = 37
        paper_dim = 289
        self.topic_fc = nn.Linear(topic_dim, num_node_features)
        self.au_fc = nn.Linear(au_dim, num_node_features)
        self.paper_fc = nn.Linear(paper_dim, num_node_features)
        # 卷积
        self.conv1_dim = conv_dim
        self.conv1 = SAGEConv(num_node_features, self.conv1_dim)
        # self.conv2 = SAGEConv(self.conv1_dim, self.conv2_dim)

        # mlp
        fc_dim1 = 2*self.conv1_dim
        fc_dim2 = int(fc_dim1 / 2)
        fc_dim3 = 1 # 1
        self.prednet1 = nn.Linear(3*self.conv1_dim, fc_dim1)
        self.prednet2 = nn.Linear(fc_dim1, fc_dim2)
        self.prednet3 = nn.Linear(fc_dim2, fc_dim3)
        for name, param in self.named_parameters():
            if 'weight' in name:
                print(name)
                nn.init.xavier_normal_(param)

    def forward(self, author_ids, topic_ids, auth_cnts, topic_cnts, paper_ids, id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, au_embs, paper_embs):
        bs = len(author_ids)
        pred_logits = []
        for idx in range(bs):
            author_id, topic_id, auth_cnt, topic_cnt, paper_id, id_map, edge_index, paper_set, author_set, topic_set, topic_emb, au_emb, paper_emb = author_ids[idx], topic_ids[idx], auth_cnts[idx], topic_cnts[idx], paper_ids[idx], id_maps[idx], edge_indexs[idx], paper_sets[idx], author_sets[idx], topic_sets[idx], topic_embs[idx], au_embs[idx], paper_embs[idx]
            # feat, edge_index = data.x, data.edge_index
            feat = torch.zeros(len(id_map)+1, self.num_node_features).to(self.device)
            topic_f = self.topic_fc(topic_emb.to(self.device))
            au_f = self.au_fc(au_emb.to(self.device))
            paper_f = self.paper_fc(paper_emb.to(self.device))
            feat[topic_set] = topic_f
            feat[author_set] = au_f
            feat[paper_set] = paper_f
            # data_graph = Data(x=feat, edge_index=edge_index)
            # author_ids [bs, 6]
            # topic_ids [bs, 12]
            feat = self.conv1(feat, edge_index.to(self.device))
            feat = F.relu(feat)
            feat = F.dropout(feat, p=self.dropout, training=self.training)
            feat[0] = 0.00001
            # mlp 
            topic_emb = feat[topic_id] # bs, 12, emb
            author_emb = feat[author_id] # bs, 6, emb
            topic_emb = topic_emb.sum(0) / (0.00001+topic_cnt)# avg bs, emb

            # author_emb = author_emb.sum(0) / (0.00001+auth_cnt)# avg bs, emb
            author_emb = torch.cat([author_emb[0,:], author_emb[1:,:].sum(0)/(-1.00001+auth_cnt)], 0)

            # author_emb = author_emb.view(author_emb.shape[0], -1) # bs ,6*emb
            # print(author_emb.shape, topic_emb.shape)
            cat_emb = torch.cat([author_emb, topic_emb], 0)
            pred_logits.append(cat_emb)

        cat_emb = torch.stack(pred_logits, 0)
        output = self.prednet1(cat_emb)
        output = torch.tanh(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.prednet2(output) # [bs, fc_dim2]
        output = torch.tanh(output)
        output = self.prednet3(output) # [bs, fc_dim2]

        # 分类
        # return output
        # output = F.dropout(output, p=self.dropout, training=self.training)

        # output = torch.argmax(output, dim=1)
        # output = torch.relu(output).view(-1)
        # output = torch.log(output)
        
        # output = torch.log(output)
        # output = torch.sigmoid(output)
        
        # output = F.dropout(output, p=self.dropout, training=self.training)

        return output.view(-1)
    def save_model(self, ep, note):
        torch.save(self.state_dict(), 'model/sage_parameter_{}_ep{}.pkl'.format(note, ep))
        print('model saved')
        
    def load_model(self, path):
        state_dict = torch.load(path)
        load_dict = {k:v for k, v in state_dict.items()}
        self.load_state_dict(load_dict)



class deepwalk_net(torch.nn.Module):
    def __init__(self, num_node_features, dropout, device):
        # num_node + 1
        super(deepwalk_net, self).__init__()
        self.device = device
        self.dropout = dropout
        self.num_node_features = num_node_features
        # self.node_feats = nn.Embedding(num_nodes, num_node_features)

        # mlp
        fc_dim1 = 2*self.num_node_features
        fc_dim2 = self.num_node_features
        fc_dim3 = 1
        self.prednet1 = nn.Linear(3*self.num_node_features, fc_dim1)
        self.prednet2 = nn.Linear(fc_dim1, fc_dim2)
        self.prednet3 = nn.Linear(fc_dim2, fc_dim3)
        for name, param in self.named_parameters():
            if 'weight' in name:
                print(name)
                nn.init.xavier_normal_(param)

    def forward(self, authors_embs, topics_embs, auth_cnts, topic_cnts, paper_ids):
        # authors_embs, topics_embs, targets, auth_cnts, topic_cnts, paper_ids
        bs = len(authors_embs)
        pred_logits = []
        for _ in range(bs):
            authors_emb, topics_emb, auth_cnt, topic_cnt, paper_id = authors_embs[0].to(self.device), topics_embs[0].to(self.device), auth_cnts[0], topic_cnts[0], paper_ids[0]
            topic_emb = topics_emb.sum(0) / (0.00001+topic_cnt)# avg bs, emb
            author_emb = torch.cat([authors_emb[0,:], authors_emb[1:,:].sum(0)/(-1.00001+auth_cnt)], 0)
            cat_emb = torch.cat([author_emb, topic_emb], 0)
            pred_logits.append(cat_emb)

        cat_emb = torch.stack(pred_logits, 0)
        cat_emb = torch.tanh(cat_emb)
        # cat_emb = F.dropout(cat_emb, p=self.dropout, training=self.training)
        output = self.prednet1(cat_emb)
        output = torch.tanh(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.prednet2(output) # [bs, fc_dim2]
        output = torch.tanh(output)
        # output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.prednet3(output) # [bs, fc_dim2]

        return output.view(-1)
    def save_model(self, ep, note):
        torch.save(self.state_dict(), 'model/deepwalk_parameter_{}_ep{}.pkl'.format(note, ep))
        # print('model saved')

    def load_model(self, path):
        state_dict = torch.load(path)
        load_dict = {k:v for k, v in state_dict.items()}
        self.load_state_dict(load_dict)


if __name__ == "__main__":

    CONFIG = yaml.load(open('./config.yml', 'r'), Loader=yaml.Loader)
    model_name = CONFIG['model'] # 需要修改
    print(model_name)
    print('cite_year:', CONFIG['cite_year'])
    for m in CONFIG[model_name]:
        print(m, CONFIG[model_name][m])
    # os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    device = torch.device(CONFIG['device'])
    if model_name == 'deepwalk':
        dataset = train_dataset_deepwalk(max_author=CONFIG['max_author'], max_topic=CONFIG['max_topic'], cite_year=CONFIG['cite_year'], task='train')
    else:
        dataset = train_dataset(max_author=CONFIG['max_author'], max_topic=CONFIG['max_topic'], cite_year=CONFIG['cite_year'], task='train')
    # loader = DataLoader(dataset, batch_size=CONFIG[model_name]['batch_size'], num_workers=5)

    # 生成Data图结构 从1开始
    print('loading the graph ...')
    sub_adj = pickle.load(open("graph/sub_adj.pkl", "rb" ))
    edge_index = torch.LongTensor(sub_adj)

    # x = torch.zeros(dataset.num_nodes+1, CONFIG[model_name]['feats_num'])
    # data = Data(x=x, edge_index=edge_index)
    # data = Data(edge_index=edge_index)
    # data.num_node_features = dataset.num_nodes+1
    # print('num_node: {}, num_feats: {}'.format(data.num_nodes-1, data.num_node_features))

    # 加载模型
    # model = gcn_net(data.num_node_features, CONFIG[model_name]['dropout'])
    if model_name == 'gat':
        model = gat_net(CONFIG[model_name]['feats_num'], CONFIG[model_name]['dropout'], CONFIG[model_name]['conv_dim'], device)
    elif model_name == 'gcn':
        model = gcn_net(CONFIG[model_name]['feats_num'], CONFIG[model_name]['dropout'], CONFIG[model_name]['conv_dim'], device)
    elif model_name == 'sage':
        model = sage_net(CONFIG[model_name]['feats_num'], CONFIG[model_name]['dropout'], CONFIG[model_name]['conv_dim'], device)
    elif model_name == 'deepwalk':
        model = deepwalk_net(CONFIG[model_name]['feats_num'], CONFIG[model_name]['dropout'], device)


    # device_ids
    # device_ids = [0,1]
    # model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG[model_name]['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                    step_size=CONFIG[model_name]['step_size'], gamma=CONFIG[model_name]['gamma'])
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    #网络训练
    log_interval = CONFIG['log_interval']
    # data = data.to(device)
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss(reduction='mean')
    # if CONFIG['max_author'] == 'rmse':

    # elif CONFIG['max_author'] == 'acc':
    all_rmse, all_acc = [], []
    for epoch in range(1, CONFIG[model_name]['epoch']+1):
        model.train()
        dataset.switch_task('train')
        loader = DataLoader(dataset, shuffle=True, batch_size=CONFIG[model_name]['batch_size'], num_workers=1, collate_fn=CustomBatch())
        for i, one_data in enumerate(loader):
            optimizer.zero_grad()
            if model_name == 'deepwalk':
                authors_embs, topics_embs, targets, auth_cnts, topic_cnts, paper_ids = one_data
                modelout = model(authors_embs, topics_embs, auth_cnts, topic_cnts, paper_ids)
            else:
                author_ids, topic_ids, targets, auth_cnts, topic_cnts, paper_ids,\
                            id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, \
                            au_embs, paper_embs = one_data
                modelout = model(author_ids, topic_ids, auth_cnts, topic_cnts, paper_ids,
                            id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, 
                            au_embs, paper_embs)
            # print(type(topic_embs))
            # print(type(topic_embs[0]))

            loss = loss_mse(modelout, torch.log(torch.FloatTensor(targets)).to(device))
            pred_cite = np.exp(tensor_to_numpy(modelout))-1
            # print(pred_cite)
            # print(modelout)
            # print(np.array(targets)-1)
            # print(np.logical_and(targets >= 0.5*pred_cite, targets <= 1.5*pred_cite))
            acc = np.sum(np.logical_and(np.array(targets)-1 >= 0.5*pred_cite, np.array(targets)-1 <= 1.5*pred_cite)) / len(targets)
            # loss = loss_ce(modelout, labels.long().to(device))
            # pred_labels = torch.argmax(modelout, dim=1)
            # print(label_to_score(pred_labels, torch.log(targets.float()).to(device)))
            # print('{} : {}'.format(i, pred_labels))
            # print('{} : {}'.format(i, torch.log(targets.float())))

            loss.backward()
            optimizer.step()
            scheduler.step()
            # if i % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_rmse: {:.6f},Acc: {:.6f}'.format(
            #         epoch, (i+1) * loader.batch_size, len(loader.dataset),
            #         100. * (i+1) / len(loader), torch.sqrt(loss), acc))
            
        # 保存模型
        model.save_model(ep=epoch, note='year{}'.format(CONFIG['cite_year']))
        #测试
        all_preds, all_targets_log, all_targets = [], [], []
        model.eval()
        dataset.switch_task('test')
        loader = DataLoader(dataset, shuffle=True, batch_size=CONFIG[model_name]['batch_size'], num_workers=1, collate_fn=CustomBatch())
        with torch.no_grad():
            for one_data in loader:               
                if model_name == 'deepwalk':
                    authors_embs, topics_embs, targets, auth_cnts, topic_cnts, paper_ids = one_data
                    modelout = model(authors_embs, topics_embs, auth_cnts, topic_cnts, paper_ids)
                else:
                    author_ids, topic_ids, targets, auth_cnts, topic_cnts, paper_ids,\
                        id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, \
                        au_embs, paper_embs = one_data
                    modelout = model(author_ids, topic_ids, auth_cnts, topic_cnts, paper_ids,
                                id_maps, edge_indexs, paper_sets, author_sets, topic_sets, topic_embs, 
                                au_embs, paper_embs)

                all_preds += tensor_to_numpy(modelout).tolist()        
                all_targets_log += tensor_to_numpy(torch.log(torch.FloatTensor(targets))).tolist()  
                all_targets += tensor_to_numpy(torch.FloatTensor(targets)).tolist() 

        all_targets = np.array(all_targets)
        all_targets_log = np.array(all_targets_log)
        all_preds = np.array(all_preds)

        rmse = np.sqrt(mean_squared_error(all_targets_log, all_preds))
        pred_cite = np.exp(all_preds)-1
        acc = np.sum(np.logical_and(all_targets-1 >= 0.5*pred_cite, all_targets-1 <= 1.5*pred_cite)) / len(all_targets)
        all_rmse.append(rmse)
        all_acc.append(acc)
        # print(all_targets, all_targets_log, pred_cite) 

        
            # _, pred = model(data).max(dim=1)
            # correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            # acc = correct / data.test_mask.sum().item()
        print('epoch: {} test: RMSE: {:.4f}, ACC: {:.4f}'.format(epoch, rmse, acc))

    cat_log_path = 'result/{}.txt'.format(model_name)

    f = open(cat_log_path, 'a')
    f.write("cite_year: {}, ".format(CONFIG['cite_year']))
    for m in CONFIG[model_name]:
        f.write("{}: {}, ".format(m, CONFIG[model_name][m]))
    f.write('\n')
    f.write('rmse\t{}\n'.format('\t'.join('%s' %x for x in all_rmse)))
    f.write('acc\t{}\n'.format('\t'.join('%s' %x for x in all_acc)))
    f.close()
