import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import CoraGraphDataset
import dgl.function as fn

# 自己实现的一个gcn layer
class GCNlayer(nn.Module):
    def __init__(self,in_feats, out_feats):
        super(GCNlayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.Linear = nn.Linear(in_feats,out_feats)
        # 消息函数，接收edges，被用于在dgl内部生成表示一些边,
        # 内置消息函数默认 u 表示源节点， v表示目标节点，e表示边， 这些函数的参数表示 相应节点和边的输入和输出特征的字段名
        # 把源节点的 'h'特征 复制（保存）到  'm'字段
        self.gcn_msg = fn.copy_u(u= 'h',out = 'm')
        # 聚合函数接受一个参数nodes，它用来在DGL内部生成表示一些节点。 nodes.mailbox 用来访问节点收到的信息
        # 内置聚合函数， 接受两个参数， 一个指定mailbox中的字段名， 一个用于表示目标节点特征的字段名
        # 将message中的 'm'字段求和， 保存到'h'字段
        self.gcn_reduce = fn.sum(msg = 'm', out = 'h')



    def forward(self,g,feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(self.gcn_msg,self.gcn_reduce)
            h = g.ndata['h']  # h size is
            print(h.shape)
            res = self.Linear(h)
            print(res.shape)
            return res


class GCN( nn.Module ):
    def __init__(self,
                 g, #DGL的图对象
                 in_feats, #输入特征的维度
                 n_hidden, #隐层的特征维度
                 n_classes, #类别数
                 n_layers, #网络层数
                 activation, #激活函数
                 dropout #dropout系数
                 ):
        super( GCN, self ).__init__()
        self.g = g
        # self.layers = nn.ModuleList()
        # # 输入层
        # self.layers.append( GCNlayer( in_feats, n_hidden ))
        # # 隐层
        # for i in range(n_layers - 1):
        #     self.layers.append(GCNlayer(n_hidden, n_hidden ))
        # # 输出层
        # self.layers.append( GCNlayer( n_hidden, n_classes ) )
        self.GCN_layer1 = GCNlayer(in_feats, n_hidden)
        self.GCN_layer2 = GCNlayer(n_hidden, n_hidden)
        self.GCN_layer3 = GCNlayer(n_hidden, n_classes)

        # self.dropout = nn.Dropout(p = dropout)

    def forward( self,g, features ):
        x = F.relu(self.GCN_layer1(g,features))
        x = F.relu(self.GCN_layer2(g,x))
        x = F.relu(self.GCN_layer3(g,x))
        return x


        # h = features
        # for i, layer in enumerate( self.layers ):
        #     if i != 0:
        #         h = self.dropout( h )
        #     h = layer( self.g, h )
        # return h



def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g,features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(n_epochs=100, lr=1e-2, weight_decay=5e-4, n_hidden=16, n_layers=1, activation=F.relu , dropout=0.5):
    data = CoraGraphDataset()
    # 该数据集用于semi-supervised的节点分类任务

    g=data[0] # 2708个节点
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1] # features => [2708, 1433维的feature]
    n_classes = data.num_labels

    # train_mask：训练集的mask向量，标识哪些节点属于训练集。
    # val_mask：验证集的mask向量，标识哪些节点属于验证集。
    # test_mask：测试集的mask向量，表示哪些节点属于测试集。

    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                activation,
                dropout)

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( model.parameters(),
                                 lr = lr,
                                 weight_decay = weight_decay)
    for epoch in range( n_epochs ):
        model.train()
        logits = model(g, features ) # logits => [2708,7]
        loss = loss_fcn( logits[ train_mask ], labels[ train_mask ] )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(model, g, features, labels, val_mask)
        print("Epoch {} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, loss.item(), acc ))
    print()
    acc = evaluate(model, g,  features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

if __name__ == '__main__':
    train()