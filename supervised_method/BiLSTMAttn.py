import torch
import torch.nn as nn
import torch.nn.functional as f


# seq_lengths = torch.IntTensor([[10, 11, 13, 15], [101, 111, 131, 151]])
# seq_lengths.size()
# seq_lengths.permute(1, 0)
# print(seq_lengths)

# x = torch.tensor(
#     [[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
#      [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]], )
# y = torch.tensor([3, 4])
# z = torch.cat((x, y), 2)


class BiLSTMAttn(nn.Module):
    """
    Implementation of BiLSTM Concatenation for sentiment classification task
    """

    def __init__(self, sen_embeddings, info_embeddings, input_dim, hid_word_dim, hid_sen_dim, info_dim, num_layers,
                 att_word_output_dim, att_sen_output_dim, max_len=40, dropout=0.5):
        # 在这里初始化 层
        super(BiLSTMAttn, self).__init__()

        # 这里是初始化一些结构的，不要在这里输入数据，因为数据会变化的，这是用来初始化模型的，
        # 模型不是每输入一次数据就再初始化，因为要在模型上面更新参数
        self.batch_size = 0
        self.step_number = 0
        self.sen_emb = None

        # self.emb.weight = nn.Parameter(embeddings)        # 句子embedding看起来不需要更新吧。。。

        # info embedding 容器
        self.info_emb = nn.Embedding(num_embeddings=info_embeddings.size(0), embedding_dim=info_embeddings.size(1),
                                     padding_idx=0)

        # word encoder
        self.sen_len = max_len
        self.input_dim = input_dim
        self.hid_word_dim = hid_word_dim
        self.word_lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hid_word_dim, num_layers=num_layers,
                                 dropout=dropout,
                                 batch_first=True, bidirectional=True)
        # sentence encoder
        self.hid_sen_dim = hid_sen_dim
        self.word_lstm = nn.LSTM(input_size=self.hid_word_dim * 2, hidden_size=self.hid_sen_dim, num_layers=num_layers,
                                 dropout=dropout, batch_first=True, bidirectional=True)

        # attention
        self.tanh = nn.Tanh()
        self.info_dim = info_dim
        # word attention
        self.att_word_output_dim = att_word_output_dim  # b^u_w 的维度，也是 v^u_w的维度
        self.word_attn = nn.Linear(hid_word_dim * 2 + info_dim, att_word_output_dim)

        # sentence attention
        self.att_sen_output_dim = att_sen_output_dim  # b^u_s 的温度，也是v^u_s的维度
        self.sen_attn = nn.Linear(hid_sen_dim * 2 + info_dim, att_sen_output_dim)

    def attention(self, hidden_state, info_vec: torch.Tensor, attn_layer: nn.Linear, v_weight):
        """

        :param hidden_state: BiLSTM输出的hidden层  (batch_size, step_number , hidden_dim)
        :param info_vec: 信息 embedding   (info_vec_dim);   将会输入 user_vec 或者 product_vec
        :param attn_layer: 传入的注意力层，因为为了复用这个函数，所以将其抽象出来，用来分别应用到 word level和 sentence level
        :param v_weight:
        :return:
        """
        ''' # 处理输入数据的维度'''
        # hidden_state 和 info_vec 的维度很不一样   # 经过查 api 输入输出Linear的是三维的张量
        # 所以要将 info_vec (info_vec_dim) 转换成 (batch_size, step_number, info_vec_dim)
        '''     ## 扩展维度'''
        # 因为从embedding中拿出来是tensor , 注意别用成numpy和list的方法了 , 先增加维数,再cat进行堆叠
        info_vec = info_vec.unsqueeze(0).unsqueeze(0)
        # v_weight 本身是(att_sen_output_dim,1) 要转置且增加维数才能进行bmm，同时还要叠加维数
        v_weight = v_weight.permute(1, 0).unsqueeze(0)
        '''     ## 堆叠'''
        clone_info_vec = info_vec.clone()
        clone_v_weight = v_weight.clone()
        info_vec = torch.zeros_like(info_vec)
        v_weight = torch.zeros_like(v_weight)
        for _ in range(hidden_state.size(0)):  # 堆叠第2维
            info_vec = torch.cat((info_vec, clone_info_vec), 1)
            v_weight = torch.cat((v_weight, clone_v_weight), 1)

        clone_info_vec = info_vec.clone()
        clone_v_weight = v_weight.clone()
        info_vec = torch.zeros_like(info_vec)
        v_weight = torch.zeros_like(v_weight)
        for _ in range(hidden_state.size(1)):  # 堆叠第1维
            info_vec = torch.cat((info_vec, clone_info_vec), 0)
            v_weight = torch.cat((v_weight, clone_v_weight), 0)

        ''' # 求出attention权重'''
        attn = self.attn_layer(torch.cat((hidden_state, info_vec), 2), self.att_word_output_dim)  # 要指定最里层的axis
        attn = self.tanh(attn)  # (batch_size, step_number, att_*_output_dim)
        attn = torch.bmm(v_weight, attn)  # 之前将 v_weight 进行转置，注意，v_weight必须要是三维的，不然会报错
        attn = f.softmax(attn)
        return attn  # (batch_size, step_number, 1)

    def forward(self, sen_batch, embeddings):
        """
        这里定义 architecture的一支，这个类就只是一支分支而已
        :param embeddings: user embedding 或者是 product embedding
        :param sen_batch:
        :return:
        """
        ''' # Word Level '''
        '''     ## Word Representation'''
        '''         Embedding Layer | Padding | Sequence_length 40 '''
        # 句子的embedding容器
        self.batch_size = sen_batch.size(0)
        self.step_number = sen_batch.size(1)
        self.sen_emb = nn.Embedding(num_embeddings=self.batch_size, embedding_dim=self.step_number,
                                    padding_idx=0)  # 从这个直接就等于embedding_dim可以发现，应该是已经把padding都补充完整了
        sen_batch = self.sen_emb(sen_batch)
        batch_size = len(sen_batch)

        '''     ## BiLSTM Layer'''
        # -1 表示自动补充； 这里的 -1 最后算出来的维度是 sen_len*dim 一个句子的所有单词转成embedding后拼凑起来的维度
        # sen_outs 是最后一层lstm的每个词向量对应隐藏层的输出, 其与层数无关，只与序列长度相关
        # (hn,cn) = _  hn,cn是所有层最后一个隐藏元和记忆元的输出
        word_outs, (word_hidden, word_cell) = self.word_lstm(sen_batch.view(batch_size, -1, self.input_dim))
        '''     ## Attention Mechanism'''
        # 如果张量在内存中不连续，contiguous()可以让它变得连续
        # as for 2 * self.hidden_dim,根据BiLSTM的博客 https://zhuanlan.zhihu.com/p/47802053
        # 可以看出输出的是 正反两个LSTM输出的向量拼凑起来的  输出的是每个单词对应的hidden_dim，所以要resize 成2*hidden_dim
        word_hidden = word_hidden.contiguous().view(batch_size, -1, 2 * self.hid_word_dim)  # (batch, sen_len, 2*hid)
        v_word_weight = torch.nn.Parameter(torch.randn(self.att_word_output_dim))  # 要加入计算图
        # word_hidden:(batch, num_of_words, hidden_dim * 2) ;
        # word_attn:(batch_size, step_number, 1) ; num_of_words == step_number
        word_attn = self.attention(word_hidden, embeddings, self.word_attn, v_word_weight)
        s = torch.bmm(word_hidden.permute(0, 2, 1),
                      word_attn)  # (batch, hidden_dim*2, step_num)*(batch, step_num, 1)

        ''' # Sentence Level'''
        '''     ## BiLSTM Layer'''
        sen_outs, (sen_hidden, sen_cell) = self.sen_rnn(s.view(batch_size, 1, 2 * self.hid_word_dim))
        '''     ## Attention Mechanism'''
        sen_hidden = sen_hidden.contiguous().view(batch_size, -1, 2 * self.hid_sen_dim)  # (batch, 1, 2*hid)
        v_sen_weight = torch.nn.Parameter(torch.randn(self.att_sen_output_dim))  # 要加入计算图
        # sen_hidden:(batch, 1, 2 * hid_sen_dim); sen_attn:(batch, 1, 1)
        sen_attn = self.attention(sen_hidden, embeddings, self.sen_attn, v_sen_weight)
        d = torch.bmm(sen_hidden.permute(1, 2, 0),  # todo:sen_attn检查这里有没有问题 (2, 0, 1)，要是用view会怎么样
                      sen_attn.permute(1, 0, 2))  # (1, 2 * hid_sen_dim, batch) * (1, batch, 1)
        # 最后输出的是一个document的representation,输入的应该是很多个document
        return d
