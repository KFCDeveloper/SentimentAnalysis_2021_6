import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable


class BiLSTM(nn.Module):
    """
    Implementation of BiLSTM Concatenation for sentiment classification task
    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        # 在这里初始化 层
        super(BiLSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # sen encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.output = nn.Linear(2 * self.hidden_dim, output_dim)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        """

        :param rnn_outs:
        :param seq_lengths: 每个句子的长度
        :param batch_size:
        :param max_len: max_len 这个命名就很能说明问题，说明要补全参差不齐的句子
        :return:
        """
        # 这里resize成了四维， batch_size:句子个数; max_len:句子单词个数; 2:每个单词向量分成部分，因为BiLSTM是将它们合起来了的
        # -1: 正常的LSTM输出的hidden的长度
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)
        # (batch_size, max_len, 1, -1)
        # cuda() 将数据转移到GPU显存上，可是这里为啥要把 0 移动到GPU上？难道不移动会报错吗？？
        # 注意，维数没有变，还是四维，在每句话中，取每个单词对应的两个hidden 向量的第一个向量，
        # 那么就是 num of sen * num of word in sen * 1 * hidden
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        # 现在将三维的矩阵resize成了二维的，变成了 (num of sen * num of word in sen) * hidden
        # 相当于把所有的 正 向产生的hidden向量全都剥离出来，排成一排了
        fw_out = fw_out.view(batch_size * max_len, -1)
        # 同上，取第二个hidden向量
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        # 相当于把所有的 反 向产生的hidden向量全都剥离出来，排成一排了
        bw_out = bw_out.view(batch_size * max_len, -1)
        # pytorch会将 range(batch_size) 转成list  [0,max_len,2*max_len,..., (batch_size-1)*max_len]
        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()
        # seq_lengths是一个list，里面保存了每个句子的长度，这样子，fw_index就会定位到真实的每句话的最后一个单词，而不会是padding 0的部分
        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros  # 这是一个list
        # 提取出了每句话的第一个单词产生的hidden,所以输出了 batch_size 个元素的 list
        bw_out = torch.index_select(bw_out, 0, bw_index)
        # 这是一个有 batch_size 个元素的 list，每一个都是正向LSTM的每句话的最后一个单词对应的hidden 和 反向LSTM的每句话的第一个单词的对应hidden
        # 它们都是集成了整句话的信息的hidden，之前是每个单词的正反LSTM相应的两个hidden组合起来，且一共有 batch_size*max_length个
        # 现在每句话的最后一个的正向LSTM和第一个的反向LSTM，且一共只有 batch_size 个
        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """
        # 在这里定义结构，将层搭建起来
        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """
        """ Embedding Layer | Padding | Sequence_length 40 """
        sen_batch = self.emb(sen_batch)
        batch_size = len(sen_batch)
        """ Bi-LSTM Computation """
        # -1 表示自动补充； 这里的 -1 最后算出来的维度是 sen_len*dim 一个句子的所有单词转成embedding后拼凑起来的维度
        # sen_outs 是最后一层lstm的每个词向量对应隐藏层的输出, 其与层数无关，只与序列长度相关
        # (hn,cn) = _  hn,cn是所有层最后一个隐藏元和记忆元的输出
        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))

        # 如果张量在内存中不连续，contiguous()可以让它变得连续
        # as for 2 * self.hidden_dim,根据BiLSTM的博客 https://zhuanlan.zhihu.com/p/47802053
        # 可以看出输出的是 正反两个LSTM输出的向量拼凑起来的  输出的是每个单词对应的hidden_dim，所以要resize 成2*hidden_dim
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)
        """ Fetch the truly last hidden layer of both sides """
        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)
        representation = sentence_batch
        out = self.output(representation)
        out_prob = f.softmax(out.view(batch_size, -1))

        return out_prob
