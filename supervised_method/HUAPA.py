import torch
import torch.nn as nn
import torch.nn.functional as f
import BiLSTMAttn


class HUAPA(nn.Module):
    def __init__(self, sen_embeddings, user_embeddings, product_embeddings, input_dim, hid_word_dim, hid_sen_dim,
                 user_dim, product_dim, num_layers,
                 att_word_output_dim, att_sen_output_dim, classes, max_len=40, dropout=0.5):
        # 在这里初始化 层
        super(HUAPA, self).__init__()
        self.single_branch_user = BiLSTMAttn.BiLSTMAttn(sen_embeddings, user_embeddings, input_dim, hid_word_dim,
                                                        hid_sen_dim,
                                                        user_dim, num_layers,
                                                        att_word_output_dim, att_sen_output_dim, max_len=max_len,
                                                        dropout=dropout)
        self.single_branch_product = BiLSTMAttn.BiLSTMAttn(sen_embeddings, product_embeddings, input_dim, hid_word_dim,
                                                           hid_sen_dim,
                                                           product_dim, num_layers,
                                                           att_word_output_dim, att_sen_output_dim, max_len=max_len,
                                                           dropout=dropout)
        self.classes = classes
        self.C = len(self.classes)
        # todo: 目前不能完全确认这个文章的意思，是不是将 d 通过全连接层映射成 C维，主要觉得 so weird
        self.linear_1 = nn.Linear(4 * hid_sen_dim, self.C)  # classes_num 也是b的维度，W的行数
        self.linear_2 = nn.Linear(2 * hid_sen_dim, self.C)  # classes_num 也是b^u的维度，W^u的行数
        self.linear_3 = nn.Linear(2 * hid_sen_dim, self.C)  # classes_num 也是b^p的维度，W^p的行数

    def forward(self, docs_embeddings, user_embeddings, product_embeddings):
        """

        :param docs_embeddings: 所有的 reviews 的 embeddings (即documents)   应该是一个三维的，通过索引去embedding容器中得到嵌入
        :param docs_classes: 所有 reviews 的类别以及它们的索引 ，就是一个一维数组   eg. [c1,c2,c5,c1,...]
        :param user_embeddings:
        :param product_embeddings:
        :return:
        """
        ''' # 计算三个p以及三个d'''
        # 把所有计算出来的 d p 全部装在一起，方便后面的loss的计算
        d_user, d_pro, d, p, p_u, p_p = torch.nn.Parameter(torch.zeros([], dtype=torch.double))

        for sen_batch in docs_embeddings:
            # 加 _d 后缀表示这是一篇 document 的结果
            d_user_d = self.single_branch_user(sen_batch, user_embeddings)  # (1, 2 * hid_sen_dim, 1)
            d_pro_d = self.single_branch_product(sen_batch, product_embeddings)  # (1, 2 * hid_sen_dim, 1)
            d_d = torch.cat((d_user, d_pro), 1)  # (1, 4 * hid_sen_dim, 1)  todo:论文上d^u默认的是列向量？
            p_d = f.softmax(self.linear_1(d_d))
            p_u_d = f.softmax(self.linear_2(d_user_d))
            p_p_d = f.softmax(self.linear_3(d_pro_d))

            torch.cat((d_user, d_user_d), 0)
            torch.cat((d_pro, d_pro_d), 0)
            torch.cat((d, d_d), 0)
            torch.cat((p, p_d), 0)
            torch.cat((p_u, p_u_d), 0)
            torch.cat((p_p, p_p_d), 0)

        return d_user, d_pro, d, p, p_u, p_p
