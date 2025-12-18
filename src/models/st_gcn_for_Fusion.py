import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from net.utils.tgcn import ConvTemporalGraphical
# from net.utils.graph import Graph
# from utils.tgcn import ConvTemporalGraphical
# from utils.graph import Graph
from stgcn_utils.tgcn import ConvTemporalGraphical
from stgcn_utils.graph import Graph



class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        ### changed : from in_channels -> C_
        C_ = 3
        self.data_bn = nn.BatchNorm1d(C_ * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 512, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        ### Attention added:
        stgcn_out_channel = 256
        self.st_att = ST_Joint_Att(channel=stgcn_out_channel, reduct_ratio=2, bias=True)

        ### embed from SGN
        self.pos_embed = embed(3, 16, bias=True)
        self.dif_embed = embed(3, 16, bias=True)
        self.dif_r_embed = embed(3, 16, bias=True)

    def forward(self, x):

        # # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        ### position
        pos = x                         # STGCN: n*m, c, t, v

"""	#modified {05/07/2024:19h} to test st-gcn on SIRFusion with only pos input:
        ### velocity
        dif = torch.zeros([N*M, C, T, V]).to('cuda:0')
        dif[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, 0:-1, :]
        # print('dif: ', dif.size(), x[:, :, 1:, :].size(), x[:, :, 0:-1, :].size())
        # assert False, '------a----------'
        # dif = torch.cat([dif.new(N*M, dif.size(1), V, 1).zero_(), dif], dim=-1)

        ### relative distance  | calculate relative distance between all joints and the tranc node as a reference:
        dif_r = torch.zeros([N*M, C, T, V]).to('cuda:0')
        for v in range(V):
            dif_r[:, :, :, v] = x[: ,:, :, v] - x[:, :, :, 1] ### Relative Position (ref is tranc)
"""
        ### embeding
        pos = self.pos_embed(pos)
"""        dif = self.dif_embed(dif)
        dif_r = self.dif_r_embed(dif_r)
"""
        ### the sum <=> Dynamic Representation like SGN
"""        x = pos + dif + dif_r                           # n*m, c=16, t, v
"""
        x = pos

        # # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        # print('1) x: ', x.size()); assert False, '_----sa---'  #(8=2*N, 512, 5, V=25)
        ### added:
        x = x.view(N, -1, T, V)  #[4, 256, 20, 25]        #print('x: ', x.size())
        x_f = x
	### commented to check if STGCN-without-st_att is better than the STGCN-with-st_att: {04/07/2024}
        #x = self.st_att(x)       #[4, 256, 20, 25]
        x = x.view(N, x.size()[1]*2, -1 , V)  #[N=4, 256*2=512, T/2=10, V=25]
        x = F.avg_pool2d(x, x.size()[2:]) #; print('x avg_pool: ', x.size())

        # # global pooling
        # x = F.avg_pool2d(x, x.size()[2:])           #; print('1): ', x.size(), x.view(N, M, -1, 1, 1).size())   #[8, 512, 1, 1] & [4, 2, 512, 1, 1]
        # x = x.view(N, M, -1, 1, 1).mean(dim=1)      #; print('2): ', x.size())                                  #[4, 512, 1, 1]
        # print('x out: ', x.size())

        # # prediction
        # x = self.fcn(x)
        # x = x.view(x.size(0), -1)

        return x, x_f

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A



############################################### added from EfficientGCN attentions.py | the function is modified to fit requiremnets:
class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(), #nn.Hardswish(), ### changed
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att

############################### added cnn_reduce():
class cnn_reshape(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(cnn_reshape, self).__init__()
        self.cnn = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.cnn(x))

################################################ added from SGN
class embed(nn.Module):
    def __init__(self, dim=3, dim1=16, bias=False):   ### set norm=False, to avoid norm_data() problem
        super(embed, self).__init__()

        # if norm:
        #     self.cnn = nn.Sequential(
        #         norm_data(dim),
        #         cnn1x1(dim, 64, bias=bias),
        #         nn.ReLU(),
        #         cnn1x1(64, dim1, bias=bias),
        #         nn.ReLU(),
        #     )
        # else:
        self.cnn = nn.Sequential(
            cnn1x1(dim, 9, bias=bias),
            nn.ReLU(),
            cnn1x1(9, dim1, bias=bias),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x



class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

################################################
if __name__ == '__main__':

    ### stgcn branch
    stgcn = st_gcn(in_channels=3, out_channels=64, kernel_size=(9, 1), stride=1)
    # print(stgcn)
    print('----------------------hello world----------------------')

    ### stgcn model
    graph_args = {'layout':'ntu-rgb+d'}
    model = Model(in_channels=3, num_class=60, graph_args=graph_args, edge_importance_weighting=True)
    # print(model)

    ### test input:
    N, C, T, V, M = 4, 3, 20, 25, 2
    input = torch.randn((N, C, T, V, M ))
    # output = model(input)
    # print(f'input: {input.size()} | output: {output.size()}')  ## output: [bs, 512, 1, 1] the wright size for FUSION-model


    ### attention
    stgcn_out_channel = 256
    st_att = ST_Joint_Att(channel=stgcn_out_channel, reduct_ratio=2, bias=True)
    print(st_att)
    input = torch.randn((N, stgcn_out_channel, T, V))
    output = st_att(input)
    print('output: ', output.size())

    ### test
    x1 = torch.rand(N, 512, 1, 1)
    cnn = cnn_reshape(512, 500, True)
    x1 = cnn(x1)
    x1 = x1.view(N, 1, 20, 25)
    cnn2 = cnn_reshape(1, 256, True)
    x2 = cnn2(x1)
    print(x1.size(), x2.size())

    # x2 = x1.view(N, -1, 20, 25)
    # print(x1.size(), x2.size())
