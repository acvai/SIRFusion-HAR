"""
line 106 + line 108 have changed:
"""
r"""
Contains a PyTorch model fusing IR and pose data for improved classification. Also contains a helper function which
normalizes pose and IR tensors.

"""
import torch
from torch import nn
# from src.models.torchvision_models import *
from torchvision_models import *
import torchvision.models as models
import torch.nn.functional as F

# Custom imports
from src.models.utils import *
# from utils import *  ## for laptop_test

#### SGN added:
# from src.models.SGN_custom import SGN

#### STGCN added:
from src.models.stgcn_utils.tgcn import ConvTemporalGraphical
from src.models.stgcn_utils.graph import Graph
from src.models.st_gcn_for_Fusion import Model as ST_GCN
from src.models.st_gcn_for_Fusion import ST_Joint_Att
####

class FUSION(nn.Module):
    r"""This model is built on three submodules. The first is called a "pose module", which takes a skeleton sequence
    mapped to an image an outputs a 512-long feature vector. The second one is an "IR module", which takes an IR sequence
    and outputs a 512-long feature vector. The third one is a "classification module", which combines the 2 feature
    vectors (concatenation) and predicts a class via an MLP. This model can achieve over 90% accuracy on both
    benchmarks of the NTU RGB+D (60) dataset.

    Attributes:
        - **use_pose** (bool): Include skeleton data
        - **use_ir** (bool): Include IR data
        - **pose_net** (PyTorch model): Pretrained ResNet-18. Only exists if **use_pose** is True.
        - **ir_net** (PyTorch model): Pretrained R(2+1)D-18. Only exists if **use_ir** is True.
        - **class_mlp** (PyTorch model): Classification MLP. Input size is adjusted depending on the modules used.
          Input size is 512 if only one module is used, 1024 for two modules.

    Methods:
        *forward(X)*: Forward step. X contains pose/IR data

    """
    def __init__(self, use_pose, use_ir, pretrained, fusion_scheme, batch_size):
        super(FUSION, self).__init__()

        # Parts of the network to activate
        self.use_pose = use_pose
        self.use_ir = use_ir
        self.fusion_scheme = fusion_scheme

        # Pretrained pose network
        if use_pose:
            # self.pose_net = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
            # self.pose_net = SGN(60, 'NTU', 224, 1, batch_size) #SGN(args.num_classes, args.dataset, args.seg, args)  ### <---- this works ----
            ### stgcn model
            graph_args = {'layout': 'ntu-rgb+d'}
            ## note: in_channels=16 (16: embed() in stgcn-new-model with 3xInputs and embeding) | in_channels used to be = 3 |
            self.pose_net = ST_GCN(in_channels=16, num_class=60, graph_args=graph_args, edge_importance_weighting=True)


        # Pretrained IR network
        if use_ir:
            self.ir_net = nn.Sequential(*list(r2plus1d_18(pretrained=pretrained).children())[:-1])

        # Compute number of classification MLP input features
        mlp_input_features = 512
        if use_ir and use_pose:
            if self.fusion_scheme == "CONCAT":
                mlp_input_features = 2 * 512
            elif self.fusion_scheme == "SUM":
                mlp_input_features = 512
            elif self.fusion_scheme == "MULT":
                mlp_input_features = 512
            elif self.fusion_scheme == "AVG":
                mlp_input_features = 512
            elif self.fusion_scheme == "MAX":
                mlp_input_features = 512
            elif self.fusion_scheme == "CONV":
                mlp_input_features = 512
                self.conv_features = nn.Conv2d(1, 1, kernel_size=(1, 2))		
            else:
                print("Fusion scheme not recognized. Exiting ...")
                exit()

        # Classification MLP
        self.class_mlp = nn.Sequential(
            nn.BatchNorm1d(mlp_input_features),
            nn.Linear(mlp_input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 60)
        )

        ### added
        self.cnn1 = cnn_reshape(512, 500, True)
        self.cnn2 = cnn_reshape(1, 256, True)
        self.cnn3 = cnn_reshape(256, 512, True)
        ### added attention
        stgcn_out_channel = 256
        self.st_att = ST_Joint_Att(channel=stgcn_out_channel, reduct_ratio=2, bias=True)



    def forward(self, X):
        r"""Forward step of the FUSION model. Input X contains a list of 2 tensors containing pose and IR data. The
        input is already normalized as specified in the PyTorch pretrained vision models documentation, using the
        *prime_X_fusion* function. Each tensor is then passed to its corresponding module. The 2 feature vectors are
        concatenated, then fed to the classification module (MLP) which then outputs a prediction.

        Inputs:
            **X** (list of PyTorch tensors): Contains the following tensors:
                - **X_skeleton** (PyTorch tensor): pose images of shape `(batch_size, 3, 224, 224)` if **use_pose** is
                  True. Else, tensor = None.
                - **X_ir** (PyTorch tensor): IR sequences of shape `(batch_size, 3, seq_len, 112, 112)` if **use_ir** is
                  True. Else, tensor = None

        Outputs:
            **pred** (PyTorch tensor): Contains the the log-Softmax normalized predictions of shape
            `(batch_size, n_classes=60)`

        """
        X_skeleton = X[0]  # shape (batch_size, 3, 224, 224) or None
        X_ir = X[1]  # shape(batch_size, 3, seq_len, 112, 112) or None

        # Compute each data stream
        if self.use_pose:
            ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< x_f added
            # out_pose = self.pose_net(X_skeleton)[:, :, 0, 0]  # shape (batch_size, 512)
            out_pose, x_f = self.pose_net(X_skeleton)
            out_pose = out_pose[:, :, 0, 0]  # shape (batch_size, 512)
        if self.use_ir:
            out_ir = self.ir_net(X_ir)
            out_ir_ = out_ir[:, :, :, :, 0]  # shape (batch_size, 512, 1, 1)
            out_ir = out_ir[:, :, 0, 0, 0]  # shape (batch_size, 512)
            # out_ir = self.ir_net(X_ir.permute(0, 2, 1, 3, 4))[:, :, 0, 0, 0]  # shape (batch_size, 512)   ###-------------------------------< changed
            ### added:
            out_ir_ = self.cnn1(out_ir_)
            bs = out_ir.size()[0]
            out_ir_ = out_ir_.view(bs, 1, 20, 25)
            out_ir_ = self.cnn2(out_ir_) ##(N, 256, 20, 25)

        ### added: intermidiate fusion:
        fuse = x_f + out_ir_
        fuse = self.st_att(fuse)
        fuse = self.cnn3(fuse) # N, 512, 20, 25
        # # global pooling
        fuse = F.avg_pool2d(fuse, fuse.size()[2:]) #[N, 512, 1, 1]
        fuse = fuse[:, :, 0, 0] #[N, 512] like out_ir and out_pose sizes in 'SUM' scheme


        # Create feature vector
        if self.use_pose and not self.use_ir:
            features = out_pose
        elif not self.use_pose and self.use_ir:
            features = out_ir
        elif self.use_pose and self.use_ir:
            if self.fusion_scheme == "CONCAT":
                features = torch.cat([out_pose, out_ir], dim=1)
            elif self.fusion_scheme == "SUM":
                # features = out_pose + out_ir
                ### added
                features = out_ir 	#out_pose + out_ir + fuse
            elif self.fusion_scheme == "MULT":
                features = out_pose * out_ir
            elif self.fusion_scheme == "AVG":
                features = out_pose + out_ir / 2
            elif self.fusion_scheme == "MAX":
                features = torch.max(out_pose, out_ir)
            elif self.fusion_scheme == "CONV":
                features = torch.stack([out_pose, out_ir], dim=2).unsqueeze(1)
                features = self.conv_features(features).squeeze(1).squeeze(2)

        pred = self.class_mlp(features)  # shape (batch_size, 60)


        pred = F.softmax(pred, dim=1)

        return torch.log(pred + 1e-12)


def prime_X_fusion(X, use_pose, use_ir):
    r"""Normalizes X (list of tensors) as defined in the pretrained Torchvision models documentation. **Note** that
    **X_ir** is reshaped in this function.

    Inputs:
        - **X** (list of PyTorch tensors): Contains the following tensors:
            - **X_skeleton** (PyTorch tensor): pose images of shape `(batch_size, 3, 224, 224)` if **use_pose** is
              True. Else, tensor = -1.
            - **X_ir** (PyTorch tensor): IR sequences of shape `(batch_size, seq_len, 3, 112, 112)` if **use_ir** is
              True. Else, tensor = -1
        - **use_pose** (bool): Include skeleton data
        - **use_ir** (bool): Include IR data

    Outputs:
        **X** (list of PyTorch tensors): Contains the following tensors:
            - **X_skeleton** (PyTorch tensor): pose images of shape `(batch_size, 3, 224, 224)` if **use_pose** is
              True. Else, tensor = None.
            - **X_ir** (PyTorch tensor): IR sequences of shape `(batch_size, 3, seq_len, 112, 112)` if **use_ir** is
              True. Else, tensor = None
    """
    """
    custom model:
        Inputs:
            X_skeleton: n, c, t, v, m
            IR : not changed
        
    """
    if use_pose:
        X_skeleton = X[0] / 255.0  # shape (batch_size, 3, 224, 224)

        # Normalize X_skeleton
        normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])  # [[mean], [std]]
        X_skeleton = ((X_skeleton.permute(0, 2, 3, 1) - normalize_values[0]) / normalize_values[1]).permute(0, 3, 1, 2)

        if not use_ir:
            return [X_skeleton.to(device), None]

    if use_ir:
        X_ir = X[1] / 255.0 # shape (batch_size, seq_len, 3, 113, 112)

        # Normalize X
        normalize_values = torch.tensor([[0.43216, 0.394666, 0.37645],
                                         [0.22803, 0.22145, 0.216989]])  # [[mean], [std]]
        X_ir = ((X_ir.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

        if not use_pose:
            return [None, X_ir.permute(0, 2, 1, 3, 4).to(device)]

    return [X_skeleton.to(device), X_ir.permute(0, 2, 1, 3, 4).to(device)]

########################################## added for state_6-3-2:

def prime_X_fusion_3(X, use_pose, use_ir):
    """
    custom model for state_6-3-2: IR + stgcn + SUM-fusion-scheme :
        Inputs:
            X_skeleton: n, c, t, v, m
            IR : not changed

    """
    if use_pose:
        X_skeleton = X[0] #/ 255.0 => "we didn't add *255 in data_augmentation.py"   ## shape (batch_size, 3, 224, 224)

        # Normalize X_skeleton
        normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])  # [[mean], [std]]
        X_skeleton = ((X_skeleton.permute(0, 4, 2, 3, 1) - normalize_values[0]) / normalize_values[1]).permute(0, 4, 2, 3, 1)
        # org model:                     bs, 224, 224, c                                                     bs, c, 224, 224
        # bs <=> n       n, m, t, v, c -> 0, 4, 2, 3, 1                                      n, c, t, v, m -> 0, 4, 2, 3, 1
        if not use_ir:
            return [X_skeleton.to(device), None]

    if use_ir:
        X_ir = X[1] / 255.0  # shape (batch_size, seq_len, 3, 113, 112)

        # Normalize X
        normalize_values = torch.tensor([[0.43216, 0.394666, 0.37645],
                                         [0.22803, 0.22145, 0.216989]])  # [[mean], [std]]
        X_ir = ((X_ir.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

        if not use_pose:
            return [None, X_ir.permute(0, 2, 1, 3, 4).to(device)]

    return [X_skeleton.to(device), X_ir.permute(0, 2, 1, 3, 4).to(device)]


###########################################  added function to avoid the normalisation:

def prime_X_fusion_2(X, use_pose, use_ir):

    if use_pose:
        X_skeleton = X[0]     #X_skeleton = X[0] / 255.0  # shape (batch_size, 3, 224, 224)
        print('1)  n, c, t, v, m:', X_skeleton.size())
        assert False, '---------a------------'
        # Normalize X_skeleton
        # normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])  # [[mean], [std]]
        # X_skeleton = ((X_skeleton.permute(0, 2, 3, 1) - normalize_values[0]) / normalize_values[1]).permute(0, 3, 1, 2)
        #
        if not use_ir:
            return [X_skeleton.to(device), None]

    if use_ir:
        X_ir = X[1] / 255.0 # shape (batch_size, seq_len, 3, 113, 112)

        # Normalize X
        normalize_values = torch.tensor([[0.43216, 0.394666, 0.37645],
                                         [0.22803, 0.22145, 0.216989]])  # [[mean], [std]]
        X_ir = ((X_ir.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

        if not use_pose:
            return [None, X_ir.permute(0, 2, 1, 3, 4).to(device)]

    return [X_skeleton.to(device), X_ir.permute(0, 2, 1, 3, 4).to(device)]


############################### added cnn_reduce():
class cnn_reshape(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(cnn_reshape, self).__init__()
        self.cnn = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.cnn(x))




#########################################################
if __name__ == '__main__':
    ### """tested on laptop : works"""

    # fusion = FUSION(True, False, False, 'CONCAT', 4) ### skeleton only => works
    fusion = FUSION(True, True, False, 'SUM', 4) ### skeleton + IR => works
    print(fusion)
