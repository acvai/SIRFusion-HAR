import torch
from torch import nn
from models.cnn import cnn3d
from models import (cnn, C3DNet, resnet, resnet_amine, ResNetV2, ResNeXt, ResNeXtV2, WideResNet, PreActResNet,
        EfficientNet, DenseNet, ShuffleNet, ShuffleNetV2, SqueezeNet, MobileNet, MobileNetV2)

# from opts import parse_opts
# ### added:
# import argparse


def main(cnn_name, model_depth, n_classes, in_channels, sample_size):
 
    # simple CNN 
    if cnn_name == 'cnn':
        """
        3D simple cnn model
        """
        print(cnn_name)
        model = cnn3d()
    
    # C3D
    elif cnn_name == 'C3D':
        """
        "Learning spatiotemporal features with 3d convolutional networks." 
        """
        model = C3DNet.get_model(
            sample_size=sample_size,
            sample_duration=16,
            num_classes=n_classes,
            in_channels=1)

    # ResNet
    elif cnn_name == 'resnet':
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        model = resnet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)

############################################################ added a custom resnet with 2 outputs
    # ResNet_amine
    elif cnn_name == 'resnet_amine':
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        ### changed: >resnet.gen -> >resnet_amine.gen
        model = resnet_amine.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)
############################################################

    # ResNetV2
    elif cnn_name == 'ResNetV2':
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        model = ResNetV2.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)

    # ResNeXtV2
    elif cnn_name == 'ResNeXt':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = ResNeXt.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            in_channels=in_channels,
            sample_size=sample_size,
            sample_duration=16)
    
    # ResNeXtV2
    elif cnn_name == 'ResNeXtV2':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = ResNeXtV2.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels)

    # PreActResNet
    elif cnn_name == 'PreActResNet':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = PreActResNet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels)

    # WideResNet
    elif cnn_name == 'WideResNet':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = WideResNet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels)

    # DenseNet
    elif cnn_name == 'DenseNet':
        """
        3D resnet
        model_depth = [121, 169, 201]
        """
        model = DenseNet.generate_model(
            model_depth=model_depth,
            num_classes=n_classes,
            n_input_channels=in_channels)

    # SqueezeNet
    elif cnn_name == 'SqueezeNet':
        """
        SqueezeNet
        "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and 
        <0.5MB model size"
        """
        model = SqueezeNet.get_model(
            version=1.0,
            sample_size=sample_size,
            sample_duration=16,
            num_classes=n_classes,
            in_channels=in_channels)
   
    # ShuffleNetV2
    elif cnn_name == 'ShuffleNetV2':
        """
        ShuffleNetV2
        "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
        """
        model = ShuffleNetV2.get_model(
            sample_size=sample_size,
            num_classes=n_classes,
            width_mult=1.,
            in_channels=in_channels)

    # ShuffleNet
    elif cnn_name == 'ShuffleNet':
        """
        ShuffleNet
        """
        model = ShuffleNet.get_model(
            groups=3,
            num_classes=n_classes,
            in_channels=in_channels)

    # MobileNet
    elif cnn_name == 'MobileNet':
        """
        MobileNet
        "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" 
        """
        model = MobileNet.get_model(
            sample_size=sample_size,
            num_classes=n_classes,
            in_channels=in_channels)

    # MobileNetV2
    elif cnn_name == 'MobileNetV2':
        """
        MobileNet
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
        """
        model = MobileNetV2.get_model(
            sample_size=sample_size,
            num_classes=n_classes,
            in_channels=in_channels)
    
    # EfficientNet
    elif cnn_name == 'EfficientNet':
        """
        EfficientNet
        """
        model = EfficientNet3D.from_name(
            'efficientnet-b4', 
            override_params={'num_classes': n_classes}, 
            in_channels=in_channels)
    else:
        assert False, '-----model not recognized Looser !!!------'

    ############################################################## note: if using model.cuda() => input in >output = model(input) : need to be > input.cuda() first
    if torch.cuda.is_available():
        model.cuda()

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')
    parser.add_argument('--cnn_name', default='resnet_amine', type=str, help='cnn model names')                   ### modified: ResNet -> resnet
    parser.add_argument('--model_depth', default=18, type=str, help='model depth (18|34|50|101|152|200)')
    parser.add_argument('--n_classes', default=10, type=str, help='model output classes')
    parser.add_argument('--in_channels', default=1, type=str, help='model input channels (1|3)')
    parser.add_argument('--sample_size', default=112, type=str, help='image size')
    args = parser.parse_args()

    model = main(cnn_name=args.cnn_name,
                 model_depth=args.model_depth,
                 n_classes=args.n_classes,
                 in_channels=args.in_channels,
                 sample_size=args.sample_size
                )
                #sample_size = args.sample_size  #### line corrected

    ### model summary:
    # print(model)
    ### test input:
    n, c, seq, h, w = 3, 1, 20, 112, 112
    x = torch.rand(n, c, seq, h, w)
    x = x.cuda()
    # x_2 = torch.zeros_like(x)
    print(x.size())

    y = model(x)
    # print(f"outputs: {x.size()}, {y[0].size()}, {y[1].size()}, {y[2].size()}, {y[3].size()}, {y[4].size()}")
    print(f"outputs: {x.size()}, {y[0].size()}, {y[1].size()}")
    # print("output: ", y.size())
