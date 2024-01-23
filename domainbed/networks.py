# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Identity_Featurizer(nn.Module):
    """An identity layer"""
    def __init__(self, input_shape, hparams):
        super(Identity_Featurizer, self).__init__()
        self.n_outputs = input_shape 

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        pretrained = hparams['resnet_pretrained']
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=pretrained)
            self.n_outputs = 512
        elif hparams['resnet34']:
            self.network = torchvision.models.resnet34(pretrained=pretrained)
            self.n_outputs = 512
        elif hparams['resnet101']:
            self.network = torchvision.models.resnet101(pretrained=pretrained)
            self.n_outputs = 2048
        elif hparams['resnet152']:
            self.network = torchvision.models.resnet152(pretrained=pretrained)
            self.n_outputs = 2048
        else:
            self.network = torchvision.models.resnet50(pretrained=pretrained)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            print("Adapting number of channels from 3 to {}".format(nc))
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        if pretrained:
            self.freeze_bn()
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams['freeze_bn']:
            for m in self.network.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class CMNIST_Classifier(nn.Module):
    def __init__(self) -> None:
        super(CMNIST_Classifier, self).__init__()
        self.featurizer = MNIST_CNN(input_shape=[28, 28, 2])
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.featurizer(x)
        out = self.classifier(x)
        return out


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, hparams=None):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpool = nn.MaxPool2d((14, 14))

        self.dropout = nn.Identity() 
        if hparams and hparams['mnist_dropout'] is not None:
            self.dropout = nn.Dropout(hparams['mnist_dropout'])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        # NOTE: modified to maxpool
        x = self.avgpool(x)
        # x = self.maxpool(x)
        x = x.view(len(x), -1)

        # NOTE: add dropout
        self.dropout(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if hparams['no_featurizer']:
        return Identity_Featurizer(input_shape, hparams)
    if input_shape is None:
        return DistilBertFeaturizer.from_pretrained('distilbert-base-uncased')
    elif len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape, hparams)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

class simpleMLP(nn.Module):
    # default has 2 hidden layers = 3 linear layers
    def __init__(self, input_dim, output_dim, hidden_dim=500, num_layers=2):
        super(simpleMLP, self).__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(self.num_layers+1):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == self.num_layers:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            layers.append(layer)
            if i < self.num_layers:
                layers.append(nn.ReLU())
            else:
                # layers.append(nn.Sigmoid())
                continue
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


from transformers import DistilBertForSequenceClassification, DistilBertModel

class DistilBertClassifier(nn.Module):
    def __init__(self) -> None:
        super(DistilBertClassifier, self).__init__()
        self.featurizer = DistilBertFeaturizer.from_pretrained('distilbert-base-uncased')
        self.classifier = torch.nn.Linear(featurizer.d_out, 2)

    def forward(self, x):
        x = self.featurizer(x)
        out = self.classifier(x)
        return out


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_outputs = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output
