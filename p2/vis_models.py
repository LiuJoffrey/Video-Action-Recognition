import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, transforms, models

#__all__ = ['vgg16_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'
}


class VGG(nn.Module):

    def __init__(self, features, output_size=1274, batch_size=64):
        super(VGG, self).__init__()
        self.features = features
        self._initialize_weights()
        self.batch_size = batch_size

    def forward(self, x, length):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ATT_Classifier(nn.Module):

    def __init__(self):
        super(ATT_Classifier, self).__init__()
        
        self.linear = nn.Sequential(
            #TODO
            #nn.Linear(in_features=256*2, out_features=256),
            nn.Linear(in_features=256*2*2, out_features=256),
            # nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(True), 
            # nn.Linear(in_features=128, out_features=11)
        )

        self.output_layer = nn.Linear(in_features=128, out_features=11)

        self.att = nn.Sequential(
            nn.Linear(in_features=256*2, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU(True)
        )

        self.rnn = torch.nn.GRU(2048, 256, batch_first=True, bidirectional=True, num_layers=2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.5)

        # self._initialize_weights()

    def forward(self, x, length):
        assert( length.sum() == x.size(0))
        max_len = max(length)
        frame_feature = []
        start = 0
        for l in length:
            video = x[start:start+l]
            if video.size(0) < max_len:
                diff = max_len - video.size(0)
                pad = torch.zeros((diff, video.size(1), video.size(2), video.size(3))).type(torch.FloatTensor).cuda()
                video = torch.cat((video, pad), dim=0)
            video = video.view(max_len, -1).unsqueeze(0)

            frame_feature.append(video)
            start += l

        frame_feature = torch.cat(frame_feature, dim=0)
        
        # frame_feature, sorted_lengths, unsorted_idx = self.sort_sequences(frame_feature, length)
        # frame_feature = torch.nn.utils.rnn.pack_padded_sequence(
        #       frame_feature, sorted_lengths, batch_first=True)
        frame_feature, _ = self.rnn(frame_feature)
        # frame_feature, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #                         frame_feature, batch_first=True)

        # frame_feature = frame_feature.index_select(0, unsorted_idx)
        # frame_feature = self.dropout(frame_feature)
        

        att = self.att(frame_feature)
        att = self.softmax(att)
        frame_feature_att = torch.bmm(frame_feature.transpose(1,2),att).squeeze(-1)
        frame_feature_att = self.dropout(frame_feature_att)

        frame_feature_max = frame_feature.max(1)[0]
        frame_feature_max = self.dropout(frame_feature_max)
        

        frame_feature = torch.cat((frame_feature_att, frame_feature_max), dim=-1)

        

        # frame_feature = frame_feature[:,-1]
        # frame_feature = frame_feature.mean(dim=1)

        x = self.linear(frame_feature)
        x = self.output_layer(x)
        x = self.log_softmax(x)
        return x
        
    def sort_sequences(self,inputs, lengths):
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        _, unsorted_idx = sorted_idx.sort()
        return inputs[sorted_idx], lengths_sorted.cuda(), unsorted_idx.cuda()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        
        self.linear = nn.Sequential(
            #TODO
            #nn.Linear(in_features=256*2, out_features=256),
            nn.Linear(in_features=256*2, out_features=256),
            # nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(True), 
            # nn.Linear(in_features=128, out_features=11)
        )

        self.output_layer = nn.Linear(in_features=128, out_features=11)

        # self.att = nn.Sequential(
        #     nn.Linear(in_features=256*2, out_features=256),
        #     nn.Tanh(),
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.Tanh(),
        #     nn.Linear(in_features=128, out_features=1),
        #     nn.ReLU(True)
        # )
        # self.att = nn.Sequential(
        #     nn.Linear(in_features=256*2, out_features=128),
        #     nn.Tanh(),
        #     nn.Linear(in_features=128, out_features=1),
        #     nn.ReLU(True)
        # )

        # self.rnn = torch.nn.LSTM(2048, 256, batch_first=True, bidirectional=False, num_layers=2)
        self.rnn = torch.nn.GRU(2048, 256, batch_first=True, bidirectional=True, num_layers=2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.softmax = nn.Softmax(dim=1)
        # self.dropout = torch.nn.Dropout(p=0.2)

        # self._initialize_weights()

    def forward(self, x, length):
        assert( length.sum() == x.size(0))
        max_len = max(length)
        frame_feature = []
        start = 0
        for l in length:
            video = x[start:start+l]
            if video.size(0) < max_len:
                diff = max_len - video.size(0)
                pad = torch.zeros((diff, video.size(1), video.size(2), video.size(3))).type(torch.FloatTensor).cuda()
                video = torch.cat((video, pad), dim=0)
            video = video.view(max_len, -1).unsqueeze(0)

            frame_feature.append(video)
            start += l

        frame_feature = torch.cat(frame_feature, dim=0)
        
        # frame_feature, sorted_lengths, unsorted_idx = self.sort_sequences(frame_feature, length)
        # frame_feature = torch.nn.utils.rnn.pack_padded_sequence(
        #       frame_feature, sorted_lengths, batch_first=True)
        frame_feature, _ = self.rnn(frame_feature)
        # frame_feature, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #                         frame_feature, batch_first=True)

        # frame_feature = frame_feature.index_select(0, unsorted_idx)
        # frame_feature = self.dropout(frame_feature)
        

        # att = self.att(frame_feature)
        # att = self.softmax(att)
        # frame_feature = torch.bmm(frame_feature.transpose(1,2),att).squeeze(-1)
        
        frame_feature = frame_feature.max(1)[0]
        # frame_feature = frame_feature[:,-1]
        # frame_feature = frame_feature.mean(dim=1)

        x = self.linear(frame_feature)
        return x 
        # x = self.output_layer(x)
        # x = self.log_softmax(x)
        # return x
        
    def sort_sequences(self,inputs, lengths):
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        _, unsorted_idx = sorted_idx.sort()
        return inputs[sorted_idx], lengths_sorted.cuda(), unsorted_idx.cuda()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag=True
    for v in cfg:
        s=1
        if (v==64 and first_flag):
            s=2
            first_flag=False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def vgg16bn(pretrained=True, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
    yolo.load_state_dict(yolo_state_dict)
    return yolo

class Resnet(nn.Module):

    def __init__(self, features):
        super(Resnet, self).__init__()
        self.features = features

    def forward(self, x, length):
        x = self.features(x)
        
        return x


def Resnet50(pretrained=False, **kwargs):
    resnet50 = models.resnet50(pretrained=pretrained)
    newmodel = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

    model = Resnet(newmodel, **kwargs)
    return model

def test_res():
    import torch
    model = Resnet50(pretrained=True)
    # img = torch.rand(1,3,448,448)
    batch_size = 1

    img = torch.rand(10, 3, 240,320)
    length = [3,7]

    #img = img.view(batch_size*img.size(1), img.size(2),img.size(3),img.size(4))
    model.eval()
    output = model(img, length)
    print(output.size())

def test():
    import torch
    model = vgg16bn(pretrained=True, batch_size=2)
    # img = torch.rand(1,3,448,448)
    batch_size = 1

    img = torch.rand(10, 3, 240,320)
    length = torch.tensor([3,7])

    #img = img.view(batch_size*img.size(1), img.size(2),img.size(3),img.size(4))
    model.eval()
    output = model(img, length)
    print(output.size())

if __name__ == '__main__':
    test()
