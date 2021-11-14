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

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim=1024,activation='relu'):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_linear = nn.Linear(in_dim , in_dim//8)
        self.key_linear = nn.Linear(in_dim , in_dim//8)
        self.value_linear = nn.Linear(in_dim , in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        
        # m_batchsize,C,width ,height = x.size()
        m_batchsize,S,H = x.size()
        

        proj_query  = self.query_linear(x).view(m_batchsize,S,-1) # B X CX(N)
        proj_key =  self.key_linear(x).view(m_batchsize,S,-1).permute(0,2,1) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_linear(x).view(m_batchsize,S,-1) # B X C X N
        out = torch.bmm(attention, proj_value )
        out = out.view(m_batchsize,S,H)
        
        out = self.gamma*out + x
        return out



class ATT_Classifier(nn.Module):

    def __init__(self):
        super(ATT_Classifier, self).__init__()
        
        self.linear = nn.Sequential(
            #TODO
            nn.Linear(in_features=256*2, out_features=256),
            #nn.Linear(in_features=256, out_features=256),
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
            nn.Linear(in_features=256*2, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=1),
            nn.ReLU(True)
        )

        # self.one_att_1 = nn.Sequential(
        #     nn.Linear(in_features=256+2048, out_features=1024),
        #     nn.Tanh(),
        #     nn.Linear(in_features=1024, out_features=1),
        #     nn.ReLU(True)
        # )

        # self.one_att_2 = nn.Sequential(
        #     nn.Linear(in_features=256+256, out_features=128),
        #     nn.Tanh(),
        #     nn.Linear(in_features=128, out_features=1),
        #     nn.ReLU(True)
        # )

        self.rnn_1 = torch.nn.GRU(512, 256, batch_first=True, bidirectional=True, num_layers=1)
        self.rnn_2 = torch.nn.GRU(512, 256, batch_first=True, bidirectional=True, num_layers=1)

        self.proj_dim = nn.Linear(2048, 512)

        self.att_1 = Self_Attn( 512, 'relu') 
        self.nor_1 = nn.LayerNorm(512)

        self.att_2 = Self_Attn( 512, 'relu') 
        self.nor_2 = nn.LayerNorm(512)

        self.att_3 = Self_Attn( 512, 'relu') 
        self.nor_3 = nn.LayerNorm(512)



        # self.Lstm_cell_1 = nn.LSTMCell(2048, 256)
        # self.Lstm_cell_2 = nn.LSTMCell(256, 256)


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

        ### Project dim ###
        frame_feature = self.proj_dim(frame_feature)
        frame_feature = self.dropout(frame_feature)

        ### Img feature ATT ###
        img_feature_att = self.att_1(frame_feature)
        frame_feature = frame_feature+img_feature_att
        frame_feature = self.nor_1(frame_feature)

        ### RNN feature ATT ###
        rnn_feature, _ = self.rnn_1(frame_feature)
        rnn_feature_att = self.att_2(rnn_feature)
        frame_feature = rnn_feature+rnn_feature_att
        frame_feature = self.nor_2(frame_feature)

        ### RNN feature ATT ###
        rnn_feature,_ = self.rnn_2(frame_feature)
        rnn_feature_att = self.att_3(rnn_feature)
        frame_feature = rnn_feature+rnn_feature_att
        frame_feature = self.nor_3(frame_feature)

        ### All frame weight sum ###
        att = self.att(frame_feature)
        att = self.softmax(att)
        frame_feature = torch.bmm(frame_feature.transpose(1,2),att).squeeze(-1)

        frame_feature = self.linear(frame_feature)
        frame_feature = self.output_layer(frame_feature)
        frame_feature = self.log_softmax(frame_feature)

        return frame_feature

        print(frame_feature.size())
        exit()

        h0 = torch.zeros(frame_feature.size(0), 256).cuda()
        c0 = torch.zeros(frame_feature.size(0), 256).cuda()
        
        #frame_feature_t = frame_feature.transpose(0,1)

        cell_output_1 = []
        for i in range(frame_feature.size(1)):
            context = self.one_step_att_1(frame_feature, h0)
            h0,c0 = self.Lstm_cell_1(context, (h0,c0))

            cell_output_1.append(h0.unsqueeze(0))
        
        cell_output_1 = torch.cat(cell_output_1, dim=0).transpose(0,1)

        
        h0 = torch.zeros(frame_feature.size(0), 256).cuda()
        c0 = torch.zeros(frame_feature.size(0), 256).cuda()
        cell_output_2 = []
        for i in range(cell_output_1.size(1)):
            context = self.one_step_att_2(cell_output_1, h0)
            h0,c0 = self.Lstm_cell_2(context, (h0,c0))

            cell_output_2.append(h0.unsqueeze(0))
        
        frame_feature = torch.cat(cell_output_2, dim=0).transpose(0,1)

        
        # frame_feature, sorted_lengths, unsorted_idx = self.sort_sequences(frame_feature, length)
        # frame_feature = torch.nn.utils.rnn.pack_padded_sequence(
        #       frame_feature, sorted_lengths, batch_first=True)
        #frame_feature, _ = self.rnn(frame_feature)
        # frame_feature, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #                         frame_feature, batch_first=True)

        # frame_feature = frame_feature.index_select(0, unsorted_idx)
        # frame_feature = self.dropout(frame_feature)
        

        att = self.att(frame_feature)
        att = self.softmax(att)
        frame_feature_att = torch.bmm(frame_feature.transpose(1,2),att).squeeze(-1)
        #frame_feature_att = self.dropout(frame_feature_att)

        # = frame_feature.max(1)[0]
        #frame_feature_max = self.dropout(frame_feature_max)
        

        #frame_feature = torch.cat((frame_feature_att, frame_feature_max), dim=-1)

        

        # frame_feature = frame_feature[:,-1]
        # frame_feature = frame_feature.mean(dim=1)

        x = self.linear(frame_feature_att)
        x = self.output_layer(x)
        x = self.log_softmax(x)
        return x
    
    def one_step_att_1(self, a, s_prev):

        # repeat = s_prev.view(s_prev.size(0), 1, -1).expand_as(a)
        repeat = s_prev.view(s_prev.size(0), 1, -1).repeat(1, a.size(1), 1)
        
        concat = torch.cat((a, repeat), dim=-1)
        
        energies = self.one_att_1(concat)
        att = self.softmax(energies)

        context = torch.bmm(att.transpose(1,2),a).squeeze(1)

        return context
    
    def one_step_att_2(self, a, s_prev):

        # repeat = s_prev.view(s_prev.size(0), 1, -1).expand_as(a)
        repeat = s_prev.view(s_prev.size(0), 1, -1).repeat(1, a.size(1), 1)
        
        concat = torch.cat((a, repeat), dim=-1)
        
        energies = self.one_att_2(concat)
        att = self.softmax(energies)

        context = torch.bmm(att.transpose(1,2),a).squeeze(1)

        return context



        
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
