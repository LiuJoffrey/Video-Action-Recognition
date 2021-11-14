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
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
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

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.hidden_size =  512
        self.n_layers = 1
        dropout = 0.5
        self.dropout = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(2048, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else dropout), bidirectional=True,
                          batch_first=True)
        self.feed1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)

        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else dropout), bidirectional=True,
                          batch_first=True)
        self.feed2 = nn.Linear(self.hidden_size*2, self.hidden_size*2)


        
        self.output_layer = nn.Linear(in_features=self.hidden_size*2, out_features=11)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=1)

        self.attn1 = Self_Attn( 1024, 'relu')        
        # self.attn1_1 = Self_Attn( 1024, 'relu')
        # self.attn1_2 = Self_Attn( 1024, 'relu')
        # self.attn1_3 = Self_Attn( 1024, 'relu')
        # self.attn1_4 = Self_Attn( 1024, 'relu')
        # self.Linear_multi_head1 = nn.Linear(4096, 1024)
        
        self.attn2 = Self_Attn( 1024,  'relu')
        # self.attn2_1 = Self_Attn( 1024,  'relu')
        # self.attn2_2 = Self_Attn( 1024,  'relu')
        # self.attn2_3 = Self_Attn( 1024,  'relu')
        # self.attn2_4 = Self_Attn( 1024,  'relu')
        # self.Linear_multi_head2 = nn.Linear(4096, 1024)

    def forward(self, x, length):
        ### Layer 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, 
                                                         length, 
                                                         batch_first=True)
        outputs, _ = self.lstm1(packed)
        outputs_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = self.attn1(outputs_rnn)        
        # outputs1 = self.attn1_1(outputs_rnn)
        # outputs2 = self.attn1_2(outputs_rnn)
        # outputs3 = self.attn1_3(outputs_rnn)
        # outputs4 = self.attn1_4(outputs_rnn)
        # head1 = torch.cat((outputs1, outputs2, outputs3, outputs4),-1)
        # outputs = self.Linear_multi_head1(head1)
        outputs = self.feed1(outputs)
        outputs = outputs_rnn+outputs
        outputs = self.dropout(outputs)

        ### Layer 2
        packed = torch.nn.utils.rnn.pack_padded_sequence(outputs, 
                                                         length, 
                                                         batch_first=True)
        outputs, _ = self.lstm2(packed)
        outputs_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = self.attn2(outputs_rnn)
        # outputs1 = self.attn2_1(outputs_rnn)
        # outputs2 = self.attn2_2(outputs_rnn)
        # outputs3 = self.attn2_3(outputs_rnn)
        # outputs4 = self.attn2_4(outputs_rnn)
        # head2 = torch.cat((outputs1, outputs2, outputs3, outputs4),-1)
        # outputs = self.Linear_multi_head2(head2)
        outputs = self.feed2(outputs)
        outputs = outputs_rnn+outputs
        outputs = self.dropout(outputs)

        outputs = self.output_layer(outputs)
        return outputs
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, model_output, groundtruth, lengths):
        w = torch.tensor([1., 5., 2., 2. ,2. ,2., 2., 2., 5., 2. ,2.]).cuda()
        criterion = nn.CrossEntropyLoss(weight=w)
        loss = 0
        batch_size = model_output.size(0)

        for i in range(batch_size):
            sample_length = lengths[i]
            target = groundtruth[i].type(torch.LongTensor).cuda()
            prediction = model_output[i][:sample_length]
            partial_loss = criterion(prediction, target)
            loss += partial_loss
        loss = loss / batch_size

        return loss




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

    def forward(self, x):
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
