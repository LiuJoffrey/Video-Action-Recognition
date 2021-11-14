class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        
        # last = []
        # last.append(nn.Conv2d(512, 256, 3))
        # last.append(nn.BatchNorm2d(256))
        # last.append(nn.ReLU())
        # self.last = nn.Sequential(*last)

        self.linear = nn.Sequential(
            nn.Linear(in_features=256*2, out_features=256),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=11),
        )

        self.rnn = torch.nn.GRU(2048, 256, batch_first=True, bidirectional=True, num_layers=2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.softmax = nn.Softmax(dim=1)
        # self.dropout = torch.nn.Dropout(p=0.2)

        self._initialize_weights()

    def forward(self, x, length):
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
        
        frame_feature, sorted_lengths, unsorted_idx = self.sort_sequences(frame_feature, length)
        frame_feature = torch.nn.utils.rnn.pack_padded_sequence(
              frame_feature, sorted_lengths, batch_first=True)
        frame_feature, _ = self.rnn(frame_feature)
        frame_feature, _ = torch.nn.utils.rnn.pad_packed_sequence(
                                frame_feature, batch_first=True)

        frame_feature = frame_feature.index_select(0, unsorted_idx)
        # frame_feature = self.dropout(frame_feature)
        # frame_feature = frame_feature.mean(dim=1)
        frame_feature = frame_feature.max(1)[0]
        x = self.linear(frame_feature)
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