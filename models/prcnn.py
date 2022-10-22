import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class PRCNN(nn.Module):
  def __init__(self, backbone_model, return_nodes):
    super(PRCNN, self).__init__()
    self.feature_extractor = create_feature_extractor(backbone_model, return_nodes, suppress_diff_warning=True)
    self.bn_1 = nn.BatchNorm2d(384)
    self.dropout_1 = nn.Dropout(0.2)
    self.avg_pool_1 = nn.AvgPool2d(kernel_size=(6, 33))
    self.maxpool_1 = nn.MaxPool2d((2, 1), stride=(2, 1))
    self.bi_rnn = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
    self.dropout_2 = nn.Dropout(0.2)
    self.linear_1 = nn.Linear(640, 128)
    self.relu_1 = nn.ReLU()
    self.linear_2 = nn.Linear(128, 10)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    input = x
    feature = self.feature_extractor(input)["fe_output"]
    feature = self.bn_1(feature)
    feature = self.dropout_1(feature)
    flattened = self.avg_pool_1(feature).squeeze(dim=-1).squeeze(dim=-1)

    # rnn branch
    rnn_feat = self.maxpool_1(input)
    rnn_feat = rnn_feat.squeeze(dim=1)
    rnn_feat = torch.permute(rnn_feat, (0, 2, 1))
    _, h_n = self.bi_rnn(rnn_feat)
    rnn_feat = h_n[0]
    rnn_feat = torch.concat([rnn_feat[0], rnn_feat[1]], dim=-1)
    rnn_feat = self.dropout_2(rnn_feat)

    concat_feat = torch.concat([flattened, rnn_feat], dim=-1)
    output = self.linear_1(concat_feat)
    output = self.relu_1(output)
    output = self.linear_2(output)
    output = self.softmax(output)

    return output