import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class Linear(nn.Module):
  def __init__(
      self,
      input_dims,
      backbone_model,
      backbone_out_dims,
      hidden_dims,
      return_nodes):

    super(Linear, self).__init__()
    self.feature_extractor = create_feature_extractor(backbone_model, return_nodes, suppress_diff_warning=True)
    self.bn_1 = nn.BatchNorm2d(backbone_out_dims[0])
    self.dropout_1 = nn.Dropout(0.2)
    self.avg_pool_1 = nn.AvgPool2d(kernel_size=(backbone_out_dims[1], backbone_out_dims[2]))
    self.linear_1 = nn.Linear(backbone_out_dims[0], hidden_dims)
    self.relu_1 = nn.ReLU()
    self.linear_2 = nn.Linear(hidden_dims, 10)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    input = x
    feature = self.feature_extractor(input)["fe_output"]
    feature = self.bn_1(feature)
    feature = self.dropout_1(feature)
    flattened = self.avg_pool_1(feature).squeeze(dim=-1).squeeze(dim=-1)
    
    output = self.linear_1(flattened)
    output = self.relu_1(output)
    output = self.linear_2(output)
    output = self.softmax(output)

    return output