import torch
import torch.nn as nn


class FoodClassifierBaseline(nn.Module):
 def __init__(self,num_classes = 101):
   super().__init__()
   # Layer 1: Flatten Layer, to flatten the images from [batch_size, 3, 224, 224] to [batch_size, 3*224*224]
   self.flatten = nn.Flatten()
   # Layer 2: Linear
   self.linear1 = nn.Linear(in_features=3*224*224, out_features=256)
   # Layer 3: ReLU
   self.relu1 = nn.ReLU()
   # Layer 4: Linear
   self.linear2 = nn.Linear(in_features=256, out_features=num_classes)


 def forward(self,x):
   x=self.flatten(x)
   x=self.linear1(x)
   x=self.relu1(x)
   x=self.linear2(x)
   return x


if __name__ == '__main__':
 print("Initializing Model Architecture and running Sanity check")
 model = FoodClassifierBaseline(num_classes=101)
 dummy_input = torch.randn(28,3,224,224)
 dummy_output = model(dummy_input)
 print(f'Input batch shape: [batch_size, channels, height, width]->{dummy_input.shape}')
 print(f'Output batch shape: [batch_size, num_classes]->{dummy_output.shape}')