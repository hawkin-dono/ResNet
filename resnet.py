import torch
import torch.nn as nn 
from torchvision.models import resnet18, resnet101

## Basic Block for ResNet18, 34
class SimpleBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleBasicBlock, self).__init__()
        first_stride = 1 if in_channels == out_channels else 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=first_stride, padding=1)   
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x   # batch_size, in_channels, H, W
        out = self.conv1(x)  # batch_size, out_channels, H/2, W/2
        out = self.bn1(out) # batch_size, out_channels, H/2, W/2
        out = self.relu(out) # batch_size, out_channels, H/2, W/2
        out = self.conv2(out) # batch_size, out_channels, H/2, W/2 
        out = self.bn2(out) # batch_size, out_channels, H/2, W/2
        
        if hasattr(self, 'downsample'):
            out += self.downsample(identity)
        else:
            out += identity
        out = self.relu(out) # batch_size, out_channels, H/2, W/2
        return out
    
### Basic Block for ResNet50, 101, 152
class ComplexBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexBasicBlock, self).__init__()
        hidden_channels = out_channels // 4
        mid_layer_stride = 1 if in_channels == out_channels else 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=mid_layer_stride, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x  # batch_size, in_channels, H, W
        out = self.conv1(x) # batch_size, hidden_channels, H, W
        out = self.bn1(out) # batch_size, hidden_channels, H, W
        out = self.conv2(out) # batch_size, hidden_channels, H/2, W/2 (if downsample) or H, W
        out = self.bn2(out) # batch_size, hidden_channels, H/2, W/2 (if downsample) or H, W
        out = self.conv3(out)   # batch_size, out_channels, H/2, W/2 (if downsample) or H, W
        out = self.bn3(out) # batch_size, out_channels, H/2, W/2 (if downsample) or H, W
        
        if hasattr(self, 'downsample'):
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out) # batch_size, out_channels, H/2, W/2 (if downsample) or H, W
        return out     
    
    
### Refactored Resnet
class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks, num_classes= 10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block_type, in_channels =64, out_channels= num_blocks[0][1], freq = num_blocks[0][0])
        self.layer2 = self._make_layer(block_type, in_channels = num_blocks[0][1], out_channels= num_blocks[1][1], freq = num_blocks[1][0])
        self.layer3 = self._make_layer(block_type, in_channels = num_blocks[1][1], out_channels= num_blocks[2][1], freq = num_blocks[2][0])
        self.layer4 = self._make_layer(block_type, in_channels = num_blocks[2][1], out_channels= num_blocks[3][1], freq = num_blocks[3][0])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_blocks[3][1], num_classes)
        
    def _make_layer(self, block_type, in_channels, out_channels, freq):
        
        layers = []
        if in_channels != out_channels:
            layers.append(block_type(in_channels, out_channels))
            freq -= 1
        for _ in range(freq):
            layers.append(block_type(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x) # batch_size, 64, H/2, W/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x) # batch_size, 64, H/4, W/4
        x = self.layer2(x) # batch_size, 128, H/8, W/8
        x = self.layer3(x) # batch_size, 256, H/16, W/16
        x = self.layer4(x) # batch_size, 512, H/32, W/32
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
        
def get_model(name: str):
    config = {"resnet18": (SimpleBasicBlock, [(2, 64), (2, 128), (2, 256), (2, 512)]),
              "resnet34": (SimpleBasicBlock, [(3, 64), (4, 128), (6, 256), (3, 512)]),
              "resnet50": (ComplexBasicBlock, [(3, 256), (4, 512), (6, 1024), (3, 2048)]),
              "resnet101": (ComplexBasicBlock, [(3, 256), (4, 512), (23, 1024), (3, 2048)]),
              "resnet152": (ComplexBasicBlock, [(3, 256), (8, 512), (36, 1024), (3, 2048)])}
    block, num_blocks = config[name]
    model = ResNet(block, num_blocks)
    return model


##### Test fuction
def test_simple_basic_block():
    model = SimpleBasicBlock(64, 128)
    x = torch.randn(1, 64, 56, 56)
    y = model(x)
    print(y.size())
    
def test_complex_basic_block():
    model = ComplexBasicBlock(64, 256)
    x = torch.randn(1, 64, 56, 56)
    y = model(x)
    print(y.size()) 
    
def test_resnet(name: str):
    model = get_model(name)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())
    
def main():
    test_simple_basic_block()
    test_complex_basic_block()
    test_resnet("resnet18")
    test_resnet("resnet34")
    test_resnet("resnet50")
    test_resnet("resnet101")
    test_resnet("resnet152")
    
    resnet18 = get_model("resnet18")
    print(resnet18)
    
    
if __name__ == "__main__":
    main()
        