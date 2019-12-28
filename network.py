from torch import nn
# from lib.non_local_concatenation import NONLocalBlock2D
# from lib.non_local_gaussian import NONLocalBlock2D
from lib.non_local_embedded_gaussian import NONLocalBlock3D
# from lib.non_local_dot_product import NONLocalBlock2D


class Network(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Network, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
        )
            #nn.MaxPool2d(2),

        self.nl_1 = NONLocalBlock3D(in_channels=output_channels)
        self.conv_2  = nn.Sequential(
            nn.Conv3d(in_channels=output_channels, out_channels=output_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_channels*2),
            nn.ReLU(),
        )
            #nn.MaxPool2d(2),

            #NONLocalBlock3D(in_channels=output_channels*2),
            #nn.Conv3d(in_channels=output_channels*2, out_channels=output_channels*4, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(output_channels*4),
            #nn.ReLU(),
            #nn.MaxPool2d(2),
        

        self.fc = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        #print(x.shape)
        x = x.view(-1, 1024, 6, 7,7)
        batch_size = x.size(0)
        feature_1 = self.conv_1(x)
        #nl_feature_1, nl_map_1 = self.nl_1(feature_1, return_nl_map=True)
        nl_feature_1 = self.nl_1(feature_1)
        output = self.conv_2(nl_feature_1)
        #output = self.convs(x)
        #output = self.convs(x).view(batch_size, -1)
        #output = self.fc(output)
        return output
    
    def forward_with_nl_map(self,x):
        x = x.view(-1, 1024, 6,7,7)
        batch_size = x.size(0)
        feature_1 = self.conv_1(x)
        nl_feature_1, nl_map_1 = self.nl_1(feature_1, reture_nl_map=True)
        output = self.conv_2(nl_feature_1)
        return output, nl_map_1



if __name__ == '__main__':
    import torch

    img = torch.randn(8, 1024,6, 7, 7)
    net = Network(1024, 32)
    out = net(img)
    print(out.size())

