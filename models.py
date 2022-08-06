import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import num_classes


class GoModelBase(pl.LightningModule):
    def __init__(self):
        super(GoModelBase, self).__init__()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        # x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc)


class SmallModel(GoModelBase):
    def __init__(self, input_size):
        super(SmallModel, self).__init__()

        self.features = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(in_channels=input_size[0], out_channels=48, kernel_size=(7, 7)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=11552, out_features=512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MediumModel(GoModelBase):
    def __init__(self, input_size):
        super(MediumModel, self).__init__()

        self.features = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(in_channels=input_size[0], out_channels=64, kernel_size=(7, 7), padding='valid'),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=23104, out_features=512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LargeModel(GoModelBase):
    def __init__(self, input_size):
        super(LargeModel, self).__init__()

        self.features = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(in_channels=input_size[0], out_channels=64, kernel_size=(7, 7), padding='valid'),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=11552, out_features=1024),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def alphago_model(input_size, is_policy_net=False, num_filter=192, first_kernel_size=5,
                  other_kernel_size=3):
    model = nn.Sequential()
    model.append(nn.Conv2d(in_channels=input_size[0], out_channels=num_filter,
                           kernel_size=(first_kernel_size, first_kernel_size), padding='same'))
    model.append(nn.ReLU())

    for i in range(2, 12):
        model.append(nn.Conv2d(in_channels=num_filter, out_channels=num_filter,
                               kernel_size=(other_kernel_size, other_kernel_size), padding='same'))
        model.append(nn.ReLU())

    if is_policy_net:
        model.append(
            nn.Conv2d(in_channels=num_filter, out_channels=1, kernel_size=(1, 1), padding='same'))
        model.append(nn.ReLU())
        model.append(nn.Flatten())

        model.append(nn.Linear(in_features=19 * 19, out_features=1000))
        model.append(nn.ReLU())

        model.append(nn.Linear(in_features=1000, out_features=19 * 19))

        return model

    else:
        model.append(
            nn.Conv2d(in_channels=num_filter, out_channels=num_filter,
                      kernel_size=(other_kernel_size, other_kernel_size), padding='same'))
        model.append(nn.ReLU())

        model.append(
            nn.Conv2d(in_channels=num_filter, out_channels=1, kernel_size=(1, 1), padding='same'))
        model.append(nn.ReLU())

        model.append(nn.Flatten())

        model.append(nn.Linear(in_features=19 * 19, out_features=256))
        model.append(nn.ReLU())

        model.append(nn.Linear(in_features=256, out_features=1))
        model.append(nn.Tanh())

        return model


class AlphaGoModel(GoModelBase):
    def __init__(self, input_size=(7, 19, 19), is_policy_net=False):
        super(AlphaGoModel, self).__init__()

        self.features = alphago_model(input_size=input_size, is_policy_net=is_policy_net)

    def forward(self, x):
        x = self.features(x)

        return x


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class AlphaZeroModel(GoModelBase):
    def __init__(self, input_size=(1, 19, 19), block=ResidualBlock):
        super(AlphaZeroModel, self).__init__()

        self.in_channels = 256

        self.conv_bn_relu_block = nn.Sequential(
            conv3x3(input_size[0], 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual_tower = self.make_layer(block, 256, 20)
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 19 * 19, 19 * 19)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(19 * 19, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        first_conv = self.conv_bn_relu_block(x)
        res_tower = self.residual_tower(first_conv)
        # return self.policy_head(res_tower), self.value_head(res_tower)
        return self.policy_head(res_tower)


if __name__ == '__main__':
    from torchsummary import summary

    input_size = (7, 19, 19)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SmallModel(input_size).to(device)
    # model = MediumModel(input_size).to(device)
    # model = LargeModel(input_size).to(device)
    # model = AlphaGoModel(input_size, is_policy_net=True).to(device)
    model = AlphaZeroModel(input_size).to(device)
    summary(model, input_size)

    input = torch.rand(input_size)
    input = torch.unsqueeze(input, dim=0)
    out = model(input)
    print(out.size())
    print(out)

    # model = AlphaGoModel(input_shape=(7, 19, 19), is_policy_net=False).to(device)
    # summary(model, (7, 19, 19))
