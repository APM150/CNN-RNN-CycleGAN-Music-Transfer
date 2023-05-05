import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, image):
        """
            Predict whether the given image is fake (0) or not (1).
            image: (batch, 64, 84, 1)
            return: (batch, 16, 21, 1)
        """
        image_ = image.view(image.shape[0], image.shape[3], image.shape[1], image.shape[2])
        h = self.conv(image_)
        pred = h.view(h.shape[0], h.shape[2], h.shape[3], h.shape[1])
        return pred


class Residual(nn.Module):

    def __init__(self):
        super(Residual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        """
            x: (batch, 256, 16, 21)
            return: (batch, 256, 16, 21)
        """
        y = F.pad(x, (1, 1, 1, 1), "constant", 0)
        y = self.conv(y)
        y = F.pad(y, (1, 1, 1, 1), "constant", 0)
        y = self.conv(y)
        return F.relu(y + x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        """
            Output a fake image with the same size.
            image: (batch, 64, 84, 1)
            return: (batch, 64, 84, 1)
        """
        image_ = image.view(image.shape[0], image.shape[3], image.shape[1], image.shape[2])
        h0 = F.pad(image_, (3, 3, 3, 3), "constant", 0)
        h1 = self.conv1(h0)
        h2 = F.pad(h1, (3, 3, 3, 3), "constant", 0)
        h3 = self.conv2(h2)
        pred = h3.view(h3.shape[0], h3.shape[2], h3.shape[3], h3.shape[1])
        return pred


class ResidualV2(nn.Module):

    def __init__(self):
        super(ResidualV2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.InstanceNorm2d(256),
        )

    def forward(self, x):
        """
            x: (batch, 256, 16, 21)
            return: (batch, 256, 16, 21)
        """
        y = F.pad(x, (1, 1, 1, 1), "constant", 0)
        y = self.conv(y)
        y = F.pad(y, (1, 1, 1, 1), "constant", 0)
        y = self.conv(y)
        return F.relu(y + x)


class GeneratorV2(nn.Module):

    def __init__(self):
        super(GeneratorV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            ResidualV2(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        """
            Output a fake image with the same size.
            image: (batch, 64, 84, 1)
            return: (batch, 64, 84, 1)
        """
        image_ = image.view(image.shape[0], image.shape[3], image.shape[1], image.shape[2])
        h0 = F.pad(image_, (3, 3, 3, 3), "constant", 0)
        h1 = self.conv1(h0)
        h2 = F.pad(h1, (3, 3, 3, 3), "constant", 0)
        h3 = self.conv2(h2)
        pred = h3.view(h3.shape[0], h3.shape[2], h3.shape[3], h3.shape[1])
        return pred


class Classifer(nn.Module):

    def __init__(self):
        super(Classifer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 12), stride=(1, 12)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 1), stride=(4, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(8, 1), stride=(8, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 7), stride=(1, 7)),
            # nn.Softmax(dim=1)
        )

    def forward(self, image):
        """
            Output the probability distribution of given image genre (A or B).
            image: (batch, 64, 84, 1)
            return: (batch, 2)
        """
        image_ = image.view(image.shape[0], image.shape[3], image.shape[1], image.shape[2])
        pred = self.conv(image_)
        return pred.squeeze(-1).squeeze(-1)


class RNNGenerator(nn.Module):

    def __init__(self, args):
        super(RNNGenerator, self).__init__()
        self.args = args
        self.rnn1 = nn.RNN(input_size=84, hidden_size=128, num_layers=2, batch_first=True)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=84, num_layers=1, batch_first=True)

    def forward(self, x):
        """
            Given the sequential music pitch data, output the fake data with same size
            x: (batch, 64, 84, 1)
            return: (batch, 64, 84, 1)
        """
        x_ = x.squeeze(-1)
        h0_1 = torch.zeros((2, x.shape[0], 128)).to(self.args.device)
        out1, _ = self.rnn1(x_, h0_1)     # (batch, 64, 128)
        h0_2 = torch.zeros((1, out1.shape[0], 84)).to(self.args.device)
        out2, _ = self.rnn2(out1, h0_2)   # (batch, 64, 84)

        return out2.unsqueeze(-1)


class LSTMGenerator(nn.Module):

    def __init__(self, args):
        super(LSTMGenerator, self).__init__()
        self.args = args
        self.lstm1 = nn.LSTM(input_size=84, hidden_size=128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=84, num_layers=1, batch_first=True)

    def forward(self, x):
        """
            Given the sequential music pitch data, output the fake data with same size
            x: (batch, 64, 84, 1)
            return: (batch, 64, 84, 1)
        """
        x_ = x.squeeze(-1)
        h0_1 = torch.zeros((2, x.shape[0], 128)).to(self.args.device)
        c0_1 = torch.zeros((2, x.shape[0], 128)).to(self.args.device)
        out1, _ = self.lstm1(x_, (h0_1, c0_1))     # (batch, 64, 128)
        h0_2 = torch.zeros((1, out1.shape[0], 84)).to(self.args.device)
        c0_2 = torch.zeros((1, out1.shape[0], 84)).to(self.args.device)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))   # (batch, 64, 84)

        return out2.unsqueeze(-1)


class LSTMGeneratorV2(nn.Module):

    def __init__(self, args):
        super(LSTMGeneratorV2, self).__init__()
        self.args = args
        self.lstm1 = nn.LSTM(input_size=84, hidden_size=256, num_layers=4, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=4, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=64, hidden_size=84, num_layers=1, batch_first=True)

    def forward(self, x):
        """
            Given the sequential music pitch data, output the fake data with same size
            x: (batch, 64, 84, 1)
            return: (batch, 64, 84, 1)
        """
        x_ = x.squeeze(-1)

        h0_1 = torch.zeros((4, x.shape[0], 256)).to(self.args.device)
        c0_1 = torch.zeros((4, x.shape[0], 256)).to(self.args.device)
        out1, _ = self.lstm1(x_, (h0_1, c0_1))     # (batch, 64, 256)

        h0_2 = torch.zeros((4, out1.shape[0], 128)).to(self.args.device)
        c0_2 = torch.zeros((4, out1.shape[0], 128)).to(self.args.device)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))   # (batch, 64, 128)

        h0_3 = torch.zeros((2, out2.shape[0], 64)).to(self.args.device)
        c0_3 = torch.zeros((2, out2.shape[0], 64)).to(self.args.device)
        out3, _ = self.lstm3(out2, (h0_3, c0_3))   # (batch, 64, 64)

        h0_4 = torch.zeros((1, out3.shape[0], 84)).to(self.args.device)
        c0_4 = torch.zeros((1, out3.shape[0], 84)).to(self.args.device)
        out4, _ = self.lstm4(out3, (h0_4, c0_4))   # (batch, 64, 84)

        return torch.tanh(out4.unsqueeze(-1))


class LSTMDiscriminator(nn.Module):

    def __init__(self, args):
        super(LSTMDiscriminator, self).__init__()
        self.args = args
        self.lstm1 = nn.LSTM(input_size=84, hidden_size=256, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        """
            Predict whether the given sequence is fake (0) or not (1).
            x: (batch, 64, 84, 1)
            return: (batch, 1)
        """
        x_ = x.squeeze(-1)

        h0_1 = torch.zeros((2, x.shape[0], 256)).to(self.args.device)
        c0_1 = torch.zeros((2, x.shape[0], 256)).to(self.args.device)
        out1, _ = self.lstm1(x_, (h0_1, c0_1))     # (batch, 64, 256)

        h0_2 = torch.zeros((2, out1.shape[0], 128)).to(self.args.device)
        c0_2 = torch.zeros((2, out1.shape[0], 128)).to(self.args.device)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))   # (batch, 64, 128)

        h0_3 = torch.zeros((1, out2.shape[0], 64)).to(self.args.device)
        c0_3 = torch.zeros((1, out2.shape[0], 64)).to(self.args.device)
        out3, _ = self.lstm3(out2, (h0_3, c0_3))   # (batch, 64, 64)

        # Take the output of the last time step
        out4 = self.fc(out3[:, -1, :])             # (batch, 1)

        return torch.sigmoid(out4)



if __name__ == '__main__':
    rnn = RNNGenerator(args=None)
    image = torch.ones((5, 64, 84, 1))
    pred = rnn(image)
    print(pred.shape)
