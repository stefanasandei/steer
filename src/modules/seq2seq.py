import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import regnet_y_32gf


class Seq2Seq(nn.Module):

    def __init__(self):
        super().__init__()

        self.n_input = 125  # after xyz concat will be 128
        self.n_output = 256

        self.encoder = Encoder(self.n_input)
        self.decoder = Decoder(self.n_input + 3, self.n_output)

    def forward(self, past_frames, past_xyz):
        """
        past_frames: (B, T, C, W, H)
        past_xyz: (B, T, 3)
        """

        # pass sequence through the encoder
        T = past_frames.shape[1]
        image_features = []

        for t in range(T):
            # frames at time t from all batches
            encoded = self.encoder(past_frames[:, t])

            # (B, n_input)
            image_features.append(encoded)

        # (T, B, n_input)
        feature_seq = torch.stack(image_features)

        # merge frame feature with past path
        past_xyz = past_xyz.permute(1, 0, 2)  # (T, B, 3)
        feature_seq = torch.cat((feature_seq, past_xyz), axis=2)  # (T, B, n_input+3)

        # (B, n_output)
        model_output = self.decoder(feature_seq)

        return model_output


class Encoder(nn.Module):
    """
    Used to encode a batch of frames. Will process its features to
    compute an encoded hidden state, passed to a decoder.
    """

    def __init__(self, n_feat: int):
        super().__init__()

        self.n_feat = n_feat

        # only want to get features, without classes logits
        self.feature_extractor = regnet_y_32gf()
        self.feature_extractor.fc = nn.Identity()

        self.fc = nn.LazyLinear(self.n_feat)
        self.bn = nn.BatchNorm1d(self.n_feat)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # X.shape = (B, C, W, H)
        x = self.feature_extractor(x)  # (B, features)
        x = self.fc(x)  # (B, n_feat)
        x = F.relu(self.bn(x))
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    """
    Process a sequence of encoded features to output final hidden
    state, to be passed to the last fully connected output layers.
    """

    def __init__(self, n_input: int, n_output: int):
        super().__init__()

        self.n_hidden = 128
        self.n_output = n_output

        self.gru = nn.GRU(n_input, self.n_hidden)
        self.fc = nn.Linear(self.n_hidden, self.n_output)
        self.bn = nn.BatchNorm1d(self.n_output)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # X.shape = (B, T, n_input)

        # last hidden state, (1, B, n_hidden)
        _, x = self.gru(x)
        x = torch.squeeze(x, 0)  # (B, n_hidden)

        x = self.dropout(x)
        x = self.fc(x)
        x = F.relu(self.bn(x))
        x = self.dropout(x)

        return x


# let's test the model
if __name__ == "__main__":
    B, T = 2, 3
    past_frames = torch.randn((B, T, 3, 1164 // 2, 874 // 2))
    past_xyz = torch.randn((B, T, 3))

    model = Seq2Seq()
    print(model(past_frames, past_xyz).shape)
