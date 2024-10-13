import math

import torch
from torch import nn
from torch.nn import Conv2d, Sequential

# from baseline_model import BaselineModel

RELU_UPPER_BOUND = 20


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_param: int = 0):
        super(GRU, self).__init__()

        self.net = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_param,
        )

        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        x = self.batch_norm(x.transpose(1, 2)).transpose(2, 1)
        x, _ = self.net(x)

        return x


def get_conv_ouput_shape_constant(initial, kernel, stride, pad=0, delay=1):
    return math.floor((initial + 2 * pad - delay * (kernel - 1) - 1) / stride) + 1


def get_conv_ouput_shape(initial, kernel, stride, pad=0, delay=1):
    x = torch.tensor(initial + 2 * pad - delay * (kernel - 1) - 1, dtype=torch.float64)
    return torch.floor(x / stride + 1).int()


class DeepSpeech(nn.Module):
    """
    DeepSpeech 2
    """

    def __init__(
        self,
        n_feats,
        n_tokens=28,
        rnn_hidden_size: int = 512,
        rnn_blocks_num: int = 3,
        rnn_dropout: int = 0,
    ):
        super().__init__()

        self.conv_2_layers = Sequential(
            Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, RELU_UPPER_BOUND, inplace=True),
            Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, RELU_UPPER_BOUND, inplace=True),
        )

        self.conv_3_layers = Sequential(
            Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, RELU_UPPER_BOUND, inplace=True),
            Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, RELU_UPPER_BOUND, inplace=True),
            Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, RELU_UPPER_BOUND, inplace=True),
        )

        self.rnn = nn.Sequential(
            *[
                GRU(
                    input_size=384 if i == 0 else rnn_hidden_size * 2,
                    hidden_size=rnn_hidden_size,
                    dropout_param=rnn_dropout,
                )
                for i in range(rnn_blocks_num)
            ]
        )

        self.FC = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, n_tokens),
        )

    def get_output_shape(self, input_shape: torch.Tensor, three_layers: bool = False):
        first_conv_shape = get_conv_ouput_shape(input_shape, 11, 2)
        second_conv_shape = get_conv_ouput_shape(first_conv_shape, 11, 1)
        if not three_layers:
            return second_conv_shape
        third_conv_shape = get_conv_ouput_shape(second_conv_shape, 11, 1)

        return third_conv_shape

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = spectrogram.unsqueeze(1)
        batch_size = spectrogram.shape[0]
        x = self.conv_2_layers(x)

        x = self.rnn(x.reshape(batch_size, -1, x.shape[-1]).permute((0, 2, 1)))

        output = self.FC(x)

        log_probs = nn.functional.log_softmax(output, dim=-1)
        out_length = self.get_output_shape(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": out_length}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
