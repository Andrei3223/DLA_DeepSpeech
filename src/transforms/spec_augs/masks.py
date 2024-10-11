from torch import Tensor, nn
import torchaudio.transforms


class FreqMasking(nn.Module):
    def __init__(self, freq_mask_param):
        super().__init__()
        self.transform = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def __call__(self, data: Tensor):
        return self.transform(data)
        


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param):
        super().__init__()
        self.transform = torchaudio.transforms.TimeMasking(time_mask_param)

    def __call__(self, data: Tensor):
        return self.transform(data)
