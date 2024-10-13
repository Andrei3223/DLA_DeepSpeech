from torch import Tensor, nn, rand
import torchaudio.transforms


class FreqMasking(nn.Module):
    def __init__(self, freq_mask_param, p=1):
        super().__init__()
        self.transform = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.apply_prob = p

    def __call__(self, data: Tensor):
        return data if rand(1) < self.apply_prob else self.transform(data)
        


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param, p=1):
        super().__init__()
        self.transform = torchaudio.transforms.TimeMasking(time_mask_param)
        self.apply_prob = p

    def __call__(self, data: Tensor):
        return data if rand(1) < self.apply_prob else self.transform(data)
