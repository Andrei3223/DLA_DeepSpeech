import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def __init__(self):
        super().__init__(zero_infinity=True)


    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        # print("in loss", log_probs_t.shape, log_probs_length, text_encoded.shape, text_encoded_length)
        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
            # zero_infinity=True,
        )

        return {"loss": loss}
