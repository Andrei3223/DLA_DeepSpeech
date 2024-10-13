import os
from pathlib import Path

import gdown

URL = "https://drive.google.com/uc?id=1-5WqmMBK08NUUNO3oV0R17yIYl-Bu_Q3"
LM_URL = "https://drive.google.com/uc?id=1G4HUjT_l9EvRkGuVMvkK7VOQFUxhy6Ca"

root_dir = Path(__file__).absolute().resolve().parent
model_dir = root_dir / "saved"
model_dir.mkdir(exist_ok=True, parents=True)

output_model = "models/final_model_weights.pth"
lm_path = "src/utils/lowercase_3-gram.pruned.1e-7.arpa"

gdown.download(URL, output_model, quiet=False)
gdown.download(LM_URL, lm_path, quiet=False)
