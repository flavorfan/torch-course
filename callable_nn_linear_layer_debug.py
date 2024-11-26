import torch
import torch.nn as nn

in_features = torch.tensor([1,2,3,4], dtype=torch.float32)

fc = nn.Linear(in_features=4, out_features=3)

out = fc(in_features)

print(out)