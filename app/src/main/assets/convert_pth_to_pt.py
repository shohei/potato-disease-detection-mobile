import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model = model.to(device)
model.load_state_dict(torch.load("model_ft_gpu.pth", map_location=torch.device('cpu')))

model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("model.pt")

