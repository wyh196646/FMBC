#self define linear model
import torch.nn as nn

class linear_probe(nn.Module):
    def __init__(self, in_dim,latent_dim, out_dim):
        super(linear_probe, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
       


    def forward(self, images, img_coords, pad_mask):
        logits =  self.fc1(images)
        logits = logits.mean(dim=1)
        logits = logits.squeeze().unsqueeze(0)
        return logits

