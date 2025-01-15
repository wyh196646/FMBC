#self define linear model
import torch.nn as nn

class linear_probe(nn.Module):
    def __init__(self, in_dim,latent_dim, out_dim):
        super(linear_probe, self).__init__()
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, out_dim)

    def forward(self, images, img_coords, pad_mask):
        x =  self.fc1(images)
        logits = self.fc2(x)
        logits = logits.squeeze().unsqueeze(0)
        return logits