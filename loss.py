import torch.nn.functional as F
import torch
import torch.nn as nn

# unspuervised_loss for mathing SNR of weak band to PAN band
import torch.nn.functional as F
import torch
import torch.nn as nn

class SNRLoss(nn.Module):
  def __init__(self):
    super(SNRLoss,self).__init__()
    self.l1_loss = nn.L1Loss()
    
  def forward(self,output, target):   
    mean_out = output.mean([2, 3], keepdim=True)
    std_out = output.std([2, 3], keepdim=True)
    mean_target = target.mean([2, 3], keepdim=True)
    std_target = target.std([2, 3], keepdim=True)
    mean_loss = self.l1_loss(mean_out, mean_target)
    std_loss = self.l1_loss(std_out, std_target)
    l1_loss = self.l1_loss(output, target)
    return l1_loss + 0.5 * mean_loss + 0.5 * std_loss

if __name__ == "__main__":
    model=  SNRLoss()
    image1 = torch.rand(1,1,640,640)
    image2 = torch.rand(1,1,640,640)
    out= model(image1,image2)
    print(out)   
