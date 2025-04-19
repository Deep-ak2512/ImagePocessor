import torch.nn as nn
import torch

#residual doe skip connection
class ResidualBlock(nn.Module):
  def __init__(self,in_ch,out_ch,kernel_size=3,padding=1):
    super(ResidualBlock,self).__init__()
    self.residue = nn.Sequential(
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
                                nn.BatchNorm2d(out_ch)                                                         
                              )

  def forward(self,x):
    return x+self.residue(x)
  
      
#Unet model for denosing 
class DenoiseNet(nn.Module):
  def __init__(self,in_ch=1,out_ch=1):
    super(DenoiseNet,self).__init__()
    self.preconv = nn.Conv2d(in_ch,16,kernel_size=3,padding=1)
    self.body1 = ResidualBlock(16,16,kernel_size=3,padding=1)
    self.down1 = nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1)
    self.body2 = ResidualBlock(32,32,kernel_size=3,padding=1)
    self.down2 = nn.Conv2d(32,48,kernel_size=3,stride=2,padding=1)
    self.bottleneck = ResidualBlock(48,48,kernel_size=3,padding=1)
    self.up2 = nn.Upsample(scale_factor=2,mode='nearest')
    self.body3 =  nn.Conv2d(48,32,kernel_size=3,padding=1)
    self.up1 = nn.Upsample(scale_factor=2,mode='nearest')
    self.body4 = nn.Conv2d(32,16,kernel_size=3,padding=1)
    self.last = nn.Conv2d(16,out_ch,kernel_size=1)

  def forward(self,x0):
    x11 = self.preconv(x0)
    x21 = self.body1(x11)
    x31= self.down1(x21)
    x41 = self.body2(x31)
    x51 = self.down2(x41)
    x_bn = self.bottleneck(x51)
    x52= self.up2(x_bn)
    x42 = x41+self.body3(x52)
    x32= self.up1(x42)
    x22 = x21+self.body4(x32)
    xout = self.last(x22)
    return xout

if __name__ == "__main__":
    model=  DenoiseNet()
    image = torch.rand(1,1,640,640)
    out= model(image)
    print(out.shape)    
