import torch 
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

n_classes = 10

class vision_processesor(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_resnet = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(pretrained_resnet.children())[:4])

    def forward(self, x):
        x = self.model(x)
        return x


class brain_menu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=64*2, out_channels=64*64*1*3*3, kernel_size=8)   
        self.conv2 = nn.Conv2d(in_channels=64*2, out_channels=64*64*1*3*3, kernel_size=8)   

    def forward(self, prev_out, img_embed):
        x = torch.cat((prev_out, img_embed), dim=1)
        batch_size = x.shape[0]
        #c1 = self.conv1(x).view(-1, 64, 64, 3, 3)
        #c2 = self.conv2(x).view(-1, 64, 64, 3, 3)
        c1 = self.conv1(x).view(batch_size*64, 64, 3, 3)
        c2 = self.conv2(x).view(batch_size*64, 64, 3, 3)
        return c1, c2


class brain_kitchen(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, c1, c2, prev_out, vision_input):
        #x = torch.cat((prev_out, vision_input), dim=1)
        x = prev_out
        batch_size = int(c1.shape[0]/64)
  
        out = x.view(1, 64*batch_size, *x.shape[2:])
        #weight0 = c1.view(64*batch_size, 64, 3, 3).clone().detach().requires_grad_(True)
        out = F.conv2d(out, weight=c1, stride=1, padding=1, groups=batch_size)
        out = out.view(batch_size, 64, 8, 8)
        
        out = self.bn1(out)
        out = F.relu(out)
        
        out = out.view(1, 64*batch_size, *out.shape[2:])
        #weight1 = c2#.detach().clone()
        out = F.conv2d(out, weight=c2, stride=1, padding=1, groups=batch_size)
        out = out.view(batch_size, 64, 8, 8)
        
        out = x + F.relu(self.bn2(out))

        return out


class Agent(nn.Module):
    def __init__(self, max_iterations):
        super().__init__()
        self.max_iterations = max_iterations

        pretrained_resnet = models.resnet18(pretrained=True)

        #self.C1 = nn.Parameter(pretrained_resnet.layer1[0].conv1.weight)
        #self.C2 = nn.Parameter(pretrained_resnet.layer1[0].conv2.weight)
        self.C1 = nn.Parameter(torch.rand((64, 64, 3, 3))/((64*64*3*3)**0.5))
        self.C2 = nn.Parameter(torch.rand((64, 64, 3, 3))/((64*64*3*3)**0.5))
       
        self.out0 = nn.Parameter(torch.rand((1, 64, 8, 8)))

        self.vision_processesor = vision_processesor()
        self.brain_menu = brain_menu()
        self.brain_kitchen = brain_kitchen()
        self.head = nn.Sequential(
            #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            #nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, n_classes),
            nn.Sigmoid()
        )

    def forward(self, I):
        max_iterations = self.max_iterations

        img_embed = self.vision_processesor(I).detach()
        batch_size = img_embed.shape[0]
        c1 = self.C1.view(1, *self.C1.shape).repeat(batch_size, 1, 1, 1, 1)
        c2 = self.C2.view(1, *self.C2.shape).repeat(batch_size, 1, 1, 1, 1) 
        
        out = self.out0.repeat(batch_size, 1, 1, 1)

        probs_at_each_iteration = torch.zeros(max_iterations, batch_size, n_classes)

        for i in range(max_iterations):
            c1, c2 = self.brain_menu(out, img_embed)
            out = self.brain_kitchen(c1, c2, out, img_embed)
            probs = self.head(out)
            probs_at_each_iteration[i] = probs
            
        
        return probs_at_each_iteration
    