import torch 
from torch import nn
import torch.nn.functional as F

vision_processesor_dim = 500 #Embedding dimension of input image
brain_kitchen_out_dim = 320 # Dimension of output of kitchen module
brain_menu_C_dim = 500 * (vision_processesor_dim + 2*brain_kitchen_out_dim) #Dimension of computation vector
n_computations = 3
n_classes = 10

"""
device1 = torch.device("cuda:0") # vision_processor and brain_kitchen head
device2 = torch.device("cuda:0") # brain_menu
device3 = torch.device("cuda:0") # brain_kitchen
"""

class vision_processesor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*7*7, vision_processesor_dim),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class brain_menu(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = vision_processesor_dim + brain_kitchen_out_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, brain_menu_C_dim),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )        

    def forward(self, prev_out, vision_input):
        x = torch.cat((prev_out, vision_input), dim=1)
        c = self.model(x)

        return c

class brain_kitchen(nn.Module):
    def __init__(self):
        super().__init__()
        """input_dim = vision_processesor_dim + brain_menu_C_dim + brain_kitchen_out_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(500, brain_kitchen_out_dim),
            nn.Dropout(p=0.1)
        )"""
        #self.W1 = nn.Parameter(torch.rand((1, 500, vision_processesor_dim + brain_kitchen_out_dim))/((vision_processesor_dim + brain_kitchen_out_dim)**0.5))
        self.b1 = nn.Parameter(torch.rand((1, 500))/((vision_processesor_dim + brain_kitchen_out_dim)**0.5))

        #self.W2 = nn.Parameter(torch.rand((1, brain_kitchen_out_dim, 500))/(500**0.5))
        self.b2 = nn.Parameter(torch.rand((1, brain_kitchen_out_dim))/(500**0.5))


    def forward(self, c, prev_out, vision_input):
        x = torch.cat((prev_out, vision_input), dim=1)
        batch_size = x.shape[0]

        c1_dim = 500*(vision_processesor_dim + brain_kitchen_out_dim)
        c1 = c[:, :c1_dim].view(batch_size, 500, vision_processesor_dim + brain_kitchen_out_dim)
        W1 = c1#self.W1*c1

        c2 = c[:, c1_dim:].view(batch_size, brain_kitchen_out_dim, 500)
        W2 = c2#self.W2*c2

        x = F.relu((W1 @ x.unsqueeze(2)).squeeze() + self.b1)
        out = F.relu((W2 @ x.unsqueeze(2)).squeeze() + self.b2)
        
        return out


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.C0 = nn.Parameter(torch.rand((brain_menu_C_dim)))
        self.out0 = nn.Parameter(torch.rand((1, brain_kitchen_out_dim)))

        self.vision_processesor = vision_processesor()
        self.brain_menu = brain_menu()
        self.brain_kitchen = brain_kitchen()
        self.head = nn.Sequential(
            nn.Linear(brain_kitchen_out_dim, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, I):
        img_embed = self.vision_processesor(I)
        batch_size = img_embed.shape[0]
        c = self.C0.repeat(batch_size, 1)
        out = self.out0.repeat(batch_size, 1)

        for i in range(n_computations):
            c = self.brain_menu(out, img_embed)
            out = F.normalize(self.brain_kitchen(c, out, img_embed))
 
        prob = torch.log(self.head(out))
        return prob
    