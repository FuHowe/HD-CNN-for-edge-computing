import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchsummary import summary


   

class myCNN_Train(nn.Module):
    
    def __init__(self, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, full_connect_len, classification_out):
        super(myCNN_Train, self).__init__()
        
        
        self.myCNN_Convs = nn.Sequential(
            nn.Conv2d(input_dim_1, output_dim_1, kernel_1), 
            nn.ReLU(),
            nn.MaxPool2d(pooling_1),
            
            nn.Conv2d(input_dim_2, output_dim_2, kernel_2),
            nn.ReLU(),
            nn.MaxPool2d(pooling_2),

            nn.Conv2d(input_dim_3, output_dim_3, kernel_3),
            nn.ReLU(),
            nn.MaxPool2d(pooling_3), 

            nn.Flatten(),
            nn.Linear(full_connect_len, classification_out)
        )

    
    def forward(self, indata):
        output = self.myCNN_Convs(indata)

        return output


class myCNN_Para(nn.Module):
    
    def __init__(self, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, classification_out):
        super(myCNN_Para, self).__init__()
        
        
        self.myCNN_Convs = nn.Sequential(
            nn.Conv2d(input_dim_1, output_dim_1, kernel_1), 
            nn.ReLU(),
            nn.MaxPool2d(pooling_1),

            nn.Conv2d(input_dim_2, output_dim_2, kernel_2),
            nn.ReLU(),
            nn.MaxPool2d(pooling_2),
            
            nn.Conv2d(input_dim_3, output_dim_3, kernel_3),
            nn.ReLU(),
            nn.MaxPool2d(pooling_3), 

            nn.Flatten(),
        )

    
    def forward(self, indata):
        output = self.myCNN_Convs(indata)

        return output



class myDATA():
    def __init__(self):   
        self.head_len = 0 
        self.con_layers = 0 
        self.para_size = 0 
        self.image_size = 0 
        
        self.con_layer_input_dimension_1 = 0 
        self.con_layer_output_dimension_1 = 0 
        self.con_layer_kernel_1 = 0 
        self.con_layer_pooling_1 = 0 

        self.con_layer_input_dimension_2 = 0 
        self.con_layer_output_dimension_2 = 0 
        self.con_layer_kernel_2 = 0 
        self.con_layer_pooling_2 = 0 

        self.con_layer_input_dimension_3 = 0 
        self.con_layer_output_dimension_3 = 0 
        self.con_layer_kernel_3 = 0 
        self.con_layer_pooling_3 = 0 
        
        self.full_connect_len = 0 
        self.classification_out = 0 

        self.weight_bias = [] 

        
