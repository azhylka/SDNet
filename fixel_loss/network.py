import torch.nn as nn
import torch
import math
import os
import yaml



class FixelNet(nn.Module):
    """Cascade Layer"""

    def __init__(self):
        super().__init__()
        self.casc = nn.Sequential(nn.Linear(45, 1000),
                                  nn.BatchNorm1d(1000),  
                                  nn.ReLU(inplace=True),  
                                  nn.Linear(1000,800),
                                  nn.BatchNorm1d(800),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(800,600),
                                  nn.BatchNorm1d(600),  
                                  nn.ReLU(inplace=True),
                                  nn.Linear(600,400),
                                  nn.BatchNorm1d(400),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(400,200),
                                  nn.BatchNorm1d(200),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(200,100),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(100,5))

    def forward(self, x):
        return self.casc(x)
    
    def init_weight(self, opts ,init_gain = 1.0):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                if opts.init_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif opts.init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(opts.activation))
                elif opts.init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity=opts.activation)
                elif opts.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=init_gain)

# def init_fixnet(opts):
#     #Initialising the network and loading the parameters:
#     # parameter_path = '/home/jxb1336/code/project_1/SDNet/SDNet/fixel_loss/checkpoints/sh-bignet/model_dict.pt'

#     net = FixelNet()
#     # net.load_state_dict(torch.load(parameter_path))
#     net.init_weight(opts)
    
#     #Setting the network to be used appropriately as loss
#     # net.eval()
#     # net.requires_grad_(False)
#     net.to(opts.device)
#     return net

def init_fixnet(opts):
    #Initialising the network and moving it to the correct device.
    print('Initialising Network')
    net = FixelNet()
    net = nn.DataParallel(net)
    net = net.to(opts.device)
    
    #Printing the layers and number of parameters of the network.
    print(net)
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'The number of parameters in the model is: {param_num}')

    model_save_path = os.path.join('checkpoints', opts.experiment_name, 'models')
    current_training_details = {'plot_offset':0, 'previous_loss':math.inf, 'best_loss':math.inf, 'best_val_ACC':0, 'global_epochs':0}
    
    if opts.continue_training:
        assert os.path.isdir(os.path.join('checkpoints', opts.experiment_name)), 'The experiment ' + opts.experiment_name + ''' does not exist so model parameters cannot be loaded. 
                                                                            Either change continue training flag to create another experiment, or change the experiment name
                                                                            to load an existing experiment'''

        net.load_state_dict(torch.load(os.path.join(model_save_path,'best_training.pth'))['net_state'])
        
        with open(os.path.join(model_save_path,'training_details.yml'), 'r') as file:
            training_details = yaml.load(file, yaml.loader.SafeLoader)

        #Refactor this code so it is only one line (possible dictionary comprehension)
        # Shouldn't have current_training_details and training_details as two seperate objects.
        current_training_details['best_loss'] = training_details['best loss']
        current_training_details['previous_loss'] = training_details['best loss']
        current_training_details['best_val_ACC'] = training_details['best ACC']
        current_training_details['global_epochs'] = training_details['epochs_count']

        
        
        
    else:
        # This code is related to training, not the model - should be in train.py or othe code.
        assert not os.path.isdir(os.path.join('checkpoints', opts.experiment_name)), f'The experiment {opts.experiment_name} already exists, please select another experiment name'
        os.mkdir(os.path.join('checkpoints', opts.experiment_name))
        os.mkdir(os.path.join('checkpoints', opts.experiment_name, 'models'))
        os.mkdir(os.path.join('checkpoints', opts.experiment_name, 'logs'))

    return net, None, param_num, current_training_details, model_save_path