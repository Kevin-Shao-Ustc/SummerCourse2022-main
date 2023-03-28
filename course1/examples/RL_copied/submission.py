from pathlib import Path
import os

current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, 'actor_state_dict.pt')



import torch
import torch.nn as nn

#TODO: setup models 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(3, 1, 3, stride=2, padding='valid'),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(361, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
            
      
    def forward(self, X):
        out1 = self.net1(X.reshape(-1, 1, 40, 40))
        action_batch = self.net2(out1.reshape(X.shape[0], -1))
        action_batch[:, 0] = torch.tanh(action_batch[:,0])*150+50
        action_batch[:, 1] = torch.tanh(action_batch[:, 1])*30
        return action_batch

model = Net()
loaded_actor_state = torch.load(model_path)
model.load_state_dict(loaded_actor_state)

def my_controller(observation, action_space, is_act_continuous=True):

    obs_array = torch.tensor(observation['obs']['agent_obs']).float().reshape(1, -1)
    action = model(obs_array)

    return [[action[0][0].item()], [action[0][1].item()]]
