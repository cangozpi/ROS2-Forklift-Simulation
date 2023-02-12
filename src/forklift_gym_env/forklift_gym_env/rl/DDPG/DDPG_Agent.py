import torch
from torch import nn
from copy import deepcopy


class DDPG_Agent(): #TODO: make this extend a baseclass (ABC) of Agent and call its super().__init__()
    """
    Refer to https://spinningup.openai.com/en/latest/algorithms/ddpg.html for implementation details.
    """
    def __init__(self, obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims, lr, epsilon, epsilon_decay, gamma, tau):
        self.mode = "train"
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, action_dim, actor_hidden_dims)
        self.critic = Critic(obs_dim, action_dim, critic_hidden_dims)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr)

        self.critic_loss_func = torch.nn.MSELoss()

        self.train()
    

    def choose_action(self, obs):
        """
        Returns actions that are normalized in the [-1, 1] range. Don't forget to scale them up for your need.
        """
        action = self.actor(obs)
        # during training add noise to the action
        if self.mode == "train":
            noise = self.epsilon * torch.normal(mean=torch.tensor(0.0),std=torch.tensor(1.0))
            action += noise

            # decay epsilon
            self.epsilon -= self.epsilon_decay
        
        # clip action [-1, 1]
        action = torch.clip(action, -1.0, 1.0)

        return action

    
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        """
        Given a batch of data(i.e. (s,a,r,s',d)) performs training/model update on the DDPG agent's DNNs.
        """
        # Compute Q_targets
        Q_targets = reward_batch + \
            ((1 - terminal_batch.int()) * self.gamma * self.critic_target(next_state_batch, self.actor_target(next_state_batch)))

        # Update Q function (Critic)
        self.critic.zero_grad()
        Q_value_preds = self.critic(next_state_batch, action_batch)
        critic_loss = self.critic_loss_func(Q_value_preds, Q_targets)
        critic_loss.backward()
        self.optim_critic.step()


        # Update policy (Actor)
        self.actor.zero_grad()
        actor_loss = - torch.mean(self.critic(state_batch, self.actor(state_batch)))
        actor_loss.backward()
        self.optim_actor.step()

        
        # Update target networks with polyak averaging
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                (target_param.data * self.tau) + (param.data * (1.0 - self.tau)) 
                    )

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                (target_param.data * self.tau) + (param.data * (1.0 - self.tau)) 
                    )
        
        return critic_loss, actor_loss


    def train(self):
        """
        Sets actor and target networks to model.train() mode of torch.
        Also makes choose_action return actions with added noise for training (exploration reasons).
        """
        self.mode = "train"

        self.actor.train()
        self.critic.train()
        # TODO: not sure if target networks should be set to train or eval ??
        self.actor_target.train()
        self.critic_target.train() 
    

    def eval(self):
        """
        Sets actor and target networks to model.train() mode of torch.
        Also makes choose_action return actions with no noise added for testing.
        """
        self.mode = "eval"

        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval() 

    
    def save_model(self):
        """
        Saves the current state of the neural network models of the actor and the critic of the DDPG agent.
        """
        torch.save(
            self.actor.state_dict(),
            'actor.pkl'
        )
        torch.save(
            self.critic.state_dict(),
            'critic.pkl'
        )
    

    def load_model(self):
        """
        Loads the previously saved states of the actor and critic models to the current DDPG agent.
        """
        self.actor.load_state_dict(
            torch.load('actor.pkl')
        )

        self.critic.load_state_dict(
            torch.load('critic.pkl')
        )


    


class Actor(nn.Module):
    def __init__(self, obs_dim:tuple, action_dim:tuple, hidden_dims:list):
        """
        Inputs:
            obs_dim (tuple): dimension of the observations. (e.g. (C, H, W), for and RGB image observation).
            action_dim (tuple): dimension of the action space.
            hidden_dims (list): holds dimensions of the hidden layers excluding the input layer 
                and the input and the output dimensions.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = []
        prev_dim = self.obs_dim # shape of the flattened input to the network
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(torch.nn.Flatten())
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, action_dim))
        layers.append(torch.nn.Tanh()) 
                
        self.model_layers = torch.nn.ModuleList(layers)
    
    
    def forward(self, x):
        """
        Inputs:
            x (torch.Tensor): a batch of states.
        """
        # pass input through the layers of the model
        out = x # --> [B, (*self.obs_dim)]
        for layer in self.model_layers:
            out = layer(out.float())
        
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim:tuple, action_dim:tuple, hidden_dims:list):
        """
        Inputs:
            obs_dim (tuple): dimension of the observations. (e.g. (C, H, W), for and RGB image observation).
            action_dim (tuple): dimension of the actions.
            hidden_dims (list): holds dimensions of the hidden layers excluding the input layer 
                and the input and the output dimensions.
        """
        super().__init__()
        self.obs_dim = (obs_dim, action_dim) 
        self.output_dim = 1 # Q(s,a) is of shape (1,)

        layers = []
        prev_dim = sum(self.obs_dim) # shape of the flattened input to the network
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(torch.nn.Flatten())
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, self.output_dim))
                
        self.model_layers = torch.nn.ModuleList(layers)
    
    
    def forward(self, state_batch, action_batch):
        """
        Inputs:
            state_batch (torch.Tensor): a batch of states.
            action_batch (torch.Tensor): a batch of actions taken at the corresponding states in the state_batch.
        """
        # pass input through the layers of the model
        batch_dim = state_batch.shape[0]
        state_batch = state_batch.reshape(batch_dim, -1)
        action_batch = action_batch.reshape(batch_dim, -1)
        out = torch.concat((state_batch, action_batch), dim=1) # --> [B, (*self.obs_dim)]

        for layer in self.model_layers:
            out = layer(out.float())
        
        return out