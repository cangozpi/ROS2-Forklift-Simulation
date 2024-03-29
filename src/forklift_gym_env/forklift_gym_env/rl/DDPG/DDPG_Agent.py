import torch
from torch import nn
from copy import deepcopy
from forklift_gym_env.rl.DDPG.utils import log_gradients_in_model, log_training_losses, log_residual_variance


class DDPG_Agent(): #TODO: make this extend a baseclass (ABC) of Agent and call its super().__init__()
    """
    Refer to https://spinningup.openai.com/en/latest/algorithms/ddpg.html for implementation details.
    """
    def __init__(self, obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims, actor_lr, critic_lr, initial_epsilon, epsilon_decay, min_epsilon, act_noise, target_noise, clip_noise_range, gamma, tau, max_action, policy_update_delay=2, logger=None, log_full_detail=False):
        self.logger = logger
        self.log_full_detail = log_full_detail

        self.mode = "train"
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.clip_noise_range = clip_noise_range
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action # actions returned is in range [-max_action, max_action]

        self.actor = Actor(obs_dim, action_dim, actor_hidden_dims, max_action)
        self.critic = Critic(obs_dim, action_dim, critic_hidden_dims)

        self.actor_target = Actor(obs_dim, action_dim, actor_hidden_dims, max_action)
        self.critic_target = Critic(obs_dim, action_dim, critic_hidden_dims)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # self.actor_target = deepcopy(self.actor)
        # self.critic_target = deepcopy(self.critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for actor_p, critic_p in zip(self.actor_target.parameters(), self.critic_target.parameters()):
            actor_p.requires_grad = False
            critic_p.requires_grad = False

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), critic_lr)

        self.critic_loss_func = torch.nn.MSELoss()

        self.train_mode()

        self.policy_update_delay = policy_update_delay
        self._n_updates = 0
    

    def choose_action(self, obs):
        """
        Returns actions that are normalized in the [-1, 1] range. Don't forget to scale them up for your need.
        Note that obs and goal_state are torch.Tensor with no batch dimension.
        """
        obs = torch.unsqueeze(obs, dim=0) # batch dimension of 1
        # during training add noise to the action
        if self.mode == "train":
            self.eval_mode()
            action = self.actor(obs)

            noise = torch.clamp(self.epsilon * torch.randn_like(action) * self.act_noise, -self.clip_noise_range, self.clip_noise_range)
            action += noise
            action = torch.clip(action, -self.max_action, self.max_action)

            # noise = self.epsilon * torch.normal(mean=torch.tensor(0.0),std=torch.tensor(1.0))
            # noise = self.epsilon * torch.normal(mean=torch.tensor(0.0),std=torch.tensor(0.2))
            # action = action + noise

            # Decay epsilon
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
            
            self.train_mode()

        else:
            action = self.actor(obs)
        
        # clip action [-1, 1]
        action = torch.clip(action, -self.max_action, self.max_action)

        return action

    
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        """
        Given a batch of data(i.e. (s,a,r,s',d)) performs training/model update on the DDPG agent's DNNs.
        """
        self.train_mode()

        # Compute Q_targets
        with torch.no_grad():
            # Add noise to actions for "Target Policy Smoothing"
            action = self.actor_target(next_state_batch)
            noise = torch.clamp(self.epsilon * torch.randn_like(action) * self.target_noise, -self.clip_noise_range, self.clip_noise_range)
            noisy_action = action + noise
            clipped_noise_action = torch.clip(noisy_action, -self.max_action, self.max_action)
            # Find Q value estimates for the next states
            Q_targets = reward_batch + \
                ((1 - terminal_batch.int()) * self.gamma * self.critic_target(next_state_batch, clipped_noise_action))

        # Update Q function (Critic)
        Q_value_preds = self.critic(state_batch, action_batch)
        critic_loss = self.critic_loss_func(Q_value_preds, Q_targets)

        self.critic.zero_grad()
        critic_loss.backward()

        #Gradient Value Clipping
        # torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1.0)
        # Gradient Norm Clipping
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)

        self.optim_critic.step()

        # Log Critic's stats to tensorboard
        log_gradients_in_model(self.critic, self.logger, self._n_updates, "Critic", self.log_full_detail)
        log_training_losses(critic_loss.cpu().detach(), self.logger, self._n_updates, "Critic")
        # log_residual_variance(Q_value_preds.detach().cpu().numpy(), Q_targets.detach().cpu().numpy(), self.logger, self._n_updates, "Critic's Q_pred and Q_target")

        # ------------------------------------------- Delayed Policy Update:
        if self._n_updates % self.policy_update_delay == 0:
            # Freeze Q-networks so you don't waste computational effort computing gradients for them during the policy learning step.
            for critic_p in self.critic.parameters():
                critic_p.requires_grad = False

            # Update policy (Actor)
            actor_loss = - torch.mean(self.critic(state_batch, self.actor(state_batch)))

            self.actor.zero_grad()
            actor_loss.backward()

            #Gradient Value Clipping
            # torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=1.0)
            # Gradient Norm Clipping
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)

            self.optim_actor.step()

            # Log Actor's stats to tensorboard
            log_gradients_in_model(self.actor, self.logger, self._n_updates, "Actor", self.log_full_detail)
            log_training_losses(actor_loss.cpu().detach(), self.logger, self._n_updates // self.policy_update_delay, "Actor")
             
            # Unfreeze Q-networks so you can optimize it at the next DDPG update.
            for critic_p in self.critic.parameters():
                critic_p.requires_grad = True

            # Update target networks with polyak averaging
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    (target_param.data * self.tau) + (param.data * (1.0 - self.tau)) 
                        )

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    (target_param.data * self.tau) + (param.data * (1.0 - self.tau)) 
                        )
        
        self._n_updates += 1
        

        
        return critic_loss.cpu().detach(), actor_loss.cpu().detach()


    def train_mode(self):
        """
        Sets actor and target networks to model.train() mode of torch.
        Also makes choose_action return actions with added noise for training (exploration reasons).
        """
        self.mode = "train"

        self.actor.train()
        self.critic.train()
        # TODO: not sure if target networks should be set to train or eval ??
        self.actor_target.eval()
        self.critic_target.eval() 
    

    def eval_mode(self):
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

        # Save target networks too
        torch.save(
            self.critic_target.state_dict(),
            'critic_target.pkl'
        )
        torch.save(
            self.actor_target.state_dict(),
            'actor_target.pkl'
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

        # Load target networks too
        self.actor_target.load_state_dict(
            torch.load('actor_target.pkl')
        )
        self.critic_target.load_state_dict(
            torch.load('critic_target.pkl')
        )



class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims:list, max_action):
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
        self.max_action = max_action # actions returned is in range [-max_action, max_action]

        layers = []
        prev_dim = self.obs_dim  # shape of the flattened input to the network
        for i, hidden_dim in enumerate(hidden_dims):
            # if i == 0:
            #     layers.append(torch.nn.Flatten())
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        # layers.append(torch.nn.LayerNorm(prev_dim)) # Add Norm to mitigate tanh saturation problem
        layers.append(torch.nn.BatchNorm1d(prev_dim)) # Add Norm to mitigate tanh saturation problem
        layers.append(torch.nn.Linear(prev_dim, action_dim))
        layers.append(torch.nn.Tanh()) 
                
        self.model_layers = torch.nn.ModuleList(layers)
    
    
    def forward(self, obs_batch):
        """
        Inputs:
            obs_batch (torch.Tensor): a batch of states.
        """
        # pass input through the layers of the model
        batch_dim = obs_batch.shape[0]
        out = obs_batch.reshape(batch_dim, -1) # --> [B, obs_dim]

        for layer in self.model_layers:
            out = layer(out)
        
        out = out * self.max_action
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims:list):
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
            # if i == 0:
            #     layers.append(torch.nn.Flatten())
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        # layers.append(torch.nn.LayerNorm(prev_dim)) # Add Norm to mitigate tanh saturation problem
        layers.append(torch.nn.BatchNorm1d(prev_dim)) # Add Norm to mitigate tanh saturation problem
        layers.append(torch.nn.Linear(prev_dim, self.output_dim))
        # layers.append(torch.nn.ReLU())
                
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
        out = torch.concat((state_batch, action_batch), dim=1) # --> [B, (self.obs_dim + action_dim)]

        for layer in self.model_layers:
            out = layer(out)
        
        return out