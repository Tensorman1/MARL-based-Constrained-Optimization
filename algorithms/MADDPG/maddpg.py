"""
This is the implementation of Multi-Agent Deep Deterministic Policy Gradients for Network Resource Optimization under constraints
The core princples have been derived from - Lowe et. al, "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments."
Link to the GitHub repository - https://github.com/openai/maddpg/tree/master".

Claude 3.5 Haiku and ChatGPT-4o was used to debug the visualizations.

If you have any questions about the specific adaptation to network resource allocation or find any potential error in the 
implementation please reach out. Otherwise this has been created for any and all reuse without restrictions. If you find this
to be useful, please consider citing the repository as a hyperlink for now until the manuscript is in-review.

Things to Note - The run time depends on the compute resources available. This was evaluated using a humble Ryzen 5 processor with
no GPU. The run time was 2.5 hours for 250 episodes. Results obtained from this for the network resource management problem
might benefit from running more episodes or increasing the neighbourhood size, or running more training smaples per-episode. 
Application-specific heuristic modifications apply for other adaptations.
"""
# MADDPG Implementation with Neighborhood Communication
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, local_state_dim, local_action_dim, n_neighbors):
        super(Critic, self).__init__()
        """Only consider local state + neighborhood actions (not all agents, this has does not fully abide by some
        of the equilibria guarantees provided in the dissertation.
        Global information access helps mitigate the non-stationarity to some-extent. The proof presented has not been
        peer-reviewed and accepted. There are a lot of assumptions and approximations. 
        Do not take for granted.
        """
        input_dim = local_state_dim + local_action_dim * (2 * n_neighbors + 1)  # +1 for self
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        return self.net(x)
"""Details on the use of Ornstein-Uhlenbeck noise has been included in the dissertation and the source paper for which is
https://arxiv.org/pdf/1509.02971v6. This has not been used for applications on wireless communications or computational control
of networks yet (as far as we know).
"""
class OUNoise:
    def __init__(self, num_agents, action_dim, theta=0.15, sigma=0.2):
        self.theta = theta
        self.sigma = sigma
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.reset()
        
    def reset(self):
        self.noise = np.zeros((self.num_agents, self.action_dim))
        
    def sample(self):
        dx = self.theta * -self.noise + self.sigma * np.random.randn(*self.noise.shape)
        self.noise += dx
        return self.noise
    
    def decay(self):
        self.sigma *= 0.9995

"""This is the implementation of MADDPG"""
class MADDPG:
    def __init__(self, num_agents, local_state_dim, action_dim, neighborhood_size=3):
        # Hyperparameters (modify as requried)
        self.LR_ACTOR = 5e-5
        self.LR_CRITIC = 1e-4
        self.GAMMA = 0.99
        self.TAU = 0.01
        self.NOISE_DECAY = 0.9998
        self.POLICY_UPDATE_FREQ = 2
        
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.neighborhood_size = neighborhood_size
        
        # Adjacency matrix for neighborhood connections
        self.adjacency = self._create_neighborhood_graph()
        
        # Initialize actors and critics
        self.actors = [Actor(local_state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(local_state_dim, action_dim, neighborhood_size) for _ in range(num_agents)]
        
        # Target networks
        self.target_actors = [Actor(local_state_dim, action_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(local_state_dim, action_dim, neighborhood_size) for _ in range(num_agents)]
        
        # Here, a duplicate of the initial scaling factorsfrw are made.
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=self.LR_ACTOR) 
                                 for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=self.LR_CRITIC)
                                  for critic in self.critics]
        
        self.noise = OUNoise(num_agents, action_dim)
        self.lyapunov_history = []
        
        # Device mnagement - This is for advanced compute, if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()
        
    def _move_to_device(self):
        for i in range(self.num_agents):
            self.actors[i] = self.actors[i].to(self.device)
            self.critics[i] = self.critics[i].to(self.device)
            self.target_actors[i] = self.target_actors[i].to(self.device)
            self.target_critics[i] = self.target_critics[i].to(self.device)

    def _create_neighborhood_graph(self):
        """Adjacency matrix for a ring topology with neighborhood_size neighbors on each side"""
        adj = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            # Connect to neighborhood_size neighbors on each side (wrap around)
            for j in range(1, self.neighborhood_size + 1):
                adj[i, (i + j) % self.num_agents] = 1
                adj[i, (i - j) % self.num_agents] = 1
        return adj
    
    def _get_neighborhood(self, agent_idx):
        """Indices of an agent's neighbors"""
        return np.where(self.adjacency[agent_idx] == 1)[0]

    def act(self, states, episode, add_noise=True):
        actions = []
        noise = self.noise.sample() * self.noise_decay(episode) if add_noise else 0
        
        for i, actor in enumerate(self.actors):
            state = torch.FloatTensor(states[i]).to(self.device)
            with torch.no_grad():
                action = actor(state).cpu().numpy()
            if add_noise:
                action = np.clip(action + noise[i], -1, 1)
            actions.append(action)
        return np.array(actions)

    def update(self, experiences, agent_idx, episode):
        if len(self.lyapunov_history) > 100:
            self.adapt_learning_rate()
        
        states, actions, rewards, next_states, dones = experiences
        
        
        batch_size = states.shape[0]
        
       
        neighborhood = self._get_neighborhood(agent_idx)
        
        # Current agent state
        agent_state = states[:, agent_idx].to(self.device)
        agent_action = actions[:, agent_idx].to(self.device)
        agent_reward = rewards[:, agent_idx].to(self.device)
        agent_next_state = next_states[:, agent_idx].to(self.device)
        agent_done = dones[:, agent_idx].to(self.device)
        
        # Neighborhood actions (current and next)
        neighborhood_actions = torch.cat([actions[:, i] for i in np.append(neighborhood, agent_idx)], dim=1).to(self.device)
        
        # Predicted next actions for neighbors using target networks
        neighborhood_next_actions = []
        for i in np.append(neighborhood, agent_idx):
            next_state_i = next_states[:, i].to(self.device)
            next_action_i = self.target_actors[i](next_state_i)
            neighborhood_next_actions.append(next_action_i)
        neighborhood_next_actions = torch.cat(neighborhood_next_actions, dim=1)
        
        # 3. Updating critic
        with torch.no_grad():
            target_q = agent_reward.unsqueeze(1) + (1 - agent_done.unsqueeze(1)) * self.GAMMA * self.target_critics[agent_idx](
                agent_next_state, 
                neighborhood_next_actions
            )
            
        current_q = self.critics[agent_idx](agent_state, neighborhood_actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
        self.critic_optimizers[agent_idx].step()
        
        # 4. Delayed actor update
        actor_loss = torch.tensor(0.0)
        if episode % self.POLICY_UPDATE_FREQ == 0:
            # For actor update,  current agent's actions from its policy has been used
            current_agent_actions = self.actors[agent_idx](agent_state)
            
            # Collection of Actions from the neighbourhood (using their current policies)
            actions_for_critic = []
            for i, idx in enumerate(np.append(neighborhood, agent_idx)):
                if idx == agent_idx:
                    # Using the actions from current agent's policy
                    actions_for_critic.append(current_agent_actions)
                else:
                    # Using actions from replay buffer for other agents
                    actions_for_critic.append(actions[:, idx].to(self.device))
            actions_for_critic = torch.cat(actions_for_critic, dim=1)
            
            # Actor loss
            actor_loss = -self.critics[agent_idx](agent_state, actions_for_critic).mean()
            
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
            self.actor_optimizers[agent_idx].step()
            
            self._soft_update_targets(agent_idx)
            
            # Lyapunov stability check (for more details please read paper)
            lyapunov = 0.5 * (critic_loss.item() + 0.1 * actor_loss.item())
            self.lyapunov_history.append(lyapunov)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
        }
            
    def _soft_update_targets(self, agent_idx):
        """Soft update of target networks for a specific agent"""
        for target_param, param in zip(self.target_actors[agent_idx].parameters(), 
                                      self.actors[agent_idx].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.TAU) + param.data * self.TAU
            )
            
        for target_param, param in zip(self.target_critics[agent_idx].parameters(), 
                                      self.critics[agent_idx].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.TAU) + param.data * self.TAU
            )

    def adapt_learning_rate(self):
        """Adaptive learning rate based on Lyapunov stability"""
        if self.lyapunov_history[-1] > 1.2 * np.mean(self.lyapunov_history[-100:-1]):
            for optimizer in self.actor_optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
            for optimizer in self.critic_optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
            self.lyapunov_history = []

    def noise_decay(self, episode):
        return max(0.05, self.NOISE_DECAY ** episode)

"""The following lines will set-up the large network environment with 10 base stattions.
"Large" is a quite a relative word. WRT the previous implementations with 3 agents,
this is large.
"""
