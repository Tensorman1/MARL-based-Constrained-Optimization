# Environment of the large wireless network
class LargeWirelessNetworkEnv:
    def __init__(self, num_base_stations=10, num_users_per_bs=5, grid_size=1000):
        self.num_base_stations = num_base_stations
        self.num_users_per_bs = num_users_per_bs
        self.total_users = num_base_stations * num_users_per_bs
        self.grid_size = grid_size
        
        """Physical constants (the units really don't matter. Neither do specific numbers as this is not a 
        specific network type for a particular application, or a specific standard/protocol. This has been made general for natural extensions into other
        varieties).

       The following lines have imposed deterministic constraints. A more realistic etension would be to introduce chance-constraints.
        However, inclusion of the sampling process will amke this already stochastic process significantly more so.
        Itv will also signifcantly increase the computational complexity. It is recommended, that a separate program be run offline 
        to generate the probabilistic constraint samples. Alternatively, an even simpler approach of using deterministic outputs, e.g. mean, from 
        the chance-constraint distribution may be used.
        
        Path loss exponent has been chosen a constant. In reality, different users will have different PL exponents,
        depending on the urban setting the network serves.
        """
        self.MAX_POWER = 50  # Watts 
        self.MIN_SNR = 10    # dB
        self.MAX_LATENCY = 100  # ms
        self.PATHLOSS_EXPONENT = 3.5 
        self.BANDWIDTH = 20  # MHz
        self.NOISE_FLOOR = -100  # dBm
        
        self.max_steps = 100
        self.current_step = 0
        self.done = False
        
        # Initialize locations
        self._initialize_locations()
        
        # Initialize network state parameters
        self.channel_gains = None
        self.user_demands = None
        self.current_power = np.zeros(num_base_stations)
        self.current_snr = np.zeros(num_base_stations)
        self.current_latency = np.zeros(num_base_stations)
        self.current_throughput = np.zeros(num_base_stations)
        self.current_interference = np.zeros((num_base_stations, num_users_per_bs))
        
        # Store per-episode metrics
        self.episode_metrics = {
            'snr': [],
            'latency': [],
            'throughput': [],
            'power': [],
            'fairness': [],
            'rewards': []
        }
        
    def _initialize_locations(self):
        """Initialize BS and user locations in a grid"""
        # Here, the base stations have been placed in a grid pattern with some random displacement
        bs_spacing = self.grid_size / np.sqrt(self.num_base_stations)
        grid_dim = int(np.ceil(np.sqrt(self.num_base_stations)))
        
        self.bs_locations = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                if len(self.bs_locations) < self.num_base_stations:
                    x = i * bs_spacing + np.random.uniform(-bs_spacing/4, bs_spacing/4)
                    y = j * bs_spacing + np.random.uniform(-bs_spacing/4, bs_spacing/4)
                    self.bs_locations.append((x, y))
        
        # Initialize user locations - will be reset for each episode
        self.user_locations = np.zeros((self.num_base_stations, self.num_users_per_bs, 2))
        
    def _calculate_channel_gains(self):
        """Here, distance and shadowing have been considered when calculating the channel gains"""
        gains = np.zeros((self.num_base_stations, self.num_base_stations, self.num_users_per_bs))
        
        for bs_idx in range(self.num_base_stations):
            bs_x, bs_y = self.bs_locations[bs_idx]
            
            for user_bs_idx in range(self.num_base_stations):
                for user_idx in range(self.num_users_per_bs):
                    user_x, user_y = self.user_locations[user_bs_idx, user_idx]
                    
                    # Distance
                    distance = np.sqrt((bs_x - user_x)**2 + (bs_y - user_y)**2)
                    # Minimum distance to avoid infinite gain (indeterminate fraction)
                    distance = max(distance, 1)
                    
                    # Path loss (dB) = 10 * n * log10(d) where n is path loss exponent
                    path_loss_db = 10 * self.PATHLOSS_EXPONENT * np.log10(distance)
                    
                    # shadow fading (log-normal distribution)
                    shadow_fading_db = np.random.normal(0, 8)  # 8 dB standard deviation
                    
                    # Total loss in dB
                    total_loss_db = path_loss_db + shadow_fading_db
                    
                    # Convert to linear scale gain
                    gain = 10 ** (-total_loss_db / 10)
                    gains[bs_idx, user_bs_idx, user_idx] = gain
        
        return gains
        
    def reset(self):
        self.current_step = 0
        self.done = False
        
        # Clear per-episode metrics
        for key in self.episode_metrics:
            self.episode_metrics[key] = []
        
        # Randomize user locations for each BS
        for bs_idx in range(self.num_base_stations):
            bs_x, bs_y = self.bs_locations[bs_idx]
            # Users are distributed within cell radius
            cell_radius = self.grid_size / (2 * np.sqrt(self.num_base_stations))
            
            for user_idx in range(self.num_users_per_bs):
                # Random angle and distance from BS (See network topology plot)
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, cell_radius)
                
                # Convert to Cartesian coordinates
                user_x = bs_x + distance * np.cos(angle)
                user_y = bs_y + distance * np.sin(angle)
                
                # EEnsure within grid boundaries - Generated by Claude 3.5 Haiku
                user_x = np.clip(user_x, 0, self.grid_size)
                user_y = np.clip(user_y, 0, self.grid_size)
                
                self.user_locations[bs_idx, user_idx] = [user_x, user_y]
        
        # Channel gains are calculated based on the location
        self.channel_gains = self._calculate_channel_gains()
        
        # Random user demands between 0 and 1
        self.user_demands = np.random.uniform(0, 1, (self.num_base_stations, self.num_users_per_bs))
        
        # Initialization
        self.current_power = np.zeros(self.num_base_stations)
        self.current_snr = np.ones(self.num_base_stations) * self.MIN_SNR
        self.current_latency = np.ones(self.num_base_stations) * self.MAX_LATENCY
        self.current_throughput = np.zeros(self.num_base_stations)
        self.current_interference = np.zeros((self.num_base_stations, self.num_users_per_bs))
        
        # Returns a state for each base station
        states = []
        for i in range(self.num_base_stations):
            states.append(self._get_state(i))
        return states
        
    def _get_state(self, bs_idx):
        """Get local state for a base station"""
        # 1. Channel gains to own users
        own_channel_gains = self.channel_gains[bs_idx, bs_idx, :]
        
        # 2. Strongest interfering BSs' channel gains to own users
        # For each user,  top 2 interfering BSs are sought
        interference_gains = []
        for user_idx in range(self.num_users_per_bs):
            interfering_gains = []
            for interfering_bs in range(self.num_base_stations):
                if interfering_bs != bs_idx:
                    interfering_gains.append(self.channel_gains[interfering_bs, bs_idx, user_idx])
            
            # Top 2 interferers are now chosen here
            interfering_gains.sort(reverse=True)
            interference_gains.extend(interfering_gains[:2])
        
        # 3. User demands
        user_demands = self.user_demands[bs_idx]
        
        # 4. Current metrics for this BS
        metrics = np.array([
            self.current_power[bs_idx], 
            self.current_snr[bs_idx], 
            self.current_latency[bs_idx], 
            self.current_throughput[bs_idx]
        ])
        
        # 5. Current interference levels for users
        interference = self.current_interference[bs_idx]
        
        # Concatenate all to form state
        state = np.concatenate([
            own_channel_gains,
            np.array(interference_gains),
            user_demands,
            metrics,
            interference
        ])
        
        return state
        
    def step(self, actions):
        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        
        # Ensure actions is shape (num_base_stations, action_dim)
        actions = np.array(actions).reshape(self.num_base_stations, -1)
        
        # Each BS has actions for:
        # - Transmit power level (0 to 1, scaled to MAX_POWER)
        # - Bandwidth allocation fraction per user (sum to 1)
        
        # Power Allocations
        power_allocations = np.clip(0.5 * (actions[:, 0] + 1), 0, 1) * self.MAX_POWER  # Scale from [-1,1] to [0,MAX_POWER]
        self.current_power = power_allocations
        
        # Extract bandwidth allocation fractions (remaining action dimensions)
        # These need to sum to 1 for each BS
        bandwidth_actions = actions[:, 1:]
        
        bandwidth_allocations = np.zeros((self.num_base_stations, self.num_users_per_bs))
        for bs_idx in range(self.num_base_stations):
            # Use softmax to ensure allocations sum to 1, other methods work just fine, including brute force.
            exp_actions = np.exp(bandwidth_actions[bs_idx])
            bandwidth_allocations[bs_idx] = exp_actions / np.sum(exp_actions)
        
        # SINR for each user
        sinr = np.zeros((self.num_base_stations, self.num_users_per_bs))
        for bs_idx in range(self.num_base_stations):
            for user_idx in range(self.num_users_per_bs):
                # Signal power from serving BS
                signal_power = power_allocations[bs_idx] * self.channel_gains[bs_idx, bs_idx, user_idx]
                
                # Interference power from other BSs
                interference_power = 0
                for interfering_bs in range(self.num_base_stations):
                    if interfering_bs != bs_idx:
                        interference_power += power_allocations[interfering_bs] * self.channel_gains[interfering_bs, bs_idx, user_idx]
                
                #Thermal noise ( dBm to linear)
                noise_power = 10 ** (self.NOISE_FLOOR / 10) / 1000  # Convert to watts
                
                # Calculate SINR
                if signal_power > 0:
                    sinr[bs_idx, user_idx] = signal_power / (interference_power + noise_power)
                else:
                    sinr[bs_idx, user_idx] = 0
                    
                # Store interference for state representation
                self.current_interference[bs_idx, user_idx] = interference_power
        
        #Throughput using Shannon's formula
        throughput = np.zeros((self.num_base_stations, self.num_users_per_bs))
        for bs_idx in range(self.num_base_stations):
            for user_idx in range(self.num_users_per_bs):
                # Shannon capacity: B * log2(1 + SINR)
                if sinr[bs_idx, user_idx] > 0:
                    throughput[bs_idx, user_idx] = (
                        self.BANDWIDTH * 1e6 * bandwidth_allocations[bs_idx, user_idx] * 
                        np.log2(1 + sinr[bs_idx, user_idx])
                    ) / 1e6  #  Mbps
        
        # Calculation of matrics per base station - The per-epsiode averages are plotted in the network_visualization plots
        # Note that the episode is reset, it's not sequential in nature.
        fairness_values = []
        for bs_idx in range(self.num_base_stations):
            # Average SNR in dB
            if np.any(sinr[bs_idx] > 0):
                self.current_snr[bs_idx] = 10 * np.log10(np.mean(sinr[bs_idx]))
            else:
                self.current_snr[bs_idx] = self.MIN_SNR
            
            # Total throughput
            self.current_throughput[bs_idx] = np.sum(throughput[bs_idx])
            
            # Latency (inversely proportional to throughput, higher throughput = lower latency)
            demands_weighted_throughput = np.sum(self.user_demands[bs_idx] * throughput[bs_idx])
            if demands_weighted_throughput > 0:
                self.current_latency[bs_idx] = self.MAX_LATENCY * np.exp(-0.1 * demands_weighted_throughput)
            else:
                self.current_latency[bs_idx] = self.MAX_LATENCY
                
            # Fairness for this BS
            fairness = self._jains_fairness(throughput[bs_idx])
            fairness_values.append(fairness)
        
        # Rewards
        rewards = []
        for bs_idx in range(self.num_base_stations):
            reward = self._calculate_reward(bs_idx, sinr[bs_idx], throughput[bs_idx])
            rewards.append(reward)
        
        # Store per-step metrics for this episode
        self.episode_metrics['snr'].append(np.mean(self.current_snr))
        self.episode_metrics['latency'].append(np.mean(self.current_latency))
        self.episode_metrics['throughput'].append(np.mean(self.current_throughput))
        self.episode_metrics['power'].append(np.mean(self.current_power))
        self.episode_metrics['fairness'].append(np.mean(fairness_values))
        self.episode_metrics['rewards'].append(np.mean(rewards))
        
        # Get next states
        next_states = []
        for i in range(self.num_base_stations):
            next_states.append(self._get_state(i))
        
        return next_states, rewards, self.done, {}

    def _calculate_reward(self, bs_idx, sinr_values, throughput_values):
        """Calculate reward for a single base station"""
        #  Metrics normalized between zero and one here
        snr_norm = (self.current_snr[bs_idx] - self.MIN_SNR) / (40 - self.MIN_SNR)
        snr_norm = np.clip(snr_norm, 0, 1)
        
        latency_norm = (self.MAX_LATENCY - self.current_latency[bs_idx]) / (self.MAX_LATENCY - 5)
        latency_norm = np.clip(latency_norm, 0, 1)
        
        throughput_norm = np.clip(self.current_throughput[bs_idx] / 100, 0, 1)
        
        # Calculate fairness among users
        fairness = self._jains_fairness(throughput_values)
        
        # Compute penalties
        power_penalty = 0
        snr_penalty = 0
        
        # Power consumption penalty if exceeding limit
        if self.current_power[bs_idx] > self.MAX_POWER:
            power_penalty = 10 * (self.current_power[bs_idx] - self.MAX_POWER) / self.MAX_POWER
        
        # SNR penalty if below minimum
        if self.current_snr[bs_idx] < self.MIN_SNR:
            snr_penalty = 5 * (self.MIN_SNR - self.current_snr[bs_idx]) / self.MIN_SNR
        
        # Reward components with weights
        reward = (0.3 * snr_norm + 
                  0.25 * latency_norm + 
                  0.25 * throughput_norm + 
                  0.2 * fairness - 
                  power_penalty - 
                  snr_penalty)
        
        return reward
    
    def _jains_fairness(self, rates):
        """Calculate Jain's fairness index for a set of rates"""
        if np.sum(rates) == 0:
            return 0
        return (np.sum(rates) ** 2) / (self.num_users_per_bs * np.sum(rates ** 2) + 1e-8)
    
    def get_episode_metrics(self):
        """Return average metrics for the entire episode"""
        episode_avg_metrics = {}
        for key in self.episode_metrics:
            if len(self.episode_metrics[key]) > 0:
                episode_avg_metrics[key] = np.mean(self.episode_metrics[key])
            else:
                episode_avg_metrics[key] = 0
        return episode_avg_metrics


# Replay Buffer with Prioritized Experience Replay (think of this like an intermediate storage, like cache :))
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = int(capacity)
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization)
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment  # Beta annealing
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=128):
        if self.size < batch_size:
            indices = np.random.choice(self.size, batch_size, replace=True)
        else:
            # Calculate sampling probabilities
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # Sample indices based on priorities
            indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Increase beta over time for more accurate importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        return (states, actions, rewards, next_states, dones), indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return self.size
