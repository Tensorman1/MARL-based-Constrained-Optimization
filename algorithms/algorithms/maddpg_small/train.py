"""This defines the training of the small network"""
def train(num_episodes=250, num_base_stations=3, num_users_per_bs=5):
    env = SmallWirelessNetworkEnv(num_base_stations=num_base_stations, 
                                 num_users_per_bs=num_users_per_bs)
    init_states = env.reset()
    local_state_dim = len(init_states[0])
    
    # Actions: power level + bandwidth allocation
    action_dim = 2  # (1 for power, 1 for bandwidth allocation)
    
    print(f"State dimension: {local_state_dim}, Action dimension: {action_dim}")
    maddpg = MADDPG(num_base_stations, local_state_dim, action_dim, neighborhood_size=1)
    replay_buffer = PrioritizedReplayBuffer(capacity=500000, alpha=0.6)
    
    # Training parameters
    batch_size = 256
    
    # Metrics tracking
    training_metrics = {
        'rewards': [],
        'snr': [],
        'latency': [],
        'throughput': [],
        'power': [],
        'fairness': [],
        'actor_loss': [],
        'critic_loss': [],
        'episode_duration': []
    }
    
    print("Starting training...")
    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards = np.zeros(num_base_stations)
        step_count = 0
        episode_start = time.time()
        
        # Per-episode loss tracking
        episode_actor_loss = []
        episode_critic_loss = []
        
        while True:
            
            actions = maddpg.act(states, episode)
            next_states, rewards, done, _ = env.step(actions)
            replay_buffer.add(np.array(states), np.array(actions), np.array(rewards), 
                             np.array(next_states), np.array([done] * num_base_stations))
            
            episode_rewards += np.array(rewards)
            states = next_states
            step_count += 1
            

            if len(replay_buffer) > batch_size:
                losses = []
                for agent_idx in range(num_base_stations):
                    experiences, indices = replay_buffer.sample(batch_size)
                    loss = maddpg.update(experiences, agent_idx, episode)
                    losses.append(loss)
                    
                    # Updating the priorities (using TD error as proxy)
                    priorities = np.ones_like(indices) * (1.0 + np.random.rand(len(indices)) * 0.1)
                    replay_buffer.update_priorities(indices, priorities)
                

                mean_actor_loss = np.mean([loss['actor_loss'] for loss in losses])
                mean_critic_loss = np.mean([loss['critic_loss'] for loss in losses])
                episode_actor_loss.append(mean_actor_loss)
                episode_critic_loss.append(mean_critic_loss)
            
            if done:
                break
        
        episode_end = time.time()
        episode_duration = episode_end - episode_start
        episode_metrics = env.get_episode_metrics()

        training_metrics['rewards'].append(np.mean(episode_rewards))
        training_metrics['snr'].append(episode_metrics['snr'])
        training_metrics['latency'].append(episode_metrics['latency'])
        training_metrics['throughput'].append(episode_metrics['throughput'])
        training_metrics['power'].append(episode_metrics['power'])
        training_metrics['fairness'].append(episode_metrics['fairness'])
        training_metrics['actor_loss'].append(np.mean(episode_actor_loss) if episode_actor_loss else 0)
        training_metrics['critic_loss'].append(np.mean(episode_critic_loss) if episode_critic_loss else 0)
        training_metrics['episode_duration'].append(episode_duration)
        
        # Progress, this may not be very comprehensive, might need further modification - consult GitHub Repository, Speak to Ying
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Reward: {training_metrics['rewards'][-1]:.2f}")
            print(f"  Avg SNR: {training_metrics['snr'][-1]:.2f} dB")
            print(f"  Avg Latency: {training_metrics['latency'][-1]:.2f} ms")
            print(f"  Avg Throughput: {training_metrics['throughput'][-1]:.2f} Mbps")
            print(f"  Avg Power: {training_metrics['power'][-1]:.2f} W")
            print(f"  Avg Fairness: {training_metrics['fairness'][-1]:.2f}")
            print(f"  Duration: {episode_duration:.2f}s, Buffer: {len(replay_buffer)}")
            
            maddpg.noise.decay()
    
    print("Training complete!")
    return maddpg, env, training_metrics
