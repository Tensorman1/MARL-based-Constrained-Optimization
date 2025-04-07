# Evaluation Function

def evaluate(maddpg, env, num_episodes=10):
    """
    Evaluate the trained MADDPG agents
    """
    print("Starting evaluation...")
    
    # Metrics tracking
    evaluation_metrics = {
        'rewards': [],
        'snr': [],
        'latency': [],
        'throughput': [],
        'power': [],
        'fairness': []
    }
    
    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards = np.zeros(env.num_base_stations)
        
        while True:
            # Actions without exploration noise
            actions = maddpg.act(states, episode=1000, add_noise=False)
            
            # Execute 
            next_states, rewards, done, _ = env.step(actions)
            
            # Updating episode rewards
            episode_rewards += np.array(rewards)
            states = next_states
            
            if done:
                break
        
        # Get episode metrics
        episode_metrics = env.get_episode_metrics()
        
        # Updadting the evaluation metrics
        evaluation_metrics['rewards'].append(np.mean(episode_rewards))
        evaluation_metrics['snr'].append(episode_metrics['snr'])
        evaluation_metrics['latency'].append(episode_metrics['latency'])
        evaluation_metrics['throughput'].append(episode_metrics['throughput'])
        evaluation_metrics['power'].append(episode_metrics['power'])
        evaluation_metrics['fairness'].append(episode_metrics['fairness'])
        
        print(f"Evaluation Episode {episode+1}/{num_episodes}, Reward: {evaluation_metrics['rewards'][-1]:.2f}")
    
    # Compilation of results
    results = {
        'rewards': np.mean(evaluation_metrics['rewards']),
        'snr': np.mean(evaluation_metrics['snr']),
        'latency': np.mean(evaluation_metrics['latency']),
        'throughput': np.mean(evaluation_metrics['throughput']),
        'power': np.mean(evaluation_metrics['power']),
        'fairness': np.mean(evaluation_metrics['fairness'])
    }
    
    print(f"Evaluation complete!")
    print(f"Average reward: {results['rewards']:.2f}")
    print(f"Average SNR: {results['snr']:.2f} dB")
    print(f"Average latency: {results['latency']:.2f} ms")
    print(f"Average throughput: {results['throughput']:.2f} Mbps")
    print(f"Average power: {results['power']:.2f} W")
    print(f"Average fairness: {results['fairness']:.2f}")
    
    return results, evaluation_metrics
