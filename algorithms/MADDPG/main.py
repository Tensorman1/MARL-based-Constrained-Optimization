def main():
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Model parameters
    num_base_stations = 10
    num_users_per_bs = 5
    num_episodes = 250
    
    env = LargeWirelessNetworkEnv(num_base_stations=num_base_stations, 
                                 num_users_per_bs=num_users_per_bs)
    
    # Training
    maddpg, env, training_metrics = train(
        num_episodes=num_episodes,
        num_base_stations=num_base_stations,
        num_users_per_bs=num_users_per_bs
    )
    
    # Training metrics
    plot_training_metrics(training_metrics)
    
    # Evaluation
    results, evaluation_metrics = evaluate(maddpg, env, num_episodes=10)
    plot_radar_chart(results)
    
    visualize_network(env, maddpg)
    
    # Save model
    for i, actor in enumerate(maddpg.actors):
        torch.save(actor.state_dict(), f'maddpg_actor_{i}.pth')
    for i, critic in enumerate(maddpg.critics):
        torch.save(critic.state_dict(), f'maddpg_critic_{i}.pth')
    
    print("All models have been saved successfully")
    
    return maddpg, env, training_metrics, results



if __name__ == "__main__":
    main()('Episode')

