#The following are the metric values obtained during the evaluation phase
def plot_radar_chart(results, save_path='radar_chart.png'):
    """
    Radar plot to visualize the balance between different metrics
    """
    # Define thresholds for normalization
    thresholds = {
        'snr': (10.0, 40.0),       # (min acceptable, excellent)
        'latency': (100.0, 10.0),   # (max acceptable, excellent) - inverted
        'throughput': (40.0, 100.0),# (min acceptable, excellent)
        'power': (50.0, 20.0),      # (max acceptable, excellent) - inverted
        'fairness': (0.7, 1.0)      # (min acceptable, excellent)
    }
    
    # Normalized metricsbetween 0 and 1 (0 = poor, 1 = excellent, or other wsy around depending on the metric)
    normalized = {}
    
    for key, (min_val, max_val) in thresholds.items():
        if key in results:
            if key in ['latency', 'power']:
                # For these metrics, lower is better
                normalized[key] = (min_val - results[key]) / (min_val - max_val)
            else:
                # For these metrics, higher is better
                normalized[key] = (results[key] - min_val) / (max_val - min_val)
                
            """# Clip to [0, 1]
            normalized[key] = np.clip(normalized[key], 0, 1)"""
    categories = list(normalized.keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  
    
    values = list(normalized.values())
    values += values[:1]  
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Performance Polygon
    ax.plot(angles, values, 'o-', linewidth=2, color='blue', label='Performance')
    ax.fill(angles, values, color='blue', alpha=0.25)
    
    # Add threshold reference (0.5 = minimum acceptable) - Please read the relevant section for more details
    threshold_values = [0.5] * N + [0.5]  # Close the loop
    ax.plot(angles, threshold_values, 'o-', linewidth=2, color='purple', 
          label='Min Threshold')
    

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Value Annotations
    for angle, value, category in zip(angles[:-1], values[:-1], categories):
        if category in ['latency', 'power']:
            # For these metrics, lower is better
            original_value = thresholds[category][0] - value * (thresholds[category][0] - thresholds[category][1])
        else:
            # For these metrics, higher is better
            original_value = thresholds[category][0] + value * (thresholds[category][1] - thresholds[category][0])
            
        # Add annotation slightly outside the data point
        offset = 0.1
        x = (value + offset) * np.cos(angle)
        y = (value + offset) * np.sin(angle)
        ax.text(angle, value + 0.1, f"{original_value:.1f}", 
              fontsize=10, ha='center')
    
    # Customize chart
    ax.set_ylim(0, 1)
    ax.set_title('Performance Metrics Radar Chart', size=15, y=1.1)
    ax.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Radar chart saved to {save_path}")
