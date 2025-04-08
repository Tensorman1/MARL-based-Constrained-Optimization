Before using the program, please note the following -
"""
This is the implementation of Multi-Agent Deep Deterministic Policy Gradients for Network Resource Optimization under constraints
The core princples have been derived from - Lowe et. al, "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments."
Link to the GitHub repository - https://github.com/openai/maddpg/tree/master".

ChatGPT-4o was used to generate the network_topology plot and the radar chart. However, in some implementations, the graphs were
incorrect and have been manually edited. 

If you have any questions about the specific adaptation to network resource allocation or find any potential error in the 
implementation please reach out. Otherwise this has been created for any and all reuse without restrictions. If you find this
to be useful, please consider citing the repository as a hyperlink for now until the manuscript is in-review.

Things to Note - The run time depends on the compute resources available. This was evaluated using a Ryzen 5 processor with
no GPU. The run time was 2.5 hours for 250 episodes. Results obtained from this for the network resource management problem
might benefit from running more episodes or increasing the neighbourhood size, or running more training smaples per-episode. 
Application-specific heuristic modifications apply for other adaptations.

"""
This is the implementation of Multi-Agent Deep Deterministic Policy Gradients for a small network with 3 agents/Base Stations

Majority of components are the same across both implementations. 
The classes and functions unique to the small network are -
1. Small_Wireless_Environment
2. Training Function

The other parts are to be duplicated from the Large Network.

