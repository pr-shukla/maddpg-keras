import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from buffer import Buffer, update_target
from env import NUM_AGENTS, DIM_AGENT_STATE, ENVIRONMENT, reward
from model import get_actor, get_critic
from noise import OUActionNoise
from config import NUM_EPISODES, NUM_BUFFER, NUM_STEPS, STD_DEV, MODEL_PATH, BATCH_SIZE, TAU, CHECKPOINTS

save_path = MODEL_PATH

# Dimension of State Space for single agent
dim_agent_state = DIM_AGENT_STATE

# Number of Agents
num_agents = NUM_AGENTS

# Dimension of State Space
dim_state = dim_agent_state*num_agents

# Number of Episodes
num_episodes = NUM_EPISODES

# Number of Steps in each episodes
num_steps = NUM_STEPS

# For adding noise for exploration
std_dev = STD_DEV
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Neural Net Models for agents will be saved in these lists
ac_models = []
cr_models = []
target_ac = []
target_cr = []

# Appending Neural Network models in lists
for i in range(num_agents):
  ac_models.append(get_actor()) 
  cr_models.append(get_critic(dim_state))

  target_ac.append(get_actor())
  target_cr.append(get_critic(dim_state))

  # Making the weights equal initially
  target_ac[i].set_weights(ac_models[i].get_weights())
  target_cr[i].set_weights(cr_models[i].get_weights())

# Creating class for replay buffer   
buffer = Buffer(NUM_BUFFER, BATCH_SIZE)

# Executing Policy using actor models
def policy(state, noise_object, model):
    
    sampled_actions = tf.squeeze(model(state))
    noise = noise_object()

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, -1.0, 1.0)

    return [np.squeeze(legal_action)]

ep_reward_list = []

# To store average reward history of last few episodes
avg_reward_list = []

print("Training has started")
# Takes about long time to train, about a day on PC with intel core i3 processor
for ep in range(num_episodes):

    # Initializing environment
    env = ENVIRONMENT()
    prev_state = env.initial_obs()
    
    episodic_reward = 0

    for i in range(num_steps):
        
        # Expanding dimension of state from 1-d array to 2-d array
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        
        # Action Value for each agents will be stored in this list
        actions = []
        
        # Get actions for each agents from respective models and store them in list
        for j, model in enumerate(ac_models):
          action = policy(tf_prev_state[:,dim_agent_state*j:dim_agent_state*(j+1)], 
		  					ou_noise, model)
          actions.append(float(action[0]))
          
        # Recieve new state and reward from environment.
        new_state = env.step(actions)
        
        # Rewards recieved is in form of list
        # i.e for 3 agents we will get rewards
        # all 3 agents in this list
        rewards = reward(new_state)

        # Record the experience of all the agents
        # in the replay buffer
        buffer.record((prev_state, actions, rewards, new_state))
        
        # Sum of rewards of all 3 agents
        episodic_reward += sum(rewards)

        # Updating parameters of actor and critic 
        # of all 3 agents using maddpg algorithm
        buffer.learn(ac_models, cr_models, target_ac, target_cr)
        
        # Updating target networks for each agent
        update_target(TAU, ac_models, cr_models, target_ac, target_cr)


	# Saving models after every 10 episodes
    if ep%CHECKPOINTS == 0 and ep!=0:
        
        for k in range(num_agents):
            ac_models[k].save(save_path + 'actor'+str(k)+'.h5') 
            cr_models[k].save(save_path + 'critic'+str(k)+'.h5')

            target_ac[k].save(save_path + 'target_actor' + str(k)+'.h5')
            target_cr[k].save(save_path + 'target_critic' + str(k)+'.h5')
	
    ep_reward_list.append(episodic_reward)
    
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep+1, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting Reward vs Episode plot
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
