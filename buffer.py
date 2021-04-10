import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model

#Dimension of State Space for single agent
dim_agent_state = 5

#Number of Agents
num_agents = 3

#Dimension of State Space
dim_state = dim_agent_state*num_agents


# Learning rate for actor-critic models
critic_lr = 1e-4
actor_lr = 5e-5

# Creating Optimizer for actor and critic networks
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = 0.95

# Used to update target networks
tau = 0.005

class Buffer:
    def __init__(self, buffer_capacity=10000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, dim_state))
        self.action_buffer = np.zeros((self.buffer_capacity, num_agents))
        self.reward_buffer = np.zeros((self.buffer_capacity, num_agents))
        self.next_state_buffer = np.zeros((self.buffer_capacity, dim_state))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self, ac_models, cr_models, target_ac, target_cr):
    
      # Updating networks of all the agents
      # by looping over number of agent
      for i in range(num_agents):
      
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        
        # Training  and Updating ***critic model*** of ith agent
        target_actions = np.zeros((self.batch_size, num_agents))
        for j in range(num_agents):
          target_actions[:,j] = tf.reshape(
              target_ac[j](next_state_batch[:,5*j:5*(j+1)]), [self.batch_size]
              )

        target_action_batch1 = target_actions[:,0]
        target_action_batch2 = target_actions[:,1]
        target_action_batch3 = target_actions[:,2]
        action_batch1 = action_batch[:,0]
        action_batch2 = action_batch[:,1]
        action_batch3 = action_batch[:,2]
    
        # Finding Gradient of loss function
        with tf.GradientTape() as tape:
            y = reward_batch[:,i] + gamma * target_cr[i]([
                                                          next_state_batch, target_action_batch1, 
                                                          target_action_batch2, target_action_batch3
                                                          ])
            
            critic_value = cr_models[i]([
                                         state_batch, action_batch1, action_batch2, action_batch3
                                         ])
            
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, cr_models[i].trainable_variables)
        
        # Applying gradients to update critic network of ith agent
        critic_optimizer.apply_gradients(
            zip(critic_grad, cr_models[i].trainable_variables)
        )
        # Updating and training of ***critic network*** ended

        # Updating and Training of ***actor network** for ith agent
        actions = np.zeros((self.batch_size, num_agents))
        for j in range(num_agents):
          a = ac_models[j](state_batch[:,5*j:5*(j+1)])
          actions[:,j] = tf.reshape(a, [self.batch_size])

        # Finding gradient of actor model if it is 1st agent
        if i == 0:
          
          with tf.GradientTape(persistent=True) as tape:
              
              action_ = ac_models[i](np.array([state_batch[:,5*i:5*(i+1)][0]]))
              
              critic_value = cr_models[i]([np.array([state_batch[0]]), action_, np.array([actions[:,1][0]]),
                                           np.array([actions[:,2][0]])])

          critic_grad = tape.gradient(critic_value, action_)
          actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

          
          new_actor_grad = [critic_grad[0][0]*element for element in actor_grad]

          for k in range(1,self.batch_size):
              with tf.GradientTape(persistent=True) as tape:
                  
                  action_ = ac_models[i](np.array([state_batch[:,5*i:5*(i+1)][k]]))
                  
                  critic_value = cr_models[i]([np.array([state_batch[k]]), action_, np.array([actions[:,1][k]]),
                                               np.array([actions[:,2][k]])])

              critic_grad = tape.gradient(critic_value, action_)
              actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

              for l in range(len(new_actor_grad)):
                new_actor_grad[l] = new_actor_grad[l] + critic_grad[0][0]*actor_grad[l]


          # Updating gradient network if it is 1st agent
          new_actor_grad = [-1*element/self.batch_size for element in new_actor_grad]
          actor_optimizer.apply_gradients(zip(new_actor_grad, ac_models[i].trainable_variables))
        
        # Finding gradient of actor model if it is 2nd agent
        elif i == 1:
          with tf.GradientTape(persistent=True) as tape:
              
              action_ = ac_models[i](np.array([state_batch[:,5*i:5*(i+1)][0]]))
              
              critic_value = cr_models[i]([np.array([state_batch[0]]), np.array([actions[:,0][0]]),action_,
                                           np.array([actions[:,2][0]])])

          critic_grad = tape.gradient(critic_value, action_)
          actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

          
          new_actor_grad = [critic_grad[0][0]*element for element in actor_grad]

          for k in range(1,self.batch_size):
              with tf.GradientTape(persistent=True) as tape:
                  
                  action_ = ac_models[i](np.array([state_batch[:,5*i:5*(i+1)][k]]))
                  
                  critic_value = cr_models[i]([np.array([state_batch[k]]), np.array([actions[:,0][k]]),action_,
                                               np.array([actions[:,2][k]])])

              critic_grad = tape.gradient(critic_value, action_)
              actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

              for l in range(len(new_actor_grad)):
                new_actor_grad[l] = new_actor_grad[l] + critic_grad[0][0]*actor_grad[l]


          # Updating gradient network if it is 2nd agent
          new_actor_grad = [-1*element/self.batch_size for element in new_actor_grad]
          actor_optimizer.apply_gradients(zip(new_actor_grad, ac_models[i].trainable_variables))

        
        # Finding gradient of actor model if it is 3rd agent
        else:
          with tf.GradientTape(persistent=True) as tape:
              
              action_ = ac_models[i](np.array([state_batch[:,5*i:5*(i+1)][0]]))
              
              critic_value = cr_models[i]([np.array([state_batch[0]]), np.array([actions[:,0][0]]),
                                           np.array([actions[:,1][0]]), action_])

          critic_grad = tape.gradient(critic_value, action_)
          actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

          
          new_actor_grad = [critic_grad[0][0]*element for element in actor_grad]

          for k in range(1,self.batch_size):
              with tf.GradientTape(persistent=True) as tape:
                  
                  action_ = ac_models[i](np.array([state_batch[:,5*i:5*(i+1)][k]]))
                  
                  critic_value = cr_models[i]([np.array([state_batch[k]]), np.array([actions[:,0][k]]),
                                               np.array([actions[:,1][k]]), action_])

              critic_grad = tape.gradient(critic_value, action_)
              actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

              for l in range(len(new_actor_grad)):
                new_actor_grad[l] = new_actor_grad[l] + critic_grad[0][0]*actor_grad[l]


          # Updating gradient network if it is 3rd agent
          new_actor_grad = [-1*element/self.batch_size for element in new_actor_grad]
          actor_optimizer.apply_gradients(zip(new_actor_grad, ac_models[i].trainable_variables))

          
        
        


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau, ac_models, cr_models, target_ac, target_cr):

  for i in range(num_agents):
  
    new_weights = []
    target_variables = target_cr[i].weights
    
    for j, variable in enumerate(cr_models[i].weights):
        new_weights.append(variable * tau + target_variables[j] * (1 - tau))

    target_cr[i].set_weights(new_weights)

    new_weights = []
    target_variables = target_ac[i].weights
    
    for j, variable in enumerate(ac_models[i].weights):
        new_weights.append(variable * tau + target_variables[j] * (1 - tau))

    target_ac[i].set_weights(new_weights)
