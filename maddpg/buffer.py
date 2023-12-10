import numpy as np
import tensorflow as tf

from env.env import NUM_AGENTS, DIM_AGENT_STATE
from config import CRITIC_LR, ACTOR_LR, GAMMA, TAU

#Dimension of State Space for single agent
dim_agent_state = DIM_AGENT_STATE

#Number of Agents
num_agents = NUM_AGENTS

#Dimension of State Space
dim_state = dim_agent_state*num_agents

# Learning rate for actor-critic models
critic_lr = CRITIC_LR
actor_lr = ACTOR_LR

# Creating Optimizer for actor and critic networks
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = GAMMA

# Used to update target networks
tau = TAU

class Buffer:

    """
    Maintains buffer of previous experience (s,a,r,s') and 
    updates actor and critic model after calculating gradient

    Parameters
    ----------
    buffer_capacity : int, default=10000
        Max number of previous experience to store
    
    batch_size : int, default=64
        Number of experiences to select randomly for actor critic model updating

    Attributes
    ----------
    buffer_capacity : int
        Same as buffer_capacity defined in parameters
    
    batch_size : int
        Same as batch_size as defined in parameters
    
    buffer_counter : int
        Its tells us num of times record() was called
    
    state_buffer : array of shape (buffer_capacity, dim_state)
        Stores the previous states of agents in buffer

    action_buffer : array of shape (buffer_capacity, num_agents)
        Stores the previous actions of agents in buffer

    reward_buffer : array of shape (buffer_capacity, num_agents)
        Stores the previous rewards of agents in buffer

    next_state_buffer : array of shape (buffer_capacity, dim_state)
        Stores the previous updated states of agents in buffer
    """

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

        """
        Records new experience tuple in buffer

        Parameters
        ----------
        obs_tuple: tuple
            tuple of experience (s,a,r,s')
        """

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
      
      """
      Caclulate gradient for actor and critic and update their parameters

      Parameters
      ----------
      ac_models: list of length num_agents where each element is tf model
        Actor models
      
      cr_models: list of length num_agents where each element is tf model
        Critic models
      
      target_ac: list of lenght num_agents where each element is tf model
        Target actor models

      target_cr: list of legnth num_agents where each element is tf model
        Target critic models
      """

      # Updating networks of all the agents by looping over number of agent

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
        
        ##############################################################################
        #################### CRITIC MODEL UPDATE #####################################
        ##############################################################################

        # Training  and Updating critic model of ith agent

        # Create array of shape (batch_size,num_agents) with all elements zero

        target_actions = np.zeros((self.batch_size, num_agents))
        
        # Calculating target actions i.e. a_i' (i=range(0,num_agents)) to calculate y

        for j in range(num_agents):
          target_actions[:,j] = tf.reshape(
              target_ac[j](next_state_batch[:,dim_agent_state*j:dim_agent_state*(j+1)]), [self.batch_size]
              )
        
        # Creating list of arguments for target critic network (Q'), to calculate y
        # arguments of Q' will be (x', a_1', a_2', ..., a_n') where n=num_agents
        # Here batch of x' = next_state_batch

        state_target_action_batch = [next_state_batch]
        for j in range(num_agents):
            state_target_action_batch.append(target_actions[:,j])
        
        # Creating list of arguments for critic network (Q), to calculate loss 
        # arguments of Q will be (x, a_1, a_2, ..., a_n) where n=num_agents
        # Here batch of x = state_batch

        state_action_batch_critic = [state_batch]
        for j in range(num_agents):
            state_action_batch_critic.append(action_batch[:,j])
    
        # Finding Gradient of loss function

        with tf.GradientTape() as tape:

            # Calculate y = r_i + gamma * Q_i'(x',a_1',a_2', ...,a_n')
            # Here i = current_agent

            y = reward_batch[:,i] + gamma * target_cr[i](state_target_action_batch)

            # Calculate Q_i(x, a_1, a_2, ..., a_n)
            
            critic_value = cr_models[i](state_action_batch_critic)
            
            # Calculate loss = square_mean(y - Q_i(x, a_1, a_2, ..., a_n))
            # Here square_mean is taken over the batch

            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # Calculate gradient of loss w.r.t parameters of critic_model (Q_i)

        critic_grad = tape.gradient(critic_loss, cr_models[i].trainable_variables)
        
        # Applying gradients to update critic network of ith agent

        critic_optimizer.apply_gradients(
            zip(critic_grad, cr_models[i].trainable_variables)
        )
        # Updating and training of critic network ended

        ##############################################################################
        #################### ACTOR MODEL UPDATE ######################################
        ##############################################################################

        # Updating and Training of actor network for ith agent

        # Create array of shape (batch_size,num_agents) with all elements zero

        actions = np.zeros((self.batch_size, num_agents))

        # Calculating actions i.e. a_i (i=range(0,num_agents)) to calculate Q_i(x, a_1, a_2, ..., a_n))
        # to calculate gradient of Q_i(x, a_1, a_2, ...,a_i, ..., a_n)) w.r.t a_i

        # @bug
        # This step seems unecessary as we already have action values in buffer, so 
        # they are not required to be recalculated

        for j in range(num_agents):
          a = ac_models[j](state_batch[:,dim_agent_state*j:dim_agent_state*(j+1)])
          actions[:,j] = tf.reshape(a, [self.batch_size])

        # First gradient calculation is done for first element/experience of batch

        with tf.GradientTape(persistent=True) as tape:

            # Calculating a_0 from actor model(u_i), i.e a_0 = u_i(x_0)
            # Calculating action for first tuple/experience of batch 

            action_ = ac_models[i](np.array([state_batch[:,dim_agent_state*i:dim_agent_state*(i+1)][0]]))

            # Creating list of arguments for critic network (Q), to calculate  
            # gradient arguement of Q will be (x, a_1, a_2, ..., a_n) where n=num_agents
            # Here batch of x = state_batch_0 i.e. first element of batch

            state_actions = [np.array([state_batch[0]])]

            for k in range(num_agents):
                if k==i:
                    state_actions.append(action_)
                else:
                    state_actions.append(np.array([actions[:,k][0]]))

            # Calculate Q(x, a_1, a_2, ..., a_n)

            critic_value = cr_models[i](state_actions)

        # Calculate gradients
        
        # Gradient of critic_value(Q_i) w.r.t action of agent(a_i), i=current agent
        # critic_grad = d(Q_i)/d(a_i), this will be scalar value action is 1 dim

        critic_grad = tape.gradient(critic_value, action_)
        
        # Gradient of actor w.r.t its parameters
        # actor_grad = d(u_i)/d(theta_i), where theta_i are ith actor parameters
        # actor_grad will be array

        actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)
        
        # Calculating critic_grad*actor_grad = [d(Q_i)/d(a_i)]*[d(u_i)/d(theta_i)]
        # critic_grad will be 2d array of shape (1,1), hence this calculation is
        # prodcut of scalar with array

        new_actor_grad = [critic_grad[0][0]*element for element in actor_grad]

        # Here steps for gradient calculation is repeated for remaining elements of
        # experiences in batch, calculation is same just indices will change

        for k in range(1,self.batch_size):
            with tf.GradientTape(persistent=True) as tape:
                  
                action_ = ac_models[i](np.array([state_batch[:,dim_agent_state*i:dim_agent_state*(i+1)][k]]))
                
                state_actions = [np.array([state_batch[k]])]
                
                for l in range(num_agents):
                    if l==i:
                      state_actions.append(action_)
                    else:
                      state_actions.append(np.array([actions[:,l][k]]))

                critic_value = cr_models[i](state_actions)

            critic_grad = tape.gradient(critic_value, action_)
            actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

            for l in range(len(new_actor_grad)):
                new_actor_grad[l] = new_actor_grad[l] + critic_grad[0][0]*actor_grad[l]


        # Updating gradient network 
        new_actor_grad = [-1*element/self.batch_size for element in new_actor_grad]
        actor_optimizer.apply_gradients(zip(new_actor_grad, ac_models[i].trainable_variables))
        

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau, ac_models, cr_models, target_ac, target_cr):

  """
  Update target actor and target critic models

  Parameters
  ----------
  tau: float
    greater value of tau will have more effect on updating target models, and may 
    result in unstability 
  
  ac_models: list of length num_agents where each element is tf model
    Actor models

  cr_models: list of length num_agents where each element is tf model
    Critic models

  target_ac: list of lenght num_agents where each element is tf model
    Target actor models

  target_cr: list of legnth num_agents where each element is tf model
    Target critic models
  """

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
