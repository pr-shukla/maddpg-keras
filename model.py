import tensorflow as tf
from tensorflow.keras import layers

from env import DIM_AGENT_STATE, NUM_AGENTS



def get_actor():

    # Initialize weights between -3e-5 and 3-e5
    last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

    # Actor will get observation of the agent
    # not the observation of other agents
    inputs = layers.Input(shape=(5,))
    out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(inputs)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    
    # Using tanh activation as action values for
    # for our environment lies between -1 to +1
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    
    outputs = outputs 
    model = tf.keras.Model(inputs, outputs)
    return model





def get_critic(dim_state):

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    
    # State as input, here this state is
    # observation of all the agents
    # hence this state will have information
    # of observation of all the agents
    state_input = layers.Input(shape=(dim_state))
    state_out = layers.Dense(16, activation="selu", kernel_initializer="lecun_normal")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # Action all the agents as input
    action_input1 = layers.Input(shape=(1))
    action_input2 = layers.Input(shape=(1))
    action_input3 = layers.Input(shape=(1))
    action_input = layers.Concatenate()([action_input1, action_input2, action_input3])
    action_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(concat)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    
    outputs = layers.Dense(1)(out)
    
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input1, action_input2, action_input3], outputs)

    return model