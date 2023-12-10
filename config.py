# Number of Episodes
NUM_EPISODES = 1000

# Number of Steps in each episodes
NUM_STEPS = 100

# For adding noise for exploration
STD_DEV = 0.2

# Number of experiences stored in buffer
NUM_BUFFER = 10000

# Batch size to select from buffer replay
BATCH_SIZE = 1

# Saved model path
MODEL_PATH = './saved_models/'

# Learning rate for actor-critic models
CRITIC_LR = 1e-4
ACTOR_LR = 5e-5

# Discount factor for future rewards
GAMMA = 0.95

# Used to update target networks
TAU = 0.005