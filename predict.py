import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model
from matplotlib import animation
from env_predict import *
from buffer import *
from model import *
from noise import *

dt = 0.4

v = 1.0
ve = 1.2

#Dimension of State Space for single agent
dim_agent_state = 5

num_agents = 3

#Dimension of State Space
dim_state = dim_agent_state*num_agents

#Number of Episodes
num_episodes = 3000

#Number of Steps
num_steps = 400


std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

ac_models = []
cr_models = []
target_ac = []
target_cr = []

path = 'C:/Users/HP/Desktop/desktop_folders/MS_Project_Codes/maddpg/maddpg_models/'

for i in range(num_agents):
  ac_models.append(load_model(path + 'actor'+str(i)+'.h5')) 
  cr_models.append(load_model(path + 'critic'+str(i)+'.h5'))

  target_ac.append(load_model(path + 'target_actor'+str(i)+'.h5'))
  target_cr.append(load_model(path + 'target_critic'+str(i)+'.h5'))



def policy(state, noise_object, model):
    
    sampled_actions = tf.squeeze(model(state))
    noise = noise_object()

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + 0

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, -1.0, 1.0)

    return [np.squeeze(legal_action)]


ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

ag1_reward_list = []
ag2_reward_list = []
ev_reward_list = []

# Takes about 20 min to train
for ep in range(1):

    env = environment()
    prev_state = env.initial_obs()

    
    episodic_reward = 0
    ag1_reward = 0
    ag2_reward = 0
    ev_reward = 0
    
    xp1 = []
    yp1 = []
    xp2 = []
    yp2 = []
    xce = []
    yce = []

    
    #while True:
    for i in range(400):
        
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        actions = []
        for j, model in enumerate(ac_models):
          action = policy(tf_prev_state[:,5*j:5*(j+1)], ou_noise, model)
          actions.append(float(action[0]))
          

        # Recieve state and reward from environment.
        #new_state, sys_state, ev_state = transition(prev_state, sys_state, actions, ev_state)
        new_state = env.step(actions)
        rewards = reward(new_state)

        #buffer.record((prev_state, actions, rewards, new_state))
        
        episodic_reward += sum(rewards)
        ag1_reward += rewards[0]
        ag2_reward += rewards[1]
        ev_reward += rewards[2]

        '''buffer.learn(ac_models, cr_models, target_ac, target_cr)
        update_target(tau, ac_models, cr_models, target_ac, target_cr)'''

        prev_state = new_state
        xp1.append(env.p1_rx)
        yp1.append(env.p1_ry)
        xp2.append(env.p2_rx)
        yp2.append(env.p2_ry)
        
        xce.append(env.e_rx)
        yce.append(env.e_ry)

        d_p1_e = L(env.p1_rx, env.p1_ry, env.e_rx, env.e_ry)
        d_p2_e = L(env.p2_rx, env.p2_ry, env.e_rx, env.e_ry)

        if d_p1_e < 0.4 or d_p2_e < 0.4:
          env = environment()
          prev_state = env.initial_obs()
          print("Captured")
          #break

    


    xc1 = [env.e_rx]
    yc1 = [env.e_ry]

    ep_reward_list.append(episodic_reward)
    ag1_reward_list.append(ag1_reward)
    ag2_reward_list.append(ag2_reward)
    ev_reward_list.append(ev_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Trajectory plot will be generated")
    avg_reward_list.append(avg_reward)
    plt.plot(xp1,yp1)
    plt.plot(xp2,yp2)
    plt.plot(xce,yce)
    plt.plot(xc1,yc1,'.') 
    plt.plot(xp1[-1],yp1[-1],'*')
    plt.plot(xp2[-1],yp2[-1],'*')

plt.show()
    
print("Trajectory Animation will be generated")
# Creating animation of the complete episode during execution
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-1, 11), ylim=(-1, 11))
line, = ax.plot([], [], 'go')
line1, = ax.plot([], [], 'go')
line2, = ax.plot([], [], 'ro')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    return line, line1, line2,

# animation function.  This is called sequentially
def animate(i):
    x = xp1[i-1:i]
    y = yp1[i-1:i]
    x2 = xp2[i-1:i]
    y2 = yp2[i-1:i]
    x_ = xce[i-1:i]
    y_ = yce[i-1:i]
    line.set_data(x, y)
    line1.set_data(x2, y2)
    line2.set_data(x_, y_)
    return line, line1, line2,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=600, interval=1, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('basic_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

# Plotting graph
# Episodes versus Avg. Rewards

plt.show()
