import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
import random

#Dimension of State Space for single agent
dim_agent_state = 5


#Time Difference Between 2 Steps
dt = 0.4

#Number of Episodes
num_episodes = 3000

#Number of Steps

num_steps = 400




#velocity of pursuer
v = 1.0


#Velocity of Evader during training
ve = 1.2

#Minimum turing radius of Pursuer
rho = v
rho_e = ve

#angle between initial velocity and reference
te = 3*np.pi/4

num_agents = 3

#Dimension of State Space
dim_state = dim_agent_state*num_agents

import random
class environment:
  def __init__(self):
    self.p1_rx = random.uniform(0.0, 5.0)
    self.p1_ry = random.uniform(0.0, 5.0)
    self.p2_rx = random.uniform(0.0, 5.0)
    self.p2_ry = random.uniform(0.0, 5.0)
    self.p1_vx = v
    self.p1_vy = 0.0
    self.p2_vx = v
    self.p2_vy = 0.0
    self.e_rx = random.uniform(0.0, 5.0)
    self.e_ry = random.uniform(0.0, 5.0)
    self.e_vx = ve*np.cos(te)
    self.e_vy = ve*np.sin(te)
    '''
    self.state_p1_e = [self.p1_rx, self.p1_ry, self.p1_vx, self.p1_ry, self.e_rx,
                  self.e_ry]
    self.state_p2_e = [self.p2_rx, self.p2_ry, self.p2_vx, self.p2_ry, self.e_rx,
                  self.e_ry]
    self.state_p1_p2 = [self.p1_rx, self.p1_ry, self.p1_vx, self.p1_ry, self.p2_rx,
                   self.p1_ry]
    self.state_p2_p1 = [self.p2_rx, self.p2_ry, self.p2_vx, self.p2_ry, self.p1_rx,
                   self.p1_ry]
    '''

  def initial_state(self):
    state_p1_e = [self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.e_rx,
                  self.e_ry]
    state_p2_e = [self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.e_rx,
                  self.e_ry]
    state_p1_p2 = [self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.p2_rx,
                   self.p1_ry]
    state_p2_p1 = [self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.p1_rx,
                   self.p1_ry]
    return state_p1_e, state_p2_e
    

  def initial_obs(self):
    #state_p1_e, state_p2_e = self.initial_state()

    d_p1_e = L(self.p1_rx, self.p1_ry, self.e_rx, self.e_ry)
    d_p2_e = L(self.p2_rx, self.p2_ry, self.e_rx, self.e_ry)
    d_p1_p2 = L(self.p1_rx, self.p1_ry, self.p2_rx, self.p2_ry)
    d_p2_p1 = L(self.p2_rx, self.p2_ry, self.p1_rx, self.p1_ry)
    d_e_p1 = d_p1_e
    d_e_p2 = d_p2_e
    
    phi_p1_e  = phi(self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.e_rx,
                  self.e_ry, v)
    phi_p1_p2 = phi(self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.p2_rx,
                  self.p2_ry, v)
    phi_p2_e  = phi(self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.e_rx,
                  self.e_ry, v)
    phi_p2_p1 = phi(self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.p1_rx,
                  self.p1_ry, v)
    phi_e_p1  = phi(self.e_rx, self.e_ry, self.e_vx, self.e_vy, self.p1_rx,
                  self.p1_ry, ve)
    phi_e_p2  = phi(self.e_rx, self.e_ry, self.e_vx, self.e_vy, self.p2_rx,
                  self.p2_ry, ve)
    
    obs = [d_p1_e/30.0, phi_p1_e/np.pi, 0.0, d_p1_p2/30.0, phi_p1_p2/np.pi,
           d_p2_e/30.0, phi_p2_e/np.pi, 0.0, d_p2_p1/30.0, phi_p2_p1/np.pi,
           d_e_p1/30.0, phi_e_p1/np.pi, d_e_p2/30.0, phi_e_p2/np.pi, 1.0]
           #d_e_p1/30.0, phi_e_p1/np.pi, d_e_p2/30.0,  phi_e_p2/np.pi]

    return obs

  def state_step(self, actions):
    state_p1_e = [self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.e_rx,
                  self.e_ry]
    state_p2_e = [self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.e_rx,
                  self.e_ry]
    state_e_p1 = [self.e_rx, self.e_ry, self.e_vx, self.e_vy, self.p1_rx,
                  self.p1_ry]

    theta_v_p1 = thetap(self.p1_vx, self.p1_vy, v)
    theta_v_p2 = thetap(self.p2_vx, self.p2_vy, v)
    theta_v_e  = thetap(self.e_vx, self.e_vy, ve)


    del_rx_p1 = self.p1_vx * dt
    del_ry_p1 = self.p1_vy * dt
    del_rx_p2 = self.p2_vx * dt
    del_ry_p2 = self.p2_vy * dt
    del_rx_e  = self.e_vx  * dt
    del_ry_e  = self.e_vy  * dt

    del_theta_v_p1 = (v/rho)*actions[0]*dt
    del_theta_v_p2 = (v/rho)*actions[1]*dt
    del_theta_v_e  = (ve/rho_e)*actions[2]*dt

    theta_v_p1 = theta_v_p1 + del_theta_v_p1
    theta_v_p2 = theta_v_p2 + del_theta_v_p2
    theta_v_e  = theta_v_e  + del_theta_v_e

    self.p1_rx = self.p1_rx + del_rx_p1
    self.p1_ry = self.p1_ry + del_ry_p1
    self.p1_vx = v * np.cos(theta_v_p1)
    self.p1_vy = v * np.sin(theta_v_p1)

    if (self.p1_rx > 10.0 and self.p1_vx > 0.0) or (self.p1_rx < 0.0 and self.p1_vx < 0.0):
      self.p1_vx *= -1
    elif (self.p1_ry > 10.0 and self.p1_vy > 0.0) or (self.p1_ry < 0.0 and self.p1_vy < 0.0):
      self.p1_vy *= -1


    self.p2_rx = self.p2_rx + del_rx_p2
    self.p2_ry = self.p2_ry + del_ry_p2
    self.p2_vx = v * np.cos(theta_v_p2)
    self.p2_vy = v * np.sin(theta_v_p2)

    if (self.p2_rx > 10.0 and self.p2_vx > 0.0) or (self.p2_rx < 0.0 and self.p2_vx < 0.0):
      self.p2_vx *= -1
    elif (self.p2_ry > 10.0 and self.p2_vy > 0.0) or (self.p2_ry < 0.0 and self.p2_vy < 0.0):
      self.p2_vy *= -1

    self.e_rx = self.e_rx + del_rx_e
    self.e_ry = self.e_ry + del_ry_e
    self.e_vx = ve * np.cos(theta_v_e)
    self.e_vy = ve * np.sin(theta_v_e)

    if (self.e_rx > 10.0 and self.e_vx > 0.0) or (self.e_rx < 0.0 and self.e_vx < 0.0):
      self.e_vx *= -1
    elif (self.e_ry > 10.0 and self.e_vy > 0.0) or (self.e_ry < 0.0 and self.e_vy < 0.0):
      self.e_vy *= -1

    state_p1_e = [self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.e_rx,
                  self.e_ry]
    state_p2_e = [self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.e_rx,
                  self.e_ry]

    return state_p1_e, state_p2_e

  def step(self, actions):

    old_phi_p1 = phi(self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.e_rx,
                  self.e_ry, v)
    old_phi_p2 = phi(self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.e_rx,
                  self.e_ry, v)
    
    state_p1_e, state_p2_e = self.state_step(actions)

    '''
    obs_p1 = [d_p1_e/30.0, phi_p1_e/np.pi, (new_phi_p1-old_phi_p1)/(dt)]
    obs_p2 = [d_p2_e/30.0, phi_p2_e/np.pi, (new_phi_p2-old_phi_p2)/(dt)]

    obs = [d_p1_e/30.0, phi_p1_e/np.pi, (new_phi_p1-old_phi_p1)/(dt),
           d_p2_e/30.0, phi_p2_e/np.pi, (new_phi_p2-old_phi_p2)/(dt)]
    '''
    d_p1_e = L(self.p1_rx, self.p1_ry, self.e_rx, self.e_ry)
    d_p2_e = L(self.p2_rx, self.p2_ry, self.e_rx, self.e_ry)
    d_p1_p2 = L(self.p1_rx, self.p1_ry, self.p2_rx, self.p2_ry)
    d_p2_p1 = L(self.p2_rx, self.p2_ry, self.p1_rx, self.p1_ry)
    d_e_p1  = d_p1_e
    d_e_p2  = d_p2_e
    
    phi_p1_e  = phi(self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.e_rx,
                  self.e_ry, v)
    phi_p1_p2 = phi(self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.p2_rx,
                  self.p2_ry, v)
    phi_p2_e  = phi(self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.e_rx,
                  self.e_ry, v)
    phi_p2_p1 = phi(self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.p1_rx,
                  self.p1_ry, v)
    phi_e_p1  = phi(self.e_rx, self.e_ry, self.e_vx, self.e_vy, self.p1_rx,
                  self.p1_ry, ve)
    phi_e_p2  = phi(self.e_rx, self.e_ry, self.e_vx, self.e_vy, self.p2_rx,
                  self.p2_ry, ve)
    
    new_phi_p1 = phi(self.p1_rx, self.p1_ry, self.p1_vx, self.p1_vy, self.e_rx,
                  self.e_ry, v)
    new_phi_p2 = phi(self.p2_rx, self.p2_ry, self.p2_vx, self.p2_vy, self.e_rx,
                  self.e_ry, v)
    
    obs = [d_p1_e/30.0, phi_p1_e/np.pi, (new_phi_p1-old_phi_p1)/(dt), d_p1_p2/30.0, phi_p1_p2/np.pi,
           d_p2_e/30.0, phi_p2_e/np.pi, (new_phi_p2-old_phi_p2)/(dt), d_p2_p1/30.0, phi_p2_p1/np.pi,
           d_e_p1/30.0, phi_e_p1/np.pi, d_e_p2/30.0,  phi_e_p2/np.pi, 1.0]

    return obs
  
  
    
#Function for generating sigmoid output of Input Function
def sigmoid(x):
    val = 1/(1+np.exp(-x))
    return val

#Calculating Distance between Pursuer and Evader
def L(rx1, ry1, rx2, ry2):
    d = np.sqrt((rx2-rx1)**2 + (ry2-ry1)**2)
    return d


#Calculating angle between velocity and reference axis
def thetap(vx, vy, v):

    angle = math.acos(vx/v)*((vy+0.001)/abs(vy+0.001))-np.pi*((vy+0.0001)/(abs(vy)+0.0001)-1)
    return angle

def alph(state):
    l = L(state)
    angle = math.acos((state[4]-state[0])/l)*(state[5]-state[1]+0.0001)/abs(state[5]-state[1]+0.0001)-(np.pi)*(((state[5]-state[1]+0.0001)/abs(0.0001+state[5]-state[1]))-1)
    return angle

#Reward Calculator
def reward(state):
  rewards = []
  for i in range(num_agents):
    '''
    ag_state = []
    for j in state[i*4:(i+1)*4]:
      ag_state.append(j)
    for j in ev_state:
      ag_state.append(j)
    '''

    if i == 2:
      r1 = 10*state[0+i*5]#-10*np.exp(-3*state[0+i*5])
      r2 = 10*np.exp(-1*state[1+i*5])
      r3 = 5*np.arctan(1*state[2+i*5])
      r4 = 10*state[2+i*5]#-10*np.exp(-3*state[2+i*5])
      r5 = 10*np.exp(-1*state[0+i*5]*state[3+i*5])
      r =  r1  + r4 #+ r2  #+ r3 
      
      rewards.append(r)

    else:

      r1 = 10*np.exp(-3*state[0+i*5])
      r2 = 10*np.exp(-1*state[1+i*5])
      r3 = 5*np.arctan(1*state[2+i*5])
      r4 = 10*np.exp(-3*state[3+i*5])
      r5 = 10*np.exp(-1*state[0+i*5]*state[3+i*5])
      r =  r1  + r2 + r3 + r4 #+ r2  #+ r3 
      
      rewards.append(r)

  return rewards

#Calculator of Angle between velocity and line joining Evader and Pursuer
def phi(rx1, ry1, vx1, vy1, rx2, ry2, v):
    d = L(rx1, ry1, rx2, ry2)
    rx2_rx1 = rx2 - rx1
    ry2_ry1 = ry2 - ry1
    angle = math.acos(round((rx2_rx1*vx1+ ry2_ry1*vy1)/(d*v), 4))
    return angle
