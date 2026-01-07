import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import math
import tensorflow as tf
import os
import torch

# --- CONFIGURATION ---
RAIL_LIMIT = 0.092     # The strict hardware limit
TRAINING_LIMIT = 0.5   # The "virtual" limit to prevent instant resets
MASS_CART = 0.35
MASS_POLE = 0.06
LENGTH_POLE = 0.15
MOTOR_FORCE = 7.0      # Increased: Need power to swing up in short distance!

class SwingUpEnv(gym.Env):
    def __init__(self):
        super(SwingUpEnv, self).__init__()
        
        # --- CHANGE 1: BETTER INPUTS ---
        # We now use 5 inputs: [x, x_dot, cos(theta), sin(theta), theta_dot]
        # This is standard for pendulums because it avoids the 3.14 -> -3.14 jump.
        high = np.array([TRAINING_LIMIT * 2, np.inf, 1.0, 1.0, np.inf], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.dt = 0.02 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize physics slightly
        self.masscart = MASS_CART + np.random.uniform(-0.05, 0.05) 
        self.masspole = MASS_POLE + np.random.uniform(-0.01, 0.01)
        self.length = LENGTH_POLE + np.random.uniform(-0.01, 0.02)
        self.force_mag = MOTOR_FORCE + np.random.uniform(-0.5, 1.0) 
        
        # --- CHANGE 2: START EVERYWHERE ---
        # Instead of always starting down, we start at a RANDOM angle.
        # This allows the AI to experience "Winning" (being at the top) early on,
        # which teaches it what the goal is.
        rand_angle = np.random.uniform(-np.pi, np.pi)
        
        self.state = np.array([0.0, 0.0, rand_angle, 0.0], dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        # Return 5 values: [x, v, cos, sin, w]
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action[0]) * self.force_mag
        
        # Physics (Standard)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        total_mass = self.masscart + self.masspole
        pole_mass_len = self.masspole * self.length
        
        temp = (force + pole_mass_len * theta_dot**2 * sintheta) / total_mass
        thetaacc = (9.8 * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / total_mass))
        xacc = temp - pole_mass_len * thetaacc * costheta / total_mass
        
        x += self.dt * x_dot
        x_dot += self.dt * xacc
        theta += self.dt * theta_dot
        theta_dot += self.dt * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        # --- CHANGE 3: BETTER REWARDS ---
        # 1. Being Upright (The Goal)
        r_angle = (np.cos(theta) + 1.0) / 2.0  # Normalized 0.0 (Bottom) to 1.0 (Top)
        
        # 2. Stay in Center (Weak pull)
        r_pos = -0.5 * abs(x)
        
        # 3. Smooth Wall (The "Electric Fence")
        # Instead of a hard -5.0 slap, we scale the pain as it goes further out.
        # If x < 0.092: Penalty is 0.
        # If x > 0.092: Penalty grows exponentially.
        dist_violation = max(0, abs(x) - RAIL_LIMIT)
        r_wall = -200.0 * (dist_violation ** 2)
        
        reward = r_angle + r_pos + r_wall
        
        # Terminate only if we fly off the "Virtual" table (0.5m)
        terminated = bool(abs(x) > TRAINING_LIMIT)
        if terminated: reward -= 10.0

        return self._get_obs(), reward, terminated, False, {}

if __name__ == "__main__":
    # 1. TRAIN
    env = SwingUpEnv()
    print("🚀 Starting Training (God Mode Initialization)...")
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    # Increase steps: Swing up is hard!
    model.learn(total_timesteps=800000) 
    print("✅ Training Complete.")

    # 2. TRANSPLANT
    print("🔄 Converting brain...")
    params = model.policy.state_dict()
    def get_w(key): return params[key].cpu().numpy().T
    def get_b(key): return params[key].cpu().numpy()

    # --- CHANGE 4: UPDATE TF MODEL SHAPE ---
    # Input is now 5 (x, v, cos, sin, w)
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),                 
        tf.keras.layers.Dense(64, activation='tanh'),      
        tf.keras.layers.Dense(64, activation='tanh'),      
        tf.keras.layers.Dense(1, activation=None)          
    ])
    
    # Inject weights
    tf_model.layers[0].set_weights([get_w('mlp_extractor.policy_net.0.weight'), get_b('mlp_extractor.policy_net.0.bias')])
    tf_model.layers[1].set_weights([get_w('mlp_extractor.policy_net.2.weight'), get_b('mlp_extractor.policy_net.2.bias')])
    tf_model.layers[2].set_weights([get_w('action_net.weight'), get_b('action_net.bias')])

    # 3. CONVERT
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open("model.h", "w") as f:
        f.write("#pragma once\n")
        f.write(f"const int model_data_len = {len(tflite_model)};\n")
        f.write("const unsigned char model_data[] = {")
        f.write(", ".join([f"0x{b:02x}" for b in tflite_model]))
        f.write("};\n")

    print(f"🎉 SUCCESS! 'model.h' generated. REMEMBER TO UPDATE YOUR C++ INPUTS TO 5 VALUES!")