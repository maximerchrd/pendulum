import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch.nn as nn
import os

# --- CONFIGURATION ---
HIDDEN_SIZE = 64
MAX_STEPS_PER_EPISODE = 800  
TARGET_REWARD = 1500.0       # Higher target forces "Silence/Efficiency"
TOTAL_STEPS = 3000000        

class ChaosEnv(gym.Env):
    def __init__(self):
        super(ChaosEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.dt = 0.02
        self.max_steps = MAX_STEPS_PER_EPISODE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # --- PHYSICS: CONTROLLED CHAOS ---
        # Pole Length: 45cm - 55cm
        self.len_pole = 0.25 * np.random.uniform(0.9, 1.1)  
        
        # Wide Ranges for Mass & Motor
        self.m_cart = 0.35 * np.random.uniform(0.5, 2.0) 
        self.m_pole = 0.09 * np.random.uniform(0.5, 2.0)    
        self.motor_force = 1.8 * np.random.uniform(0.6, 1.5) 
        self.friction_cart = np.random.uniform(0.05, 0.60) 
        
        self.bias_angle = np.random.uniform(-0.02, 0.02)
        self.deadband_cutoff = 0.15 
        
        self.prev_action = 0.0
        
        # Start near vertical (Slightly easier start to prevent instant crash)
        self.state = np.array([0.0, 0.0, np.random.uniform(-0.005, 0.005), 0.0])
        self.int_pos = 0.0
        self.int_ang = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        # Sensor Noise
        pos_n = np.random.normal(0, 0.003)
        ang_n = np.random.normal(0, 0.003)
        x = self.state[0] + pos_n
        theta = self.state[2] + self.bias_angle + ang_n
        
        self.int_pos = self.int_pos * 0.95 + x
        self.int_ang = self.int_ang * 0.99 + theta
        
        return np.array([
            x * 5.0, self.state[1] * 0.5, theta * 10.0, self.state[3] * 0.5,
            self.int_pos * 1.0, self.int_ang * 2.0
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        raw_action = action[0]
        
        # --- DEADBAND & BACKLASH ---
        if abs(raw_action) < self.deadband_cutoff:
            effective_force = 0.0
        else:
            effective_force = raw_action
            
        # Gear Slack Logic
        if (effective_force * self.prev_action) < -0.01:
            actual_force_input = 0.0 
        else:
            actual_force_input = effective_force

        actual_force = np.clip(self.prev_action, -1.0, 1.0) * self.motor_force
        self.prev_action = effective_force 

        # --- PHYSICS SOLVER ---
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta); sintheta = np.sin(theta)
        total_mass = self.m_cart + self.m_pole
        pole_mass_len = self.m_pole * self.len_pole
        
        temp = (actual_force + pole_mass_len * theta_dot**2 * sintheta - self.friction_cart * x_dot) / total_mass
        thetaacc = (9.8 * sintheta - costheta * temp) / (self.len_pole * (4.0/3.0 - self.m_pole * costheta**2 / total_mass))
        xacc = temp - pole_mass_len * thetaacc * costheta / total_mass
        
        x_dot += self.dt * xacc; x += self.dt * x_dot
        theta_dot += self.dt * thetaacc; theta += self.dt * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        
        # Check termination
        terminated = bool(abs(x) > 0.12 or abs(theta) > 0.30)
        
        # --- FIXED TRUNCATION LOGIC ---
        # Only truncate if we hit the step limit, NOT if we are under it
        truncated = (self.current_step >= self.max_steps)
        
        # --- REWARD SYSTEM ---
        reward = 1.0
        if not terminated:
            reward -= (theta**2 * 40.0) 
            reward -= (theta_dot**2 * 0.2) 
            reward -= (x**2 * 0.1)

            # "Chill Zone" Bonus
            if abs(theta) < 0.008: 
                reward += 0.5  
                if abs(raw_action) < 0.15: 
                    reward += 1.0  # Big bonus for doing nothing
                else:
                    reward -= 0.5 
        else:
            reward = 0.0

        return self._get_obs(), reward, terminated, truncated, {}

if __name__ == "__main__":
    log_dir = "./logs/"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    env = Monitor(ChaosEnv(), log_dir)
    
    # Callback to stop if we hit our target of 1500
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=TARGET_REWARD, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, 
                                 best_model_save_path="./logs/best_model",
                                 log_path="./logs/results", eval_freq=25000)

    # ent_coef=0.01 forces exploration so it doesn't get stuck doing one thing
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, batch_size=64, ent_coef=0.01)
    
    print("🚀 Training with FIXED Logic (Target Reward: 1500)...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
    
    # Export
    del model
    model = PPO.load("./logs/best_model/best_model", env=env)
    
    print("📦 Exporting Weights...")
    net = model.policy.mlp_extractor.policy_net
    w0, b0 = net[0].weight, net[0].bias
    w1, b1 = net[2].weight, net[2].bias
    w_out, b_out = model.policy.action_net.weight, model.policy.action_net.bias

    def to_c(name, t):
        d = t.cpu().detach().numpy().flatten()
        return f"inline const float {name}[{len(d)}] = {{ " + ", ".join([f"{x:.6f}f" for x in d]) + " };\n"

    with open("model_ppo.h", "w") as f:
        f.write(f"#ifndef MODEL_PPO_H\n#define MODEL_PPO_H\n\n#define HIDDEN_SIZE {HIDDEN_SIZE}\n\n")
        f.write(to_c("w1", w0) + to_c("b1", b0))
        f.write(to_c("w2", w1) + to_c("b2", b1))
        f.write(to_c("w_out", w_out) + to_c("b_out", b_out))
        f.write("\n#endif\n")
    print("✅ Done.")