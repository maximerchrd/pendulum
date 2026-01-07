import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import math
import tensorflow as tf  # Added for the transplant
import os
import torch

# --- CONFIGURATION ---
railLengthMargin = 0.092  
massCart = 0.35           
massPole = 0.06           
lengthPole = 0.15         
motorForce = 12.0         

class IMUPendulumEnv(gym.Env):
    def __init__(self):
        super(IMUPendulumEnv, self).__init__()
        # Inputs: [Angle, Angle_Velocity]
        self.high = np.array([0.42, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.dt = 0.02 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.masscart = massCart + np.random.uniform(-0.05, 0.05) 
        self.masspole = massPole + np.random.uniform(-0.01, 0.01)
        self.length = lengthPole + np.random.uniform(-0.01, 0.02)
        self.force_mag = motorForce + np.random.uniform(-1.0, 2.0) 
        self.state = np.random.uniform(low=-0.03, high=0.03, size=(4,))
        return np.array([self.state[2], self.state[3]], dtype=np.float32), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action[0]) * self.force_mag
        
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
        
        # Reward logic
        reward = 1.0
        reward -= (theta**2) * 20.0 
        reward -= (theta_dot**2) * 0.1
        terminated = bool(abs(theta) > 0.25 or abs(x) > railLengthMargin)
        if terminated: reward = -10.0

        return np.array([theta, theta_dot], dtype=np.float32), reward, terminated, False, {}

if __name__ == "__main__":
    # 1. TRAIN (PyTorch)
    env = IMUPendulumEnv()
    print("🚀 Starting Training...")
    
    # SB3 Default PPO uses Tanh activation and two layers of 64 neurons
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=400000) # Reduced for testing, increase to 800k for real result
    print("✅ Training Complete.")

    # =========================================================
    # 2. WEIGHT TRANSPLANT (PyTorch -> TensorFlow)
    # =========================================================
    print("🔄 Transplanting brain to TensorFlow...")

    # A. GET WEIGHTS DIRECTLY FROM THE STATE DICTIONARY
    # This acts like a file system for the brain's numbers.
    params = model.policy.state_dict()

    # Helper to extract and format weights
    # PyTorch = (Outputs, Inputs)
    # TensorFlow = (Inputs, Outputs) -> We need to Transpose (.T)
    def get_w(key):
        return params[key].cpu().numpy().T
    
    def get_b(key):
        return params[key].cpu().numpy()

    # EXTRACT WEIGHTS
    # Note: These keys are specific to PPO's MlpPolicy in Stable-Baselines3
    try:
        w1 = get_w('mlp_extractor.policy_net.0.weight')
        b1 = get_b('mlp_extractor.policy_net.0.bias')
        
        w2 = get_w('mlp_extractor.policy_net.2.weight')
        b2 = get_b('mlp_extractor.policy_net.2.bias')
        
        w_out = get_w('action_net.weight')
        b_out = get_b('action_net.bias')
    except KeyError as e:
        print("\n❌ KEY ERROR: The model structure looks different than expected.")
        print("Available keys in model:", params.keys())
        raise e

    # B. BUILD TENSORFLOW TWIN
    # We recreate the exact architecture used by Stable Baselines3 PPO
    # (64 neurons -> Tanh -> 64 neurons -> Tanh -> Output)
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),                 
        tf.keras.layers.Dense(64, activation='tanh'),      
        tf.keras.layers.Dense(64, activation='tanh'),      
        tf.keras.layers.Dense(1, activation=None)          
    ])
    
    # C. INJECT WEIGHTS
    # We forcefully set the trained weights into the new TF model
    tf_model.layers[0].set_weights([w1, b1])
    tf_model.layers[1].set_weights([w2, b2])
    tf_model.layers[2].set_weights([w_out, b_out])
    
    print("✅ Transplant successful.")
    
    # Optional: Quick sanity check 
    # We run the same input through both brains to ensure they think alike
    test_in = np.array([[0.1, 0.5]], dtype=np.float32)
    
    # PyTorch Prediction
    with torch.no_grad():
        pt_obs = torch.as_tensor(test_in)
        pt_out = model.policy.predict(pt_obs, deterministic=True)[0]

    # TensorFlow Prediction
    tf_out = tf_model.predict(test_in, verbose=0)
    
    print(f"   PyTorch says: {pt_out[0][0]:.4f}")
    print(f"   TF says:      {tf_out[0][0]:.4f}")
    
    # =========================================================
    # 3. CONVERT TO ARDUINO HEADER
    # =========================================================
    print("💾 Converting to RP2040 format...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # For RP2040, we often want to ensure functions are supported
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()

    # Write .h file
    with open("model.h", "w") as f:
        f.write("#pragma once\n")
        f.write(f"// Auto-generated from PPO Training\n")
        f.write(f"// Model Size: {len(tflite_model)} bytes\n")
        f.write("const unsigned char model_data[] = {")
        hex_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
        f.write(hex_array)
        f.write("};\n\n")
        f.write(f"const int model_data_len = {len(tflite_model)};")

    print(f"🎉 SUCCESS! 'model.h' generated ({len(tflite_model)} bytes).")