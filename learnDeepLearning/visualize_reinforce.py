import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
import os

# --- 1. CONFIGURATION ---
# Physics Base Parameters (Must match training center values)
BASE_MASS_CART = 0.15
BASE_MASS_POLE = 0.05
BASE_LENGTH_POLE = 0.18
BASE_MOTOR_FORCE = 8.0

# Simulation Timing
DT_PHYSICS = 0.002  
# 5 steps * 0.002s = 0.01s per decision (100Hz Control Frequency)
PHYSICS_STEPS_PER_FRAME = 5  

# Limits
RAIL_LIMIT = 0.092
ANGLE_LIMIT = 0.21

# Neural Network Config
NB_NEURONS = 24
VARIANCE = 0.6  # High variance to test robustness

# Sensor Noise Parameters
SENSOR_NOISE_POS = 0.008
SENSOR_NOISE_VEL = 0.08
SENSOR_NOISE_ANGLE = 0.0015
SENSOR_NOISE_GYRO = 0.08

# Robustness Test: How much can the sensor lie?
MAX_TEST_BIAS = 0.06 # +/- 0.06 rads (~3.5 degrees)

# --- 2. PARSE C HEADER ---
def parse_c_array(content, name):
    # Try with newlines
    pattern = rf"const float {name}\[\d+\] = \{{(.*?)\}};"
    match = re.search(pattern, content, re.DOTALL)
    if not match: 
        # Fallback for inline definitions
        pattern = rf"const float {name}\[\d+\] = \{{(.*?)\}};"
        match = re.search(pattern, content, re.DOTALL)
    
    if not match: raise ValueError(f"Could not find array '{name}'")
    
    numbers_str = match.group(1).replace('f', '').replace('\n', '')
    return np.array([float(x) for x in numbers_str.split(',') if x.strip()])

if not os.path.exists("model_reinforce.h"):
    print("❌ Error: model_reinforce.h not found! Run the training script first.")
    exit()

with open("model_reinforce.h", "r") as f:
    c_content = f.read()

try:
    W1 = parse_c_array(c_content, "weights_1").reshape(NB_NEURONS, 4)
    B1 = parse_c_array(c_content, "biases_1").reshape(NB_NEURONS, 1)
    W2 = parse_c_array(c_content, "weights_2").reshape(2, NB_NEURONS)
    B2 = parse_c_array(c_content, "biases_2").reshape(2, 1)
    print("✅ Weights loaded successfully.")
except Exception as e:
    print(f"❌ Error parsing C header: {e}")
    exit()

# --- 3. PHYSICS ENGINE ---
class CartPoleSim:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 1. Randomize Physics (Domain Randomization)
        # We vary mass and force to ensure the AI isn't memorizing one specific robot
        self.m_cart = BASE_MASS_CART * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)
        self.m_pole = BASE_MASS_POLE * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)
        self.len_pole = BASE_LENGTH_POLE * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)
        self.force_mag = BASE_MOTOR_FORCE * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)
        
        # 2. Randomize Sensor Bias (The "Lie")
        # The robot will think 0 degrees is actually (e.g.) +3 degrees.
        self.angle_bias = np.random.uniform(-MAX_TEST_BIAS, MAX_TEST_BIAS)
        
        # 3. Randomize Start State
        start_angle = np.random.uniform(-0.05, 0.05)
        self.state = np.array([0.0, 0.0, start_angle, 0.0])
        
        self.steps = 0
        self.done = False
        self.success = False
        return self.state

    def step(self, action):
        if self.done: return self.state

        x, x_dot, theta, theta_dot = self.state
        
        total_mass = self.m_cart + self.m_pole
        pole_mass_len = self.m_pole * self.len_pole
        
        # Run physics multiple times per "AI Frame"
        for _ in range(PHYSICS_STEPS_PER_FRAME):
            force = action * self.force_mag
            costheta = np.cos(theta)
            sintheta = np.sin(theta)

            temp = (force + pole_mass_len * theta_dot**2 * sintheta) / total_mass
            thetaacc = (9.8 * sintheta - costheta * temp) / (self.len_pole * (4.0/3.0 - self.m_pole * costheta**2 / total_mass))
            xacc = temp - pole_mass_len * thetaacc * costheta / total_mass
            
            # Semi-Implicit Euler
            x_dot += DT_PHYSICS * xacc
            theta_dot += DT_PHYSICS * thetaacc
            x += DT_PHYSICS * x_dot
            theta += DT_PHYSICS * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        # Check Limits
        if abs(x) > RAIL_LIMIT or abs(theta) > ANGLE_LIMIT: 
            self.done = True
            self.success = False

        # Success Condition (5 seconds / 500 steps)
        if self.steps >= 500: # <--- CHANGED TO 500
            self.done = True
            self.success = True
            
        return self.state

def get_action_with_noise(state):
    # 1. Apply the BIAS to the angle (Robot sees the wrong angle)
    biased_angle = state[2] + sim.angle_bias

    # 2. Add Sensor Noise (Jitter)
    noisy_state = np.array([
        state[0] + np.random.normal(0, SENSOR_NOISE_POS),
        state[1] + np.random.normal(0, SENSOR_NOISE_VEL),
        biased_angle + np.random.normal(0, SENSOR_NOISE_ANGLE), 
        state[3] + np.random.normal(0, SENSOR_NOISE_GYRO)
    ])

    # 3. Neural Net Forward Pass
    x_input = noisy_state.reshape(-1, 1)
    z1 = W1 @ x_input + B1
    h = np.maximum(0, z1) # ReLU
    z2 = W2 @ h + B2
    
    # Argmax for decision
    return 1 if z2[1] > z2[0] else -1

# --- 4. ANIMATION SETUP ---
sim = CartPoleSim()
pause_counter = 0

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-0.15, 0.15)
ax.set_ylim(-0.05, 0.25)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# UI Elements
title_text = ax.text(0, 0.27, "Initializing...", ha='center', fontsize=12, weight='bold')
param_text = ax.text(-0.14, 0.20, "", fontsize=9, verticalalignment='top', family='monospace')

# Drawing Objects
cart = plt.Rectangle((0, 0), 0.02, 0.015, fc='blue', zorder=10)
pole, = ax.plot([], [], 'r-', lw=3, zorder=5)
ax.add_patch(cart)
rail = ax.axhline(0, color='k', lw=1)

# Limit Lines
ax.axvline(-RAIL_LIMIT, color='r', ls='--', alpha=0.3)
ax.axvline(RAIL_LIMIT, color='r', ls='--', alpha=0.3)

def update_gui_text():
    # Convert bias to degrees for readability
    bias_deg = sim.angle_bias * 57.296
    
    txt = (f"PHYSICS PARAMS:\n"
           f"M_Cart : {sim.m_cart:.2f} kg\n"
           f"M_Pole : {sim.m_pole:.2f} kg\n"
           f"Length : {sim.len_pole:.2f} m\n"
           f"Motor  : {sim.force_mag:.1f} N\n\n"
           f"SENSOR BIAS (The Lie):\n"
           f"Angle  : {bias_deg:+.1f} deg\n")
    param_text.set_text(txt)

def animate(i):
    global pause_counter
    
    # CASE 1: Simulation is paused (waiting to reset)
    if sim.done:
        pause_counter += 1
        
        if sim.success:
            title_text.set_text("✅ SUCCESS! 5 Seconds Reached")
            cart.set_color('green')
            pole.set_color('green')
        else:
            title_text.set_text(f"❌ CRASHED at step {sim.steps}")
            cart.set_color('red')
            pole.set_color('gray')

        # Reset after 60 frames
        if pause_counter > 60:
            sim.reset()
            update_gui_text()
            pause_counter = 0
            cart.set_color('blue')
            pole.set_color('red')
        
        return

    # CASE 2: Running
    action = get_action_with_noise(sim.state)
    state = sim.step(action)
    
    # Update Visualization (Show REALITY, not NOISE)
    x, _, theta, _ = state
    cart.set_xy((x - 0.01, 0))
    pole_x = x + sim.len_pole * np.sin(theta)
    pole_y = 0.015 + sim.len_pole * np.cos(theta)
    pole.set_data([x, pole_x], [0.015, pole_y])
    
    title_text.set_text(f"Step: {sim.steps} / 500")

# Start
update_gui_text()
# Interval 20ms = 50 FPS
ani = animation.FuncAnimation(fig, animate, interval=20, blit=False)
plt.show()