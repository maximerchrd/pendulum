import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
import os

# --- 1. CONFIGURATION ---
BASE_MASS_CART = 0.15
BASE_MASS_POLE = 0.06
BASE_LENGTH_POLE = 0.18
BASE_MOTOR_FORCE = 9.0 

DT_PHYSICS = 0.002  
PHYSICS_STEPS_PER_FRAME = 5 

RAIL_LIMIT = 0.092
ANGLE_LIMIT = 0.21

# Neural Network Config
NB_NEURONS = 32
INPUT_SIZE = 6 

# Robustness Settings
TEST_VARIANCE = 0.0     
TEST_BIAS     = 0.015   # Bent Sensor (+0.8 deg)

# Noise
SENSOR_NOISE_POS = 0.008
SENSOR_NOISE_VEL = 0.08
SENSOR_NOISE_ANGLE = 0.0015
SENSOR_NOISE_GYRO = 0.08

# Integrals (MATCH TRAINING)
ANGLE_INTEGRAL_DECAY = 0.995 
DIST_INTEGRAL_DECAY  = 0.90  
MAX_INTEGRAL = 4.0

# --- 2. LOAD WEIGHTS ---
def parse_c_array(content, name):
    pattern = rf"const float {name}\[\d+\] = \{{(.*?)\}};"
    match = re.search(pattern, content, re.DOTALL)
    if not match: raise ValueError(f"Could not find array '{name}'")
    numbers_str = match.group(1).replace('f', '').replace('\n', '')
    return np.array([float(x) for x in numbers_str.split(',') if x.strip()])

if not os.path.exists("model_reinforce.h"):
    print("❌ Error: model_reinforce.h not found!")
    exit()

with open("model_reinforce.h", "r") as f:
    c_content = f.read()

W1 = parse_c_array(c_content, "weights_1").reshape(NB_NEURONS, INPUT_SIZE)
B1 = parse_c_array(c_content, "biases_1").reshape(NB_NEURONS, 1)
W2 = parse_c_array(c_content, "weights_2").reshape(2, NB_NEURONS)
B2 = parse_c_array(c_content, "biases_2").reshape(2, 1)

# --- 3. PHYSICS ENGINE ---
class CartPoleSim:
    def __init__(self):
        self.reset()
        self.poke_text_timer = 0
        
    def reset(self):
        v = TEST_VARIANCE
        self.m_cart = BASE_MASS_CART * np.random.uniform(1.0 - v, 1.0 + v)
        self.m_pole = BASE_MASS_POLE * np.random.uniform(1.0 - v, 1.0 + v)
        self.len_pole = BASE_LENGTH_POLE * np.random.uniform(1.0 - v, 1.0 + v)
        self.force_mag = BASE_MOTOR_FORCE * np.random.uniform(1.0 - v, 1.0 + v)
        self.angle_bias = np.random.uniform(-TEST_BIAS, TEST_BIAS)
        
        start_angle = np.random.uniform(-0.05, 0.05)
        self.state = np.array([0.0, 0.0, start_angle, 0.0]) # x, v, theta, omega
        
        self.steps = 0
        self.done = False
        self.success = False
        
        self.angle_integral = 0.0
        self.dist_integral = 0.0
        return self.state

    def apply_pole_poke(self, intensity):
        """Directly modifies angular velocity (Omega) to simulate a tap."""
        self.state[3] += intensity 
        self.poke_text_timer = 20 

    def step(self, action):
        if self.done: return self.state

        x, x_dot, theta, theta_dot = self.state
        total_mass = self.m_cart + self.m_pole
        pole_mass_len = self.m_pole * self.len_pole
        
        for _ in range(PHYSICS_STEPS_PER_FRAME):
            force = action * self.force_mag
            costheta = np.cos(theta); sintheta = np.sin(theta)

            temp = (force + pole_mass_len * theta_dot**2 * sintheta) / total_mass
            thetaacc = (9.8 * sintheta - costheta * temp) / (self.len_pole * (4.0/3.0 - self.m_pole * costheta**2 / total_mass))
            xacc = temp - pole_mass_len * thetaacc * costheta / total_mass
            
            x_dot += DT_PHYSICS * xacc
            theta_dot += DT_PHYSICS * thetaacc
            x += DT_PHYSICS * x_dot
            theta += DT_PHYSICS * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        if abs(x) > RAIL_LIMIT or abs(theta) > ANGLE_LIMIT: 
            self.done = True; self.success = False
        if self.steps >= 1000: 
            self.done = True; self.success = True
            
        return self.state

# --- 4. BRAIN (UPDATED SCALING) ---
def get_action(sim_obj):
    true_x, true_v, true_theta, true_omega = sim_obj.state
    
    # Sensors + Noise
    meas_pos   = true_x + np.random.normal(0, SENSOR_NOISE_POS)
    meas_vel   = true_v + np.random.normal(0, SENSOR_NOISE_VEL)
    meas_angle = true_theta + np.random.normal(0, SENSOR_NOISE_ANGLE) + sim_obj.angle_bias
    meas_gyro  = true_omega + np.random.normal(0, SENSOR_NOISE_GYRO)

    # Integrals
    sim_obj.angle_integral = np.clip((sim_obj.angle_integral * ANGLE_INTEGRAL_DECAY) + meas_angle, -MAX_INTEGRAL, MAX_INTEGRAL)
    sim_obj.dist_integral = np.clip((sim_obj.dist_integral * DIST_INTEGRAL_DECAY) + meas_pos, -MAX_INTEGRAL, MAX_INTEGRAL)

    # Inputs: MUST MATCH TRAINING EXACTLY
    input_vec = np.array([
        meas_pos * 8.0,          # <--- FIXED (Was 5.0)
        meas_vel,
        meas_angle * 20.0, 
        meas_gyro,
        sim_obj.angle_integral, 
        sim_obj.dist_integral * 4.0  # <--- FIXED (Was 2.0)
    ]).reshape(-1, 1)

    z1 = W1 @ input_vec + B1
    h = np.maximum(0, z1)
    z2 = W2 @ h + B2
    return 1 if z2[1] > z2[0] else -1

# --- 5. VISUALIZATION SETUP ---
sim = CartPoleSim()
pause_counter = 0

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.15, 0.15); ax.set_ylim(-0.05, 0.25)
ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.5)

# Shapes
cart = plt.Rectangle((0, 0), 0.02, 0.015, fc='blue', zorder=10)
pole, = ax.plot([], [], 'r-', lw=3, zorder=5)
ax.add_patch(cart)
ax.axhline(0, color='k', lw=1)
ax.axvline(-RAIL_LIMIT, color='r', ls='--', alpha=0.3)
ax.axvline(RAIL_LIMIT, color='r', ls='--', alpha=0.3)

# Text Labels
title_text = ax.text(0, 0.27, "Init...", ha='center', fontsize=12, weight='bold')
poke_text = ax.text(0, 0.15, "", ha='center', fontsize=14, color='purple', weight='bold')
param_text = ax.text(-0.14, 0.24, "", fontsize=9, verticalalignment='top', family='monospace')
integ_text = ax.text(0.08, 0.24, "", fontsize=9, verticalalignment='top', family='monospace', color='blue')

def update_gui_text():
    bias_deg = sim.angle_bias * 57.296
    info = (
        f"M_Cart: {sim.m_cart:.2f} kg\n"
        f"M_Pole: {sim.m_pole:.2f} kg\n"
        f"Bias  : {bias_deg:+.1f}°\n"
    )
    param_text.set_text(info)
    
    integs = (
        f"INTEGRALS\n"
        f"---------\n"
        f"Ang : {sim.angle_integral:+.2f}\n"
        f"Dist: {sim.dist_integral:+.2f}\n"
    )
    integ_text.set_text(integs)

def on_key(event):
    if event.key == 'right':
        sim.apply_pole_poke(0.5)
        poke_text.set_text("👉 TAPPED POLE (RIGHT)")
    elif event.key == 'left':
        sim.apply_pole_poke(-0.5)
        poke_text.set_text("👈 TAPPED POLE (LEFT)")

fig.canvas.mpl_connect('key_press_event', on_key)

def animate(i):
    global pause_counter
    
    if sim.poke_text_timer > 0:
        sim.poke_text_timer -= 1
    else:
        poke_text.set_text("")
        
    if sim.done:
        pause_counter += 1
        if sim.success:
            title_text.set_text("✅ SUCCESS!")
            cart.set_color('green'); pole.set_color('green')
        else:
            title_text.set_text(f"❌ CRASH Step {sim.steps}")
            cart.set_color('red'); pole.set_color('gray')

        if pause_counter > 50:
            sim.reset(); update_gui_text(); pause_counter = 0
            cart.set_color('blue'); pole.set_color('red')
        return

    action = get_action(sim)
    state = sim.step(action)
    
    x, _, theta, _ = state
    cart.set_xy((x - 0.01, 0))
    pole_x = x + sim.len_pole * np.sin(theta)
    pole_y = 0.015 + sim.len_pole * np.cos(theta)
    pole.set_data([x, pole_x], [0.015, pole_y])
    title_text.set_text(f"Step: {sim.steps}")
    
    if sim.steps % 5 == 0:
        update_gui_text()

update_gui_text()
ani = animation.FuncAnimation(fig, animate, interval=20, blit=False)
plt.show()