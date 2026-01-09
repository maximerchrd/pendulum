import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
import os

# --- 1. CONFIGURATION ---
# Base values (multipliers will apply in reset)
BASE_MASS_CART = 0.35
BASE_MASS_POLE = 0.09   
BASE_LENGTH_POLE = 0.25 # Half-length (CoM)
BASE_MOTOR_FORCE = 1.8  

DT_PHYSICS = 0.02       
RAIL_LIMIT = 0.13       
ANGLE_LIMIT = 0.35      

MAX_STEPS = 1000        

HIDDEN_SIZE = 64
INPUT_SIZE = 6 

# --- 2. LOAD WEIGHTS ---
def parse_c_array(content, name):
    pattern = rf"const float {name}\[\d+\] = \{{(.*?)\}};"
    match = re.search(pattern, content, re.DOTALL)
    if not match: 
        return np.zeros(1)
    numbers_str = match.group(1).replace('f', '').replace('\n', '')
    vals = [float(x) for x in numbers_str.split(',') if x.strip()]
    return np.array(vals)

filename = "model_ppo.h"
if not os.path.exists(filename):
    print(f"❌ Error: {filename} not found! Run training first.")
    exit()

with open(filename, "r") as f:
    c_content = f.read()

W1 = parse_c_array(c_content, "w1").reshape(HIDDEN_SIZE, INPUT_SIZE)
B1 = parse_c_array(c_content, "b1").reshape(HIDDEN_SIZE, 1)
W2 = parse_c_array(c_content, "w2").reshape(HIDDEN_SIZE, HIDDEN_SIZE)
B2 = parse_c_array(c_content, "b2").reshape(HIDDEN_SIZE, 1)
W_OUT = parse_c_array(c_content, "w_out").reshape(1, HIDDEN_SIZE)
B_OUT = parse_c_array(c_content, "b_out").reshape(1, 1)

print("✅ Weights Loaded.")

# --- 3. PHYSICS ENGINE ---
class StickySim:
    def __init__(self):
        self.reset()
        self.poke_timer = 0
        self.last_poke = 0
        
    def reset(self):
        # --- SYNCHRONIZED WITH CHAOS TRAINING ---
        # Cart Mass: 0.17kg to 0.7kg
        self.m_cart = 0.35 * np.random.uniform(0.5, 2.0)
        
        # Pole Mass: 45g to 180g
        self.m_pole = 0.09 * np.random.uniform(0.5, 2.0)
        
        # Pole Length (CoM): 22.5cm to 27.5cm
        self.len_pole = 0.25 * np.random.uniform(0.9, 1.1)
        
        # Motor Force: 1.0N to 2.7N
        self.force_mag = 1.8 * np.random.uniform(0.6, 1.5)
        
        # Friction: Slippery to Sticky
        self.friction = np.random.uniform(0.05, 0.60)
        
        self.deadband_cutoff = 0.15 
        self.prev_effective_action = 0.0
        
        # Start near vertical
        self.state = np.array([0.0, 0.0, np.random.uniform(-0.02, 0.02), 0.0]) 
        self.int_pos = 0.0
        self.int_ang = 0.0
        
        self.steps = 0
        self.done = False
        self.success = False
        return self.state

    def poke(self, amount):
        self.state[3] += amount 
        self.poke_timer = 20
        self.last_poke = amount

    def step(self, raw_action):
        if self.done: return self.state

        # Deadband Logic
        if abs(raw_action) < self.deadband_cutoff:
            effective_action = 0.0
        else:
            effective_action = raw_action

        # Gear Slack / Latency Simulation
        force = self.prev_effective_action * self.force_mag
        self.prev_effective_action = effective_action

        # Physics Math
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta); sintheta = np.sin(theta)
        total_mass = self.m_cart + self.m_pole
        pole_mass_len = self.m_pole * self.len_pole

        temp = (force + pole_mass_len * theta_dot**2 * sintheta - self.friction * x_dot) / total_mass
        thetaacc = (9.8 * sintheta - costheta * temp) / (self.len_pole * (4.0/3.0 - self.m_pole * costheta**2 / total_mass))
        xacc = temp - pole_mass_len * thetaacc * costheta / total_mass
        
        x_dot += DT_PHYSICS * xacc
        x += DT_PHYSICS * x_dot
        theta_dot += DT_PHYSICS * thetaacc
        theta += DT_PHYSICS * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        if abs(x) > RAIL_LIMIT or abs(theta) > ANGLE_LIMIT: 
            self.done = True
        
        if self.steps >= MAX_STEPS: 
            self.done = True; self.success = True
            
        return self.state

# --- 4. BRAIN (INFERENCE) ---
def get_action(sim):
    x, v, theta, omega = sim.state
    
    # Add simulated sensor noise
    obs_x = x + np.random.normal(0, 0.002)
    obs_theta = theta + np.random.normal(0, 0.002)
    
    # Update Integral Terms (Memory)
    sim.int_pos = sim.int_pos * 0.95 + obs_x
    sim.int_ang = sim.int_ang * 0.99 + obs_theta
    
    # Scaling Inputs (MUST MATCH TRAINING)
    input_vec = np.array([
        obs_x * 5.0, 
        v * 0.5, 
        obs_theta * 10.0, 
        omega * 0.5,
        sim.int_pos * 1.0, 
        sim.int_ang * 2.0
    ]).reshape(-1, 1)

    # Neural Net Pass
    z1 = W1 @ input_vec + B1
    h1 = np.tanh(z1) 
    z2 = W2 @ h1 + B2
    h2 = np.tanh(z2)
    out = W_OUT @ h2 + B_OUT
    
    return np.clip(out[0][0], -1.0, 1.0)

# --- 5. VISUALIZATION ---
sim = StickySim()
pause_counter = 0

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.16, 0.16); ax.set_ylim(-0.05, 0.35) 
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title(f"Robust AI Test (Chaos Mode)", fontsize=14)

# Visual Elements
ground = plt.Rectangle((-RAIL_LIMIT, -0.005), RAIL_LIMIT*2, 0.005, fc='black', alpha=0.1)
cart = plt.Rectangle((0, 0), 0.03, 0.02, fc='#007acc', ec='black', lw=1, zorder=10)
pole, = ax.plot([], [], color='#ff4d4d', lw=5, solid_capstyle='round', zorder=5)
force_bar = plt.Rectangle((0, -0.03), 0, 0.01, fc='orange')
deadzone_area = plt.Rectangle((-0.02, -0.04), 0.04, 0.03, fc='gray', alpha=0.2, hatch='///') 

smack_arrow = ax.annotate("", xy=(0,0), xytext=(0,0), 
                          arrowprops=dict(facecolor='purple', shrink=0.05, lw=0),
                          zorder=20, visible=False)

ax.add_patch(ground); ax.add_patch(cart); ax.add_patch(force_bar); ax.add_patch(deadzone_area)
ax.axvline(-RAIL_LIMIT, color='r', ls=':', alpha=0.5)
ax.axvline(RAIL_LIMIT, color='r', ls=':', alpha=0.5)

status_text = ax.text(0, 0.30, "READY", ha='center', fontsize=12, weight='bold')

# Parameter Display Box
param_text = ax.text(-0.15, 0.34, "", fontsize=9, family='monospace', va='top', 
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

def update_param_display():
    # Convert CoM length back to total length for display (approx)
    total_len_est = sim.len_pole * 2.0 
    
    text = (
        f"RUN PARAMETERS:\n"
        f"----------------\n"
        f"Pole Mass   : {sim.m_pole*1000:.0f}g\n"
        f"Pole Len    : {total_len_est*100:.1f}cm\n"
        f"Cart Mass   : {sim.m_cart*1000:.0f}g\n"
        f"Friction    : {sim.friction:.2f}\n"
        f"Motor Force : {sim.force_mag:.2f}N\n"
    )
    param_text.set_text(text)

# Initial update
update_param_display()

def on_key(event):
    if event.key == 'right':
        sim.poke(0.5) 
    elif event.key == 'left':
        sim.poke(-0.5) 

fig.canvas.mpl_connect('key_press_event', on_key)

def animate(i):
    global pause_counter
    
    x, _, theta, _ = sim.state
    draw_len = sim.len_pole * 2.0 
    
    pole_x = x + draw_len * np.sin(theta)
    pole_y = 0.02 + draw_len * np.cos(theta)
    
    # Handle Pokes
    if sim.poke_timer > 0: 
        sim.poke_timer -= 1
        smack_arrow.set_visible(True)
        if sim.last_poke > 0: 
            smack_arrow.xy = (pole_x, pole_y)
            smack_arrow.set_position((pole_x - 0.05, pole_y)) 
        else: 
            smack_arrow.xy = (pole_x, pole_y)
            smack_arrow.set_position((pole_x + 0.05, pole_y)) 
    else:
        smack_arrow.set_visible(False)
        
    # Handle Termination
    if sim.done:
        pause_counter += 1
        if sim.success:
            status_text.set_text("🏆 STABLE!")
            cart.set_facecolor('green')
        else:
            status_text.set_text(f"💀 FALL! Step {sim.steps}")
            cart.set_facecolor('red')

        if pause_counter > 50: 
            sim.reset()
            update_param_display() 
            pause_counter = 0
            cart.set_facecolor('#007acc')
        return

    # Run Physics
    raw_action = get_action(sim)
    sim.step(raw_action)
    
    x, _, theta, _ = sim.state
    
    # --- CHILL ZONE GLOW ---
    # If angle is tiny (<0.5 deg), outline green to show efficiency
    if abs(theta) < 0.008:
        cart.set_edgecolor('#00ff00')
        cart.set_linewidth(3)
    else:
        cart.set_edgecolor('black')
        cart.set_linewidth(1)
    
    # Update Graphics
    cart.set_x(x - 0.015)
    pole.set_data([x, pole_x], [0.02, pole_y])
    
    if abs(raw_action) < sim.deadband_cutoff:
        force_bar.set_color('gray') 
        force_bar.set_alpha(0.5)
    else:
        force_bar.set_alpha(1.0)
        if raw_action > 0: force_bar.set_color('orange')
        else: force_bar.set_color('purple')

    force_bar.set_width(raw_action * 0.05) 
    force_bar.set_x(x)

    status_text.set_text(f"Angle: {theta*57.3:.1f}° | Force: {raw_action:.2f}")

ani = animation.FuncAnimation(fig, animate, interval=20, blit=False)
plt.show()