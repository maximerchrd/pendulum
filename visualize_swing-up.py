import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf
import math
import re
import os

# --- CONFIG (MUST MATCH TRAINING!) ---
# Updated to match the "Swing Up" training physics
MASS_CART = 0.35       # Was 0.15
MASS_POLE = 0.04       # Was 0.02
LENGTH = 0.12          # Was 0.2
DT = 0.02
MOTOR_FORCE_MAX = 9.0  # Was 2.0 (Needs power to swing up!)
VISUAL_X_LIMIT = 0.6   # Zoom out to see the swing

# --- SENSOR NOISE CONFIG (Standard Deviation) ---
NOISE_POS = 0.002  
NOISE_VEL = 0.01   
NOISE_ANGLE = 0.03 
NOISE_GYRO = 0.02   

# ==========================================
# 1. MAGIC FUNCTION: LOAD MODEL.H
# ==========================================
def load_brain_from_header(filename="model.h"):
    print(f"📖 Reading '{filename}'...")
    if not os.path.exists(filename):
        raise FileNotFoundError("Could not find model.h! Did you run the training script?")
        
    with open(filename, 'r') as f:
        content = f.read()
        
    hex_values = re.findall(r'0x[0-9a-fA-F]+', content)
    model_content = bytes([int(x, 16) for x in hex_values])
    
    print(f"✅ Reconstructed {len(model_content)} bytes of binary data.")
    return model_content

# ==========================================
# 2. SETUP INTERPRETER
# ==========================================
tflite_model = load_brain_from_header()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if model expects 4 or 5 inputs
expected_inputs = input_details[0]['shape'][1]
print(f"🤖 Virtual RP2040 Initialized. Brain expects {expected_inputs} inputs.")

# ==========================================
# 3. PHYSICS SIMULATION
# ==========================================
def run_physics_step(state, force):
    x, x_dot, theta, theta_dot = state
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    total_mass = MASS_CART + MASS_POLE
    pole_mass_len = MASS_POLE * LENGTH
    
    temp = (force + pole_mass_len * theta_dot**2 * sintheta) / total_mass
    thetaacc = (9.8 * sintheta - costheta * temp) / (LENGTH * (4.0/3.0 - MASS_POLE * costheta**2 / total_mass))
    xacc = temp - pole_mass_len * thetaacc * costheta / total_mass
    
    x += DT * x_dot
    x_dot += DT * xacc
    theta += DT * theta_dot
    theta_dot += DT * thetaacc
    return [x, x_dot, theta, theta_dot]

# START AT THE BOTTOM (Swing Up Test)
# pi = 3.14159... (Hanging down)
initial_angle = np.pi + np.random.uniform(-0.1, 0.1) 
state = [0.0, 0.0, initial_angle, 0.0] 
history = []

print("🚀 Simulating 500 steps...")
for _ in range(500):
    # --- 1. READ SENSORS (Add Noise) ---
    measured_x = state[0] + np.random.normal(0, NOISE_POS)
    measured_x_dot = state[1] + np.random.normal(0, NOISE_VEL)
    measured_theta = state[2] + np.random.normal(0, NOISE_ANGLE)
    measured_theta_dot = state[3] + np.random.normal(0, NOISE_GYRO)
    
    # --- 2. PREPARE INPUT (5 VALUES) ---
    # We now feed [x, v, cos, sin, w]
    # Note: We calculate cos/sin from the NOISY angle, just like the real robot will.
    cos_th = np.cos(measured_theta)
    sin_th = np.sin(measured_theta)
    
    input_data = np.array([[measured_x, measured_x_dot, cos_th, sin_th, measured_theta_dot]], dtype=np.float32)
    
    # --- 3. INFERENCE ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # --- 4. ACTION ---
    action = output_data[0][0]
    force = action * MOTOR_FORCE_MAX
    
    # --- 5. PHYSICS UPDATE ---
    state = run_physics_step(state, force)
    history.append(state)

# ==========================================
# 4. ANIMATION
# ==========================================
print("🎥 Rendering...")
fig, ax = plt.subplots()
ax.set_xlim(-VISUAL_X_LIMIT, VISUAL_X_LIMIT)
ax.set_ylim(-0.25, 0.25)
ax.set_aspect('equal')
ax.set_title(f"Swing-Up Test (Motor: {MOTOR_FORCE_MAX}N)")
ax.grid(True)

# Draw the rail limits
rail_limit = 0.092
ax.axvline(x=-rail_limit, color='k', linestyle='--', alpha=0.3, label='Hard Stop')
ax.axvline(x=rail_limit, color='k', linestyle='--', alpha=0.3)

cart_rect = plt.Rectangle((0, 0), 0.05, 0.03, fc='blue', zorder=10)
pole_line, = ax.plot([], [], 'r-', linewidth=3, zorder=5)
ax.add_patch(cart_rect)
status_text = ax.text(-0.5, 0.2, "", fontsize=10)

def animate(i):
    x, _, theta, _ = history[i]
    
    # Update Cart
    cart_rect.set_xy((x - 0.025, 0))
    
    # Update Pole
    pole_x = x + LENGTH * math.sin(theta)
    pole_y = 0.03 + LENGTH * math.cos(theta)
    pole_line.set_data([x, pole_x], [0.03, pole_y])

    # Status Logic
    is_upright = math.cos(theta) > 0.9  # Roughly upright
    is_crashed = abs(x) > rail_limit
    
    if is_crashed:
        status_text.set_text("STATUS: HIT WALL!")
        status_text.set_color("red")
        pole_line.set_color('gray')
    elif is_upright:
        status_text.set_text("STATUS: BALANCING")
        status_text.set_color("green")
        pole_line.set_color('green')
    else:
        status_text.set_text("STATUS: SWINGING UP")
        status_text.set_color("orange")
        pole_line.set_color('red')
        
    return cart_rect, pole_line, status_text

ani = animation.FuncAnimation(fig, animate, frames=len(history), interval=20, blit=True)
plt.show()