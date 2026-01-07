import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf
import math
import re
import os

# --- CONFIG ---
MASS_CART = 0.15
MASS_POLE = 0.02
LENGTH = 0.2
DT = 0.02
MOTOR_FORCE_MAX = 2.0
VISUAL_X_LIMIT = 0.3

# --- SENSOR NOISE CONFIG (Standard Deviation) ---
# 0.002 meters = 2mm jitter (Laser/Encoder noise)
NOISE_POS = 0.002  
# 0.01 m/s (Derived velocity noise)
NOISE_VEL = 0.01   
# 0.005 rad = ~0.3 degrees (IMU/Accelerometer noise)
NOISE_ANGLE = 0.03 
# 0.02 rad/s (Gyroscope noise)
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
print("🤖 Virtual RP2040 Initialized.")

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

initial_angle = np.random.uniform(-0.15, 0.15) # Reduced initial angle slightly for stability
state = [0.0, 0.0, initial_angle, 0.0] 
history = []

print("🚀 Simulating 500 steps with sensor noise...")
for _ in range(500):
    # --- ADD NOISE HERE (The "Sensor" Step) ---
    # We take the REAL state and add randomness to create the MEASURED state
    measured_x = state[0] + np.random.normal(0, NOISE_POS)
    measured_x_dot = state[1] + np.random.normal(0, NOISE_VEL)
    measured_theta = state[2] + np.random.normal(0, NOISE_ANGLE)
    measured_theta_dot = state[3] + np.random.normal(0, NOISE_GYRO)
    
    # Prepare input using the NOISY values
    input_data = np.array([[measured_x, measured_x_dot, measured_theta, measured_theta_dot]], dtype=np.float32)
    
    # SET INPUT
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # RUN INFERENCE
    interpreter.invoke()
    
    # GET OUTPUT
    output_data = interpreter.get_tensor(output_details[0]['index'])
    action = output_data[0][0]
    
    # APPLY PHYSICS (Using the REAL state, physics doesn't care about sensor noise)
    force = action * MOTOR_FORCE_MAX
    state = run_physics_step(state, force)
    history.append(state)

# ==========================================
# 4. ANIMATION
# ==========================================
print("🎥 Rendering...")
fig, ax = plt.subplots()
ax.set_xlim(-VISUAL_X_LIMIT, VISUAL_X_LIMIT)
ax.set_ylim(-0.2, 0.4)
ax.set_aspect('equal')
ax.set_title(f"Simulation with Sensor Noise (StdDev: {NOISE_ANGLE} rad)")
ax.grid(True)

rail_limit = 0.092
ax.axvline(x=-rail_limit, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=rail_limit, color='red', linestyle='--', alpha=0.5)

cart_rect = plt.Rectangle((0, 0), 0.05, 0.03, fc='blue')
pole_line, = ax.plot([], [], 'r-', linewidth=3)
ax.add_patch(cart_rect)
status_text = ax.text(-0.25, 0.35, "", fontsize=10)

def animate(i):
    x, _, theta, _ = history[i]
    
    cart_rect.set_xy((x - 0.025, 0))
    pole_x = x + LENGTH * math.sin(theta)
    pole_y = 0.03 + LENGTH * math.cos(theta)
    pole_line.set_data([x, pole_x], [0.03, pole_y])
    
    if abs(x) > rail_limit or abs(theta) > 0.25:
        pole_line.set_color('black')
        status_text.set_text("STATUS: CRASHED")
        status_text.set_color("red")
    else:
        pole_line.set_color('red')
        status_text.set_text("STATUS: BALANCING")
        status_text.set_color("green")
        
    return cart_rect, pole_line, status_text

ani = animation.FuncAnimation(fig, animate, frames=len(history), interval=20, blit=True)
plt.show()