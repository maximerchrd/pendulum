# IMPROVED REINFORCE: 
# 'State Memory' (integral inputs) to handle sensor bias/drift
# Mini-Batch updates (avg over 10 games) for stable convergence.
# Curriculum Learning (progressive difficulty) 
# "Gravity Well" reward function that guarantees positive reinforcement for survival, preventing early training collapse.

import numpy as np

# --- 1. CONFIGURATION ---
BATCH_SIZE = 10         
INPUT_SIZE = 6          

# Physics
railLengthMargin = 0.092  
massCart = 0.15           
massPole = 0.06           
lengthPole = 0.18         
motorForce = 9.0
dt_phys_sim = 0.002

# Limits
limit_angle = 0.21
limit_distance = 0.091

# Training
nb_macro_steps = 500
nb_games = 500000
success_threshold = 450

# Sensor Noise 
sensor_noise_pos = 0.008
sensor_noise_vel = 0.08
sensor_noise_angle = 0.0015
sensor_noise_gyro = 0.08

# --- INTEGRAL CONFIG ---
ANGLE_INTEGRAL_DECAY = 0.995 
DIST_INTEGRAL_DECAY  = 0.90   
MAX_INTEGRAL = 4.0

# --- CURRICULUM ---
current_variance = 0.05   
MAX_VARIANCE = 0.5        

# --- NEURAL NETWORK ---
nb_neurons = 32
learning_rate = 0.0025
gamma = 0.99

weights_1 = np.random.randn(nb_neurons, INPUT_SIZE) * np.sqrt(2/INPUT_SIZE)
biases_1 = np.zeros((nb_neurons, 1))
weights_2 = np.random.randn(2, nb_neurons) * np.sqrt(2/nb_neurons)
biases_2 = np.zeros((2, 1))

batch_dw1 = np.zeros_like(weights_1); batch_db1 = np.zeros_like(biases_1)
batch_dw2 = np.zeros_like(weights_2); batch_db2 = np.zeros_like(biases_2)

# --- HELPER FUNCTIONS ---
def calculate_returns(rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    if returns.std() > 1e-5:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        returns = returns - returns.mean()
    return returns

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / np.sum(e_x, axis=0)

def physics_step(state, action, m_cart, m_pole, l_pole, f_motor):
    x, x_dot, theta, theta_dot = state
    for _ in range(5):
        force = action * f_motor
        costheta = np.cos(theta); sintheta = np.sin(theta)
        total_mass = m_cart + m_pole; pole_mass_len = m_pole * l_pole

        temp = (force + pole_mass_len * theta_dot**2 * sintheta) / total_mass
        thetaacc = (9.8 * sintheta - costheta * temp) / (l_pole * (4.0/3.0 - m_pole * costheta**2 / total_mass))
        xacc = temp - pole_mass_len * thetaacc * costheta / total_mass

        x_dot += dt_phys_sim * xacc
        theta_dot += dt_phys_sim * thetaacc
        x += dt_phys_sim * x_dot
        theta += dt_phys_sim * theta_dot
    return [x, x_dot, theta, theta_dot]

# --- MAIN LOOP ---
avg_steps_1000 = 0
print(f"Training: Squared Penalty Mode (Gravity Well)")

for _game in range(nb_games):
    
    # Curriculum
    if avg_steps_1000 > 150 and _game % 1000 == 0:
        current_variance = min(current_variance + 0.05, MAX_VARIANCE)
        print(f"🔥 LEVEL UP! Variance: {current_variance*100:.1f}%")

    m_cart_r = massCart * np.random.uniform(1.0 - current_variance, 1.0 + current_variance)
    m_pole_r = massPole * np.random.uniform(1.0 - current_variance, 1.0 + current_variance)
    l_pole_r = lengthPole * np.random.uniform(1.0 - current_variance, 1.0 + current_variance)
    f_motor_r = motorForce * np.random.uniform(1.0 - current_variance, 1.0 + current_variance)
    
    bias_scale = 0.5 + 0.5 * (current_variance / MAX_VARIANCE) 
    angle_bias = np.random.uniform(-0.015, 0.015) * bias_scale

    current_state = [0.0, 0.0, 0.0, 0.0]
    angle_integral = 0.0
    dist_integral = 0.0
    
    state_history, reward_history, action_history = [], [], []
    prob_history, h_history, z1_history = [], [], []

    for _step in range(nb_macro_steps):
        # 1. Sensors
        meas_pos   = current_state[0] + np.random.normal(0, sensor_noise_pos)
        meas_vel   = current_state[1] + np.random.normal(0, sensor_noise_vel)
        meas_angle = current_state[2] + np.random.normal(0, sensor_noise_angle) + angle_bias
        meas_gyro  = current_state[3] + np.random.normal(0, sensor_noise_gyro)
        
        # 2. Integrals
        angle_integral = np.clip((angle_integral * ANGLE_INTEGRAL_DECAY) + meas_angle, -MAX_INTEGRAL, MAX_INTEGRAL)
        dist_integral  = np.clip((dist_integral * DIST_INTEGRAL_DECAY) + meas_pos, -MAX_INTEGRAL, MAX_INTEGRAL)

        # 3. Inputs (Boosted Position Scaling)
        input_vec = np.array([
            meas_pos * 8.0,      # <--- BOOSTED: Was 5.0. Make position LOUD.
            meas_vel,
            meas_angle * 20.0, 
            meas_gyro,
            angle_integral,
            dist_integral * 4.0  # <--- BOOSTED: Was 2.0. Force it to notice the drift.
        ]).reshape(-1, 1)

        # 4. Forward
        z1 = weights_1 @ input_vec + biases_1
        z1_history.append(z1)
        h = np.maximum(0, z1)
        h_history.append(h)
        z2 = weights_2 @ h + biases_2
        p = softmax(z2)
        prob_history.append(p)
        
        action = 1 if np.random.rand() < p[1][0] else -1
        action_history.append(action)
        state_history.append(input_vec)

        # 5. Physics
        current_state = physics_step(current_state, action, m_cart_r, m_pole_r, l_pole_r, f_motor_r)

        # 6. Check Fail
        if np.abs(current_state[0]) > limit_distance or np.abs(current_state[2]) > limit_angle:
             reward_history.append(0.0) 
             break
        else:
            # --- THE GRAVITY WELL REWARD ---
            # 1. Normalize values (0 to 1)
            norm_angle = np.abs(current_state[2]) / limit_angle
            norm_dist  = np.abs(current_state[0]) / limit_distance
            
            # 2. Apply SQUARED penalty (Quadratic cost)
            # This makes 0.1 dist -> 0.01 penalty (Negligible)
            # But 0.9 dist -> 0.81 penalty (Massive!)
            r = 1.0 - (norm_angle * 0.4) - (norm_dist**2 * 0.8)
            
            # 3. Clip to ensure survival is always better than death (0.0)
            reward_history.append(max(0.1, r))

    # --- BACKPROP ---
    returns = calculate_returns(reward_history)
    dw1, db1, dw2, db2 = 0, 0, 0, 0
    
    for i in range(len(state_history)):
        target = np.array([[1], [0]]) if action_history[i] == -1 else np.array([[0], [1]])
        dZ2 = (prob_history[i] - target) * returns[i]
        dw2 += dZ2 @ h_history[i].T
        db2 += dZ2
        
        dZ1 = (weights_2.T @ dZ2) * (z1_history[i] > 0).astype(float)
        dw1 += dZ1 @ state_history[i].T
        db1 += dZ1

    batch_dw1 += dw1; batch_db1 += db1
    batch_dw2 += dw2; batch_db2 += db2

    # --- UPDATE ---
    if (_game + 1) % BATCH_SIZE == 0:
        weights_1 -= (batch_dw1 / BATCH_SIZE) * learning_rate
        biases_1  -= (batch_db1 / BATCH_SIZE) * learning_rate
        weights_2 -= (batch_dw2 / BATCH_SIZE) * learning_rate
        biases_2  -= (batch_db2 / BATCH_SIZE) * learning_rate
        batch_dw1.fill(0); batch_db1.fill(0)
        batch_dw2.fill(0); batch_db2.fill(0)

    # --- LOG ---
    avg_steps_1000 += len(reward_history)
    if _game % 1000 == 0 and _game > 0:
        avg = avg_steps_1000 / 1000
        print(f"Game {_game} | Avg Steps: {avg:.1f} | Var: {current_variance*100:.0f}%")
        
        if avg > success_threshold and current_variance >= MAX_VARIANCE:
            print(f"🏆 Solved (Gravity Well) after {_game} games!")
            break
        avg_steps_1000 = 0

# --- EXPORT ---
print("\nExporting weights...")
with open("model_reinforce.h", "w") as f:
    f.write(f"#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    f.write(f"#define INPUT_SIZE {INPUT_SIZE}\n#define HIDDEN_SIZE {nb_neurons}\n#define OUTPUT_SIZE 2\n\n")
    
    def dump(name, arr):
        f.write(f"inline const float {name}[{arr.size}] = {{ " + ", ".join([f"{x:.6f}f" for x in arr.flatten()]) + " };\n\n")
    
    dump("weights_1", weights_1); dump("biases_1", biases_1)
    dump("weights_2", weights_2); dump("biases_2", biases_2)
    f.write("#endif\n")