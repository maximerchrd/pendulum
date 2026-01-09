import numpy as np

# Simulation parameters
VARIANCE = 0.5
railLengthMargin = 0.092  
massCart = 0.15           
massPole = 0.06           
lengthPole = 0.18         
motorForce = 9.0
dt_phys_sim = 0.002
limit_angle = 0.21
limit_distance = 0.091
nb_macro_steps = 500
nb_games = 500000
sensor_noise_pos = 0.008
sensor_noise_vel = 0.08
sensor_noise_angle = 0.0015
sensor_noise_gyro = 0.08
success_threshold = 435

# Neural network parameters
nb_neurons = 24
starting_learning_rate = 0.002
learning_rate = starting_learning_rate
gamma = 0.98

weights_1 = np.random.uniform(-0.1, 0.1, (nb_neurons, 4))
biases_1 = np.random.uniform(-0.1, 0.1, (nb_neurons, 1))
weights_2 = np.random.uniform(-0.1, 0.1, (2, nb_neurons))
biases_2 = np.random.uniform(-0.1, 0.1, (2, 1))

# Lists to store rewards and actions
current_state = [0.0, 0.0, 0.0, 0.0]
current_action = -1

def calculate_returns(rewards):
    # rewards is a list like [1, 1, 1, 1, 1]
    
    returns = []
    G = 0
    
    # 1. Iterate BACKWARDS through the rewards
    for r in reversed(rewards):
        # 2. Update the running total
        G = r + gamma * G
        
        # 3. Insert at the front of the list (since we are going backwards)
        returns.insert(0, G)
        
    # returns is now [2.97, 1.99, 1.0] (matching the order of steps)
    
    # 4. (CRITICAL) Normalize the returns to make training stable
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns

def softmax(x):
    # Subtract max for numerical stability (prevents overflow)
    e_x = np.exp(x - np.max(x)) 
    return e_x / np.sum(e_x, axis=0) # assuming x is shape (2,1)

def physics_step(state, action):
    x, x_dot, theta, theta_dot = state
    for _ in range(5):
        # 1. Calculate Forces
        force = action * motorForce
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        total_mass = massCart + massPole
        pole_mass_len = massPole * lengthPole

        temp = (force + pole_mass_len * theta_dot**2 * sintheta) / total_mass
        thetaacc = (9.8 * sintheta - costheta * temp) / (lengthPole * (4.0/3.0 - massPole * costheta**2 / total_mass))
        xacc = temp - pole_mass_len * thetaacc * costheta / total_mass

        # 2. Integrate (Semi-Implicit: Update Velocity FIRST)
        x_dot += dt_phys_sim * xacc
        theta_dot += dt_phys_sim * thetaacc
        
        # 3. Update Position (using the NEW velocity)
        x += dt_phys_sim * x_dot
        theta += dt_phys_sim * theta_dot

    return (x, x_dot, theta, theta_dot)
    
average_steps = 0
for _game in range(nb_games):
    # randomize physics parameters
    massCart = 0.35 * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)
    massPole = 0.06 * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)
    lengthPole = 0.15 * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)
    motorForce = 8.0 * np.random.uniform(1.0 - VARIANCE, 1.0 + VARIANCE)

    current_state = [0.0, 0.0, 0.0, 0.0]
    current_action = -1
    state_history = []
    reward_history = []
    action_history = []
    probability_action_history = []
    h_history = []
    z1_history = []

    # This simulates a sensor that is permanently tilted by up to +/- 3 degrees
    angle_bias = np.random.uniform(-0.015, 0.015)

    for _step in range(nb_macro_steps):
        # Forward pass
        noisy_state = [
            (current_state[0] + np.random.normal(0, sensor_noise_pos)) * 5.0,
            current_state[1] + np.random.normal(0, sensor_noise_vel),
            ((current_state[2] + np.random.normal(0, sensor_noise_angle) + angle_bias)) * 20.0,
            current_state[3] + np.random.normal(0, sensor_noise_gyro)
        ]
        x_input = np.array(noisy_state).reshape(-1, 1)
        z_1 = weights_1 @ x_input + biases_1
        z1_history.append(z_1)
        h = np.maximum(0, z_1)
        h_history.append(h)
        z_2 = weights_2 @ h + biases_2
        p = softmax(z_2)

        current_action = -1
        if np.random.rand() < p[1][0]:
            current_action = 1
        
        probability_action_history.append(p)
        state_history.append(noisy_state)
        current_state = physics_step(current_state, current_action)
        action_history.append(current_action)
        if np.abs(current_state[0]) > limit_distance or np.abs(current_state[2]) > limit_angle:
             reward_history.append(0)
             break
        else:
            # A. Distance Cost: Punish drifting near the rails
            # (Normalized 0.0 to 1.0)
            dist_cost = (current_state[0] / limit_distance) ** 2
            
            # B. Angle Cost: AGGRESSIVELY punish leaning
            # We square it so big leans are punished massively
            angle_cost = (current_state[2] / limit_angle) ** 2
            
            # C. The "Upright" Formula
            # We give a base +1.0 for surviving.
            # We subtract a HUGE penalty for angle to force verticality.
            # We subtract a smaller penalty for distance to keep it centered.
            
            reward = 1.0 - (30.0 * dist_cost) - (30.0 * angle_cost)
            reward_history.append(reward)

    # Backpropagation
    dw1 = 0
    db1 = 0
    dw2 = 0
    db2 = 0
    returns_list = calculate_returns(reward_history)
    for i, state in enumerate(state_history):
        target = np.array([[1], [0]])
        if action_history[i] == 1:
            target = np.array([[0], [1]])

        dZ2 = (probability_action_history[i] - target) * returns_list[i]
        dw2 += dZ2 @ h_history[i].T * learning_rate
        db2 += dZ2 * learning_rate
        z1_deriv = np.zeros(z1_history[i].shape)
        for k, z1_row in enumerate(z1_history[i]):
            for l, z1_el in enumerate(z1_row):
              if z1_el > 0:
                  z1_deriv[k][l] = 1
        state_input = np.array(state_history[i]).reshape(-1, 1)

        error_hidden = weights_2.T @ dZ2 
        dZ1 = error_hidden * z1_deriv
        dw1 += (dZ1 @ state_input.T) * learning_rate
        db1 += dZ1 * learning_rate

    weights_1 -= dw1
    biases_1 -= db1
    weights_2 -= dw2
    biases_2 -= db2

    average_steps += len(reward_history)
    if _game % 1000 == 0:
        average_steps = average_steps / 1000
        learning_rate = starting_learning_rate * (nb_macro_steps - average_steps) / nb_macro_steps
        print(f"Game {_game} finished. Average Steps: {average_steps}")
        if average_steps > success_threshold:
            print(f"🏆 Solved after {_game} games!")
            break
        else:
            average_steps = 0


# --- 4. EXPORT C HEADER ---
print("Exporting weights to 'model_reinforce.h'...")

def array_to_c_string(name, array):
    flat = array.flatten()
    # Added 'inline' to prevent linker errors in C++
    content = f"inline const float {name}[{len(flat)}] = {{\n    "
    content += ", ".join([f"{x:.6f}f" for x in flat])
    content += "\n};\n\n"
    return content

with open("model_reinforce.h", "w") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    f.write(f"#define INPUT_SIZE 4\n")
    f.write(f"#define HIDDEN_SIZE {nb_neurons}\n")
    f.write(f"#define OUTPUT_SIZE 2\n\n")
    
    f.write(array_to_c_string("weights_1", weights_1))
    f.write(array_to_c_string("biases_1", biases_1))
    f.write(array_to_c_string("weights_2", weights_2))
    f.write(array_to_c_string("biases_2", biases_2))
    
    f.write("#endif\n")

print("Done! Weights saved.")