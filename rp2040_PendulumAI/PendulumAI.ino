#include <Wire.h>
#include <MPU6050.h>
#include <math.h>     
#include "model_ppo.h" 

// --- 1. CONFIGURATION ---
const int PIN_IR_SENSOR = 26; 
const int PIN_MOTOR_PWM = 28;
const int PIN_MOTOR_DIR = 29;

// --- 2. SENSORS ---
const float IR_CENTER_VOLTAGE = 1.30; 
const float ANGLE_SHIFT = 1.7; // Tweak this if it leans
const int DISTANCE_SMOOTHING = 200; // Lowered slightly to save time

// IMU AVERAGING (New Feature)
// We read the Gyro/Accel this many times and average them to remove noise.
// 5 samples takes about ~3ms extra via I2C.
const int IMU_SAMPLES = 5; 

const float IR_SLOPE_NEG = -9.24;
const float IR_SLOPE_POS = -1.92;

// --- 3. PHYSICS CONSTANTS ---
const float DT = 0.02; // Target Loop Time: 20ms (50Hz)
const float ALPHA = 0.995;
const float GYRO_SCALE = 65.5; 

// --- TUNING ---
const int MIN_MOVING_PWM = 45; 
const float SMOOTH_FACTOR = 0.5; // 0.5 = Balanced smoothness

const float POS_DECAY = 0.95;
const float ANG_DECAY = 0.99;

// --- GLOBALS ---
MPU6050 mpu;
float state[4] = {0.0, 0.0, 0.0, 0.0}; 
float int_pos = 0.0;
float int_ang = 0.0;
float last_x = 0.0;
unsigned long last_time_micros = 0;
unsigned long loop_start_time = 0;
bool is_balancing = false; 
float smoothed_force = 0.0;

// Buffers
float layer_1[HIDDEN_SIZE]; 
float layer_2[HIDDEN_SIZE]; 

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10); 

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  pinMode(PIN_MOTOR_PWM, OUTPUT);
  pinMode(PIN_MOTOR_DIR, OUTPUT);
  analogWrite(PIN_MOTOR_PWM, 0);
  analogReadResolution(12);

  Wire.setSDA(6); Wire.setSCL(7); Wire.setClock(400000); Wire.begin();
  mpu.initialize();
  
  // Reduced ranges slightly to reduce sensitivity to vibration
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4); // Was 8, 4 is more precise for balancing
  mpu.setDLPFMode(MPU6050_DLPF_BW_20);
  mpu.setZGyroOffset(0);

  delay(1000);
  Serial.println("System Ready.");
  last_time_micros = micros();
}

void loop() {
  // --- 1. HZ REGULATOR ---
  unsigned long now = micros();
  if (now - last_time_micros < DT * 1000000) return;
  
  // Calculate Actual Hz for display
  float actual_dt = (now - last_time_micros) / 1000000.0;
  float actual_hz = 1.0 / actual_dt;
  last_time_micros = now;

  // --- 2. INPUT ---
  if (Serial.available() > 0) {
    if (Serial.read() == 'l') { 
      is_balancing = !is_balancing; 
      if (is_balancing) {
        digitalWrite(LED_BUILTIN, HIGH); 
        state[1] = 0; last_x = state[0]; 
        int_pos = 0.0; int_ang = 0.0;
        smoothed_force = 0.0; 
      } else {
        digitalWrite(LED_BUILTIN, LOW);
        analogWrite(PIN_MOTOR_PWM, 0); 
      }
    }
  }

  // --- 3. EXECUTION ---
  readSensorsWithAveraging(); // <--- NEW FUNCTION
  float raw_force = runInference();

  smoothed_force = (smoothed_force * (1.0 - SMOOTH_FACTOR)) + (raw_force * SMOOTH_FACTOR);
  driveMotor(smoothed_force); 

  // --- 4. DIAGNOSTICS (Updated) ---
  static int print_skip = 0;
  if (print_skip++ > 10) { 
    print_skip = 0;
    
    // Safety check: If Hz drops below 45, we are lagging!
    if (actual_hz < 45.0) Serial.print("⚠️ LAG! ");
    
    Serial.print("Hz:"); Serial.print(actual_hz, 1);
    Serial.print(" | Ang:"); Serial.print(state[2] * RAD_TO_DEG, 1);
    Serial.print(" | Pos:"); Serial.print(state[0], 2);
    Serial.print(" | F:"); Serial.println(smoothed_force, 2);
  }
}

// --- NEW SENSOR READING LOGIC ---
void readSensorsWithAveraging() {
  long ax_sum = 0, ay_sum = 0, gz_sum = 0;
  
  // OVERSAMPLING LOOP
  // Reads the hardware 5 times to smooth out vibration noise
  for (int i = 0; i < IMU_SAMPLES; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    ax_sum += ax;
    ay_sum += ay;
    gz_sum += gz;
  }

  // Average them
  float ax_avg = ax_sum / (float)IMU_SAMPLES;
  float ay_avg = ay_sum / (float)IMU_SAMPLES;
  float gz_avg = gz_sum / (float)IMU_SAMPLES;

  // --- ANGLE CALCULATION ---
  float accel_angle = atan2(ay_avg, ax_avg);
  accel_angle += (ANGLE_SHIFT * DEG_TO_RAD); 
  accel_angle = -1.0 * accel_angle;          

  float gyro_rate_rad = (gz_avg / GYRO_SCALE) * DEG_TO_RAD;

  // Complementary Filter
  state[2] = ALPHA * (state[2] + gyro_rate_rad * DT) + (1.0 - ALPHA) * accel_angle;
  state[3] = gyro_rate_rad;

  // --- POSITION READING ---
  // (Standard smoothing for Distance Sensor)
  long raw_sum = 0;
  for(int i=0; i< DISTANCE_SMOOTHING; i++) raw_sum += analogRead(PIN_IR_SENSOR);
  float raw_dist_avg = raw_sum / float(DISTANCE_SMOOTHING);
  
  float voltage = raw_dist_avg * (3.3 / 4095.0);
  float new_x = 0.0;
  if (voltage > IR_CENTER_VOLTAGE) new_x = (voltage - IR_CENTER_VOLTAGE) / IR_SLOPE_NEG;
  else new_x = (voltage - IR_CENTER_VOLTAGE) / IR_SLOPE_POS;

  float new_x_dot = (new_x - last_x) / DT;
  state[1] = 0.8 * state[1] + 0.2 * new_x_dot; 
  state[0] = new_x;
  last_x = new_x;

  int_pos = (int_pos * POS_DECAY) + state[0];
  int_ang = (int_ang * ANG_DECAY) + state[2];
}

float runInference() {
  // Same inference code...
  float obs[6];
  // Using slightly sedated inputs
  obs[0] = state[0] * 5.0f;       
  obs[1] = state[1] * 0.3f; // Reduced
  obs[2] = state[2] * 10.0f;      
  obs[3] = state[3] * 0.3f; // Reduced       
  obs[4] = int_pos  * 1.0f;       
  obs[5] = int_ang  * 2.0f;       

  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = b1[i]; 
    for (int j = 0; j < 6; j++) sum += w1[i * 6 + j] * obs[j]; 
    layer_1[i] = tanh(sum); 
  }
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = b2[i]; 
    for (int j = 0; j < HIDDEN_SIZE; j++) sum += w2[i * HIDDEN_SIZE + j] * layer_1[j];
    layer_2[i] = tanh(sum);
  }
  float action_val = b_out[0]; 
  for (int j = 0; j < HIDDEN_SIZE; j++) action_val += w_out[j] * layer_2[j];

  if (action_val > 1.0f) action_val = 1.0f;
  if (action_val < -1.0f) action_val = -1.0f;

  return action_val;
}

void driveMotor(float force_input) {
  if (!is_balancing) {
    analogWrite(PIN_MOTOR_PWM, 0);
    return;
  }
  if (abs(state[2]) > 0.40 || abs(state[0]) > 0.13) {
    is_balancing = false;
    digitalWrite(LED_BUILTIN, LOW); 
    analogWrite(PIN_MOTOR_PWM, 0);
    return;
  }

  int pwm = 0;
  float abs_force = abs(force_input);
  
  if (abs_force < 0.20) { // Slight deadband increase
      pwm = 0; 
  } else {
      long force_mapped = abs_force * 1000;
      pwm = map(force_mapped, 200, 1000, MIN_MOVING_PWM, 255);
  }
  if (pwm > 255) pwm = 255;

  if (force_input > 0) {
    digitalWrite(PIN_MOTOR_DIR, HIGH); 
    analogWrite(PIN_MOTOR_PWM, pwm);
  } else {
    digitalWrite(PIN_MOTOR_DIR, LOW); 
    analogWrite(PIN_MOTOR_PWM, pwm);
  }
}