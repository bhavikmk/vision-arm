import numpy as np
import matplotlib.pyplot as plt

# --- 1. Motor & System Parameters ---
# NOTE: These parameters are representative of a single heavy-duty industrial joint.
# Mechanical Properties
P_POLES = 50        # Number of rotor pole pairs (e.g., 1.8 degree step -> 50 pole pairs)
M_J = 5.0           # Motor and Load Inertia (kg*m^2)
M_DAMPING = 0.5     # Viscous damping coefficient

# Electrical Properties (for FOC inner loop simulation)
M_R = 0.5           # Winding Resistance (Ohms)
M_L = 0.001         # Winding Inductance (Henry)
M_K_T = 2.0         # Torque constant (Nm/A)
M_Ke = 2.0          # Back-EMF Constant (V/(rad/s)), often == M_K_T in SI units

# System Limits
LOAD_TORQUE = 15.0 # Constant external load torque (e.g., from gravity) (Nm)
MAX_TORQUE = 50.0  # Maximum motor torque (Nm)

# Control Parameters
SAMPLE_RATE = 10000  # 10 kHz (Control loop frequency)
T_S = 1.0 / SAMPLE_RATE # Sample time

# --- 2. PID Controller Class ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limit=100):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.last_measurement = 0
        self.output_limit = output_limit # Output limit for anti-windup/saturation

    def compute(self, measurement):
        error = self.setpoint - measurement
        
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term 
        self.integral += error * T_S
        I_term = self.Ki * self.integral
        
        # Derivative term on measurement to prevent "derivative kick" on setpoint changes
        derivative = (measurement - self.last_measurement) / T_S
        D_term = -self.Kd * derivative # Note the negative sign
        
        # Compute output
        output = P_term + I_term + D_term
        
        # Update state 
        self.last_measurement = measurement
        
        # Output saturation (for torque/current limits)
        if output > self.output_limit:
            output = self.output_limit
        elif output < -self.output_limit:
            output = -self.output_limit
        
        # Anti-windup clamping (only clamp integral if output is saturated)
        if output >= self.output_limit or output <= -self.output_limit:
            self.integral -= error * T_S # Undo the last integral step
            
        return output

# --- 3. Motor Electrical & Mechanical Models ---

def electrical_dynamics(voltage_q, current_q, speed_mechanical, T_S):
    """ Simulates the motor's electrical response (inner FOC loop dynamics). """
    # Electrical speed = mechanical speed * pole pairs
    speed_electrical = speed_mechanical * P_POLES
    
    # Back-EMF opposes the applied voltage
    back_emf = M_Ke * speed_electrical
    
    # Vq = R*Iq + L*dIq/dt + V_bemf  =>  dIq/dt = (Vq - R*Iq - V_bemf) / L
    current_q_derivative = (voltage_q - M_R * current_q - back_emf) / M_L
    
    # Integrate to find new current
    new_current_q = current_q + current_q_derivative * T_S
    return new_current_q

def motor_dynamics(torque, current_speed, load_torque, T_S):
    # Angular acceleration:
    angular_acceleration = (torque - M_DAMPING * current_speed - load_torque) / M_J
    
    # New speed (using Euler integration):
    new_speed = current_speed + angular_acceleration * T_S
    
    # New position (using Euler integration):
    # Position is calculated externally in the main loop for simplicity

    return new_speed, angular_acceleration

# --- 4. Main Simulation Loop ---

# System Limits (derived)
MAX_CURRENT = MAX_TORQUE / M_K_T # Max current to produce max torque
MAX_VOLTAGE = 48.0 # Assume a 48V DC bus voltage limit

# Control Loop Instances
# NOTE: The previous gains were not suitable for a high-inertia system with a constant load.
# The new gains are tuned to be more stable and effective.
# Position loop: Slower, with D-term for stability.

position_controller = PIDController(Kp=5.0, Ki=0.5, Kd=2.0)

# Speed controller: Now includes an Integral (I) term to eliminate steady-state error from the load torque.
speed_controller = PIDController(Kp=10.0, Ki=20.0, Kd=0.1, output_limit=MAX_CURRENT)

# Current controller: Remains a fast inner loop.
current_controller = PIDController(Kp=3.5, Ki=4000, Kd=0.0, output_limit=MAX_VOLTAGE)

# Simulation Setup
TIME_END = 2.0  # seconds
num_steps = int(TIME_END / T_S)
time_array = np.linspace(0, TIME_END, num_steps)

# Motion profile (Target Position - e.g., a 10 radian move)
target_position = np.full(num_steps, 10.0)
target_position[0:5000] = 0.0 # Wait for 0.5s before moving

# Initial conditions
position = 0.0 # rad
speed = 0.0    # rad/s (mechanical)
current_q = 0.0 # Amps

# Data logging
position_history = []
speed_history = []
torque_history = []
current_q_ref_history = []
current_q_actual_history = []
position_setpoint_history = []

# Loop
for i in range(num_steps):
    # --- Cascade Control Loop ---
    
    # 1. OUTER LOOP: Position Control
    # Input: Target Position, Actual Position
    # Output: Commanded Speed Setpoint (rad/s)
    position_controller.setpoint = target_position[i]
    speed_setpoint = position_controller.compute(position)
    
    # 2. MIDDLE LOOP: Speed Control
    # Input: Target Speed, Actual Speed
    # Output: Commanded Q-axis Current Setpoint (Iq_ref)
    speed_controller.setpoint = speed_setpoint
    current_q_ref = speed_controller.compute(speed)
    
    # 3. INNER LOOP: Current Control (This is the core of FOC)
    # Input: Target Current, Actual Current
    # Output: Commanded Q-axis Voltage (Vq)
    current_controller.setpoint = current_q_ref
    voltage_q_cmd = current_controller.compute(current_q)
    
    # --- System Dynamics ---
    
    # 4. Electrical Dynamics: Update the actual current based on voltage command
    current_q = electrical_dynamics(voltage_q_cmd, current_q, speed, T_S)
    
    # 5. Torque Production: Actual torque is proportional to actual current
    torque_actual = M_K_T * current_q
    
    # 6. Mechanical Dynamics: Update speed and position based on actual torque
    speed, acceleration = motor_dynamics(torque_actual, speed, LOAD_TORQUE, T_S)
    position += speed * T_S # Update position based on new speed (Encoder feedback)
    
    # Log data
    position_history.append(position)
    speed_history.append(speed)
    torque_history.append(torque_actual)
    current_q_ref_history.append(current_q_ref)
    current_q_actual_history.append(current_q)
    position_setpoint_history.append(target_position[i])

# --- 5. Visualization ---
plt.figure(figsize=(12, 8))

# Position Plot
plt.subplot(3, 1, 1)
plt.plot(time_array, position_setpoint_history, label='Target Position (rad)', linestyle='--')
plt.plot(time_array, position_history, label='Actual Position (rad)', color='C1')
plt.title('Closed-Loop Stepper Motor Simulation: Position Control')
plt.ylabel('Position (rad)')
plt.grid(True)
plt.legend()

# Speed Plot
plt.subplot(3, 1, 2)
plt.plot(time_array, speed_history, label='Actual Speed (rad/s)', color='C1')
plt.ylabel('Speed (rad/s)')
plt.grid(True)
plt.legend()

# Current & Torque Plot
plt.subplot(3, 1, 3)
plt.plot(time_array, current_q_ref_history, label='Target Current (A)', linestyle='--')
plt.plot(time_array, current_q_actual_history, label='Actual Current (A)', color='C1')
plt.plot(time_array, torque_history, label='Torque Command (Nm)')
plt.xlabel('Time (s)')
plt.ylabel('Current (A) / Torque (Nm)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()