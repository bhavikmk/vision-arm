import time

# --- PID Controller Class ---
# A helper class to manage the logic for each of the three loops (Position, Speed, Current).
class PIDController:
    """
    Implements a simple Proportional-Integral-Derivative (PID) controller.
    """
    def __init__(self, Kp, Ki, Kd, output_min, output_max):
        # PID Gains
        self.Kp = Kp  # Proportional Gain
        self.Ki = Ki  # Integral Gain
        self.Kd = Kd  # Derivative Gain
        
        # Output limits (important for "anti-windup" and motor safety)
        self.output_min = output_min
        self.output_max = output_max
        
        # State variables
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = time.time()

    def calculate(self, setpoint, measurement):
        current_time = time.time()
        # Calculate the time elapsed since the last calculation (dt)
        dt = current_time - self.last_time
        
        # 1. Error calculation
        error = setpoint - measurement
        
        # 2. Proportional Term (P)
        P_term = self.Kp * error
        
        # 3. Integral Term (I)
        self.integral += error * dt
        # Anti-windup clamping to prevent uncontrolled integral growth
        self.integral = max(self.output_min / self.Ki, min(self.output_max / self.Ki, self.integral))
        I_term = self.Ki * self.integral
        
        # 4. Derivative Term (D)
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        D_term = self.Kd * derivative
        
        # 5. Total Output
        output = P_term + I_term + D_term
        
        # 6. Output Clamping (Limit the command)
        output = max(self.output_min, min(self.output_max, output))
        
        # Update state for next cycle
        self.previous_error = error
        self.last_time = current_time
        
        return output


# --- Single Joint Control System ---
class RoboticArmJoint:
    """
    Represents one joint of the robotic arm, implementing a cascaded control loop.
    The gear ratio is 50:1 (50 motor turns = 1 output turn).
    """
    def __init__(self, joint_id):
        self.joint_id = joint_id
        print(f"Initializing Joint {joint_id}...")
        
        # --- Physical and System Parameters ---
        self.GEAR_RATIO = 50.0  # 50:1 Reduction
        self.POSITION_LIMITS = (-180, 180) # Joint angle limits in degrees
        self.current_position = 0.0      # Actual joint position (deg)
        self.current_speed = 0.0         # Actual joint speed (deg/s)
        self.current_torque = 0.0        # Simplified output torque
        self.current_draw = 0.0          # Simplified motor current (A)

        # The feedback from the sensor (Encoder/Resolver) is in the "motor" frame.
        # We need to convert the commanded position to the motor's required position.
        
        # --- CASCADED PID CONTROLLERS (UPDATED GAINS) ---
        
        # 1. Position Loop (Outer Loop) - INCREASED Kp
        # Output: Speed Command (deg/s)
        self.position_controller = PIDController(
            Kp=10.0, Ki=0.01, Kd=0.1, output_min=-100, output_max=100
        )
        # 2. Speed Loop (Middle Loop) - Slightly more aggressive
        # Output: Current/Torque Command (A)
        self.speed_controller = PIDController(
            Kp=15.0, Ki=0.1, Kd=0.05, output_min=-10, output_max=10
        )
        # 3. Current Loop (Inner Loop) - Already quite aggressive
        # Output: PWM Duty Cycle / Voltage Command (V)
        self.current_controller = PIDController(
            Kp=10.0, Ki=1.0, Kd=0.0, output_min=-1.0, output_max=1.0 # -1.0 to 1.0 = -100% to 100% duty cycle
        )

    def update_control_loop(self, position_setpoint):
        """
        Executes one iteration of the cascaded control loop.
        This function would run at the highest frequency (e.g., 8-32kHz).
        """
        # --- 1. Position Loop (Low Frequency: 1-4 kHz) ---
        # The setpoint is the desired joint angle (e.g., 90 deg).
        # The output is the required joint speed (V_C in the diagram).
        speed_command = self.position_controller.calculate(
            setpoint=position_setpoint,
            measurement=self.current_position
        )
        
        # --- 2. Speed Loop (Medium Frequency: 2-8 kHz) ---
        # The setpoint is the speed_command from the position loop.
        # The output is the required motor current (I_C in the diagram, often I_Q for FOC).
        current_command = self.speed_controller.calculate(
            setpoint=speed_command,
            measurement=self.current_speed
        )
        
        # --- 3. Current Loop (High Frequency: 8-32 kHz) ---
        # The setpoint is the current_command from the speed loop.
        # The motor current (current_draw) is the feedback (I_A, I_B, I_C in the diagram).
        # The output is the PWM duty cycle for the H-bridge (V_A, V_B, V_C in the diagram).
        pwm_duty_cycle = self.current_controller.calculate(
            setpoint=current_command,
            measurement=self.current_draw
        )
        
        # --- 4. Simulated Motor/Physical Response ---
        # In a real system, `pwm_duty_cycle` is sent to the PWM Unit.
        # For simulation, we'll use it to update the joint state.
        
        ## Simplified Physics: PWM -> Current -> Speed -> Position
        self.current_draw += (pwm_duty_cycle - self.current_draw) * 1  # Simulate current response
                
        self.current_speed += (self.current_draw * 10.0 - self.current_speed * 0.1) * 0.4 # Torque/Current -> Speed (with friction)
        
        self.current_position += self.current_speed * (time.time() - self.position_controller.last_time)* 180/3.14159 # Speed -> Position
        
        # Clamp position to limits
        self.current_position = max(self.POSITION_LIMITS[0], min(self.POSITION_LIMITS[1], self.current_position))
        
        return {
            "position": self.current_position,
            "speed_cmd": speed_command,
            "current_cmd": current_command,
            "pwm_out": pwm_duty_cycle
        }

    def reset(self):
        """Resets the state of the joint and its controllers."""
        self.current_position = 0.0
        self.current_speed = 0.0
        self.current_draw = 0.0
        self.position_controller.integral = 0.0
        self.speed_controller.integral = 0.0
        self.current_controller.integral = 0.0
        self.position_controller.previous_error = 0.0
        self.speed_controller.previous_error = 0.0
        self.current_controller.previous_error = 0.0


# --- Application Example ---
def run_robotic_arm_simulation():
    """Simulates moving a single joint to a setpoint."""
    
    # 1. Instantiate a Joint
    joint_1 = RoboticArmJoint(joint_id=1)
    
    # Target Position (e.g., move the joint 90 degrees)
    TARGET_POSITION = 90.0
    
    print("\n--- Starting Simulation ---")
    print(f"Targeting Position: {TARGET_POSITION:.2f} degrees")
    print("-" * 30)

    # Simulation loop
    for i in range(200):
        # A small pause to simulate the time between control loop cycles
        time.sleep(0.01) # Simulating a 10ms (100Hz) main loop update rate
        
        # Execute the control loop
        control_data = joint_1.update_control_loop(TARGET_POSITION)
        
        # Output status every 20 cycles
        if i % 20 == 0:
            print(f"Cycle {i:3} | Pos: {control_data['position']:7.2f} | Speed Cmd: {control_data['speed_cmd']:6.2f} | Current Cmd: {control_data['current_cmd']:5.2f} | PWM Out: {control_data['pwm_out']:4.2f}")
            
        # Stop condition: close enough to the target
        if abs(control_data['position'] - TARGET_POSITION) < 0.1 and i > 100:
            print("\n** Target position reached! **")
            break

    print("-" * 30)
    print(f"Final Position: {joint_1.current_position:.2f} degrees")


# Execute the simulation
if __name__ == "__main__":
    run_robotic_arm_simulation()