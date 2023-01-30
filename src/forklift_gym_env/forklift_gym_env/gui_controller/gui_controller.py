import tkinter as tk
from tkinter import messagebox
# from tkinter import *
# from tkinter.ttk import *

# ROS Controller Publishers
import rclpy
from forklift_gym_env.envs.controller_publishers.diff_cont_cmd_vel_unstamped_publisher import DiffContCmdVelUnstampedPublisher
from geometry_msgs.msg import Twist

def main(): 

    # ------------------------------------------ ROS
    # Create publisher for controlling forklift robot's joints: ============================== 
    # --------------------  /diff_cont/cmd_vel_unstamped

    rclpy.init()
    diff_cont_cmd_vel_unstamped_publisher = DiffContCmdVelUnstampedPublisher()



    # ------------------------------------------ GUI
    window = tk.Tk()
    window.title("Forklift Controller GUI")
    # window.resizable(width=False, height=False)
    window.columnconfigure([0,1], weight=1, minsize=300)
    window.rowconfigure(0, weight=1, minsize=300)

    differential_control_frame = tk.Frame(
        master=window,
        relief=tk.SUNKEN,
        borderwidth=3
    )

    fork_joint_controller_frame = tk.Frame(
        master=window,
        relief=tk.SUNKEN,
        borderwidth=3
    )

    differential_control_frame.grid(row=0, column=0, sticky="nsew")
    fork_joint_controller_frame.grid(row=0, column=1, sticky="nsew")

    # differential_control_frame.grid_rowconfigure([0], weight=1)
    # differential_control_frame.grid_rowconfigure([1], weight=5)
    # differential_control_frame.grid_rowconfigure([2], weight=5)
    differential_control_frame.grid_columnconfigure([0], weight=1)
    fork_joint_controller_frame.grid_columnconfigure([0], weight=1)


    # Differential Controller Frame ------------------------------------------
    # Section Title
    differential_controller_label = tk.Label(master=differential_control_frame, text="Differential Controller")
    differential_controller_label.grid(row=0, column=0, sticky="nsew", pady=5)
    

    # Directional Key Instructions
    differential_control_keys_frame = tk.Frame(
        master=differential_control_frame,
        relief=tk.RAISED,
        borderwidth=1
    )
    differential_control_keys_frame.grid(row=1, column=0, sticky="nsew", pady=5)
    differential_control_keys_frame.grid_rowconfigure([0,1], weight=1)
    differential_control_keys_frame.grid_columnconfigure([0,1,2], weight=1)

    w_label = tk.Label(master=differential_control_keys_frame, text="W")
    w_label.grid(row=0, column=1, sticky="n")

    a_label = tk.Label(master=differential_control_keys_frame, text="A")
    a_label.grid(row=1, column=0, sticky="e")

    s_label = tk.Label(master=differential_control_keys_frame, text="S")
    s_label.grid(row=1, column=1, sticky="n")

    d_label = tk.Label(master=differential_control_keys_frame, text="D")
    d_label.grid(row=1, column=2, sticky="w")


    # Twist Inputs:
    differential_control_twist_inputs_frame = tk.Frame(
        master=differential_control_frame,
        relief=tk.FLAT,
        borderwidth=1
    )
    differential_control_twist_inputs_frame.grid(row=2, column=0, sticky="nsew")
    differential_control_twist_inputs_frame.grid_rowconfigure([0,1], weight=1)
    differential_control_twist_inputs_frame.grid_columnconfigure([0,1], weight=1)

    linear_x_label = tk.Label(master=differential_control_twist_inputs_frame, text="linear.x =")
    linear_x_entry = tk.Entry(master=differential_control_twist_inputs_frame)

    linear_x_label.grid(row=0, column=0, sticky="nse", pady=5)
    linear_x_entry.grid(row=0, column=1, sticky="nsw", pady=5)

    angular_z_label = tk.Label(master=differential_control_twist_inputs_frame, text="angular.z =")
    angular_z_entry = tk.Entry(master=differential_control_twist_inputs_frame)

    angular_z_label.grid(row=1, column=0, sticky="nse")
    angular_z_entry.grid(row=1, column=1, sticky="nsw")



    # Fork Joint Controller Frame ------------------------------------------
    # Section Title
    fork_joint_label = tk.Label(master=fork_joint_controller_frame, text="Fork Joint Controller", pady=5)
    fork_joint_label.grid(row=0, column=0, sticky="nsew")
    

    # Directional Key Instructions
    fork_joint_controller_keys_frame = tk.Frame(
        master=fork_joint_controller_frame,
        relief=tk.RAISED,
        borderwidth=1
    )
    fork_joint_controller_keys_frame.grid(row=1, column=0, sticky="nsew", pady=7.45)
    fork_joint_controller_keys_frame.grid_rowconfigure([0,1], weight=1)
    fork_joint_controller_keys_frame.grid_columnconfigure([0,1,2], weight=1)

    w_label = tk.Label(master=fork_joint_controller_keys_frame, text="Up_Key")
    w_label.grid(row=0, column=1, sticky="n")

    a_label = tk.Label(master=fork_joint_controller_keys_frame, text="Down_Key")
    a_label.grid(row=1, column=1, sticky="s")


    # Twist Inputs:
    fork_joint_controller_twist_inputs_frame = tk.Frame(
        master=fork_joint_controller_frame,
        relief=tk.FLAT,
        borderwidth=1
    )
    fork_joint_controller_twist_inputs_frame.grid(row=2, column=0, sticky="nsew")
    fork_joint_controller_twist_inputs_frame.grid_rowconfigure([0], weight=1)
    fork_joint_controller_twist_inputs_frame.grid_columnconfigure([0,1], weight=1)

    velocity_label = tk.Label(master=fork_joint_controller_twist_inputs_frame, text="velocity =")
    velocity_entry = tk.Entry(master=fork_joint_controller_twist_inputs_frame)

    velocity_label.grid(row=0, column=0, sticky="nse", pady=5)
    velocity_entry.grid(row=0, column=1, sticky="nsw", pady=5)



    # Key Pressed Events ------------------------------------------
    default_linear_x = 1.0 # Differential Controller linear.x value
    default_angular_z = 15.0 # Differential Controller angular.z value
    default_velocity = 1.0 # Fork Joint Controller velocity value
    linear_x_entry.insert(tk.END, str(default_linear_x))
    angular_z_entry.insert(tk.END, str(default_angular_z))
    velocity_entry.insert(tk.END, str(default_velocity))

    def handle_keypress(event):
        # Get up to date Entry values
        try:
            linear_x = float(linear_x_entry.get())
        except:
            linear_x = default_linear_x
            linear_x_entry.delete(0, tk.END)
            linear_x_entry.insert(tk.END, str(default_linear_x))
            tk.messagebox.showerror("Error", "linear.x must be of type float!")
        try:
            angular_z = float(angular_z_entry.get())
        except:
            angular_z = default_angular_z
            angular_z_entry.delete(0, tk.END)
            angular_z_entry.insert(tk.END, str(default_angular_z))
            tk.messagebox.showerror("Error", "angular.z must be of type float!")
        try:
            velocity = float(velocity_entry.get())
        except:
            velocity = default_velocity
            velocity_entry.delete(0, tk.END)
            velocity_entry.insert(tk.END, str(default_velocity))
            tk.messagebox.showerror("Error", "velocity must be of type float!")


        # Handle Differential Controller Events -----------
        if event.keysym == 'w' or event.keysym == 'W':
            # convert diff_cont_action to Twist message
            diff_cont_msg = Twist()
            diff_cont_msg.linear.x = float(linear_x) # use this one
            diff_cont_msg.linear.y = 0.0
            diff_cont_msg.linear.z = 0.0

            diff_cont_msg.angular.x = 0.0
            diff_cont_msg.angular.y = 0.0
            diff_cont_msg.angular.z = 0.0 # use this one
            # Take action
            diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)

        elif event.keysym == 'a' or event.keysym == 'A':
            # convert diff_cont_action to Twist message
            diff_cont_msg = Twist()
            diff_cont_msg.linear.x = 0.0 # use this one
            diff_cont_msg.linear.y = 0.0
            diff_cont_msg.linear.z = 0.0

            diff_cont_msg.angular.x = 0.0
            diff_cont_msg.angular.y = 0.0
            diff_cont_msg.angular.z = float(angular_z) # use this one
            # Take action
            diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)

        elif event.keysym == 's' or event.keysym == 'S':
            # convert diff_cont_action to Twist message
            diff_cont_msg = Twist()
            diff_cont_msg.linear.x = -float(linear_x) # use this one
            diff_cont_msg.linear.y = 0.0
            diff_cont_msg.linear.z = 0.0

            diff_cont_msg.angular.x = 0.0
            diff_cont_msg.angular.y = 0.0
            diff_cont_msg.angular.z = 0.0 # use this one
            # Take action
            diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)

        elif event.keysym == 'd' or event.keysym == 'D':
            # convert diff_cont_action to Twist message
            diff_cont_msg = Twist()
            diff_cont_msg.linear.x = 0.0 # use this one
            diff_cont_msg.linear.y = 0.0
            diff_cont_msg.linear.z = 0.0

            diff_cont_msg.angular.x = 0.0
            diff_cont_msg.angular.y = 0.0
            diff_cont_msg.angular.z = -float(angular_z) # use this one
            # Take action
            diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)
        

        # Handle Fork Joint Controller Events -----------
        elif event.keysym == "Up":
            pass
        elif event.keysym == "Down":
            pass
    


    # Bind keypress event to handle_keypress()
    window.bind("<Key>", handle_keypress)
    def handle_set_focus(event):
        try:
            event.widget.focus_set()
        except:
            pass
    window.bind_all("<1>", handle_set_focus) # used to tk.make Entries loose focus when clicked on elsewhere



    # Start the Event Loop
    window.mainloop()


if __name__ == '__main__':
    main()