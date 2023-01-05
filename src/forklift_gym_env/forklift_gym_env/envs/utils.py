import os
from ament_index_python.packages import get_package_share_directory
import xacro

# from demo imports
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from multiprocessing import Process


def generateLaunchDescriptionForkliftEnv():
    """
    Generates Launch Description for starting gazebo, spawning forklift model, and loading controllers and returns it.
    """

    # specify the name of the package and path to xacro file within the package
    pkg_name = 'forklift_robot'
    urdf_file_name = 'forklift.urdf.xacro'

    # use xacro to process the file
    xacro_file = os.path.join(get_package_share_directory(pkg_name), urdf_file_name)
    robot_description_raw = xacro.process_file(xacro_file).toxml()


    # robot_state_publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw,
        'use_sim_time': True}] # add other parameters here if required
    )


    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('gazebo_ros'), '/launch', '/gazebo.launch.py']
        ),
    )

    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'forklift_bot'],
                        output='screen')

    load_joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_broad'],
        output='screen'
    )

    fork_joint_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'fork_joint_controller'],
        output='screen'
    )

    diff_controller = ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                'diff_cont'],
            output='screen'
        )


    return LaunchDescription([
        RegisterEventHandler(
          event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_controller,
                on_exit=[fork_joint_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=fork_joint_controller,
                on_exit=[diff_controller],
            )
        ),
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
    ])

def startLaunchServiceProcess(launchDesc):
    """Starts a Launch Description on a subprocess and returns it.
    Input:
         launchDesc : LaunchDescription obj.
    Return:
        started subprocess
    """
    # Create the LauchService and feed the LaunchDescription obj. to it.
    launchService = LaunchService()
    launchService.include_launch_description(launchDesc)
    process = Process(target=launchService.run)
    #The daemon process is terminated automatically before the main program exits,
    # to avoid leaving orphaned processes running
    process.daemon = True # refer to https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing
    process.start()
    print("yoo process called", process)
    process.join()
    print("HOOOJJJ")

    return process