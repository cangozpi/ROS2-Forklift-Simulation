import os
from ament_index_python.packages import get_package_share_directory
import xacro

# from demo imports
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from multiprocessing import Process

def get_robot_description_raw():
    """
    Reads robots xacro files, converts it to xml and returns it.
    """
    # specify the name of the package and path to xacro file within the package
    forklift_robot_pkg_name = 'forklift_robot'
    urdf_file_name = 'forklift.urdf.xacro'

    # use xacro to process the file
    xacro_file = os.path.join(get_package_share_directory(forklift_robot_pkg_name), urdf_file_name)
    robot_description_raw = xacro.process_file(xacro_file).toxml()
    return robot_description_raw


def generateLaunchDescriptionForkliftEnv():
    """
    Generates Launch Description for starting gazebo, spawning forklift model, and loading controllers and returns it.
    """

    # specify the name of the package and path to xacro file within the package
    forklift_robot_pkg_name = 'forklift_robot'

    # Get robot description as xml
    robot_description_raw = get_robot_description_raw()


    # robot_state_publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw,
        'use_sim_time': True}] # add other parameters here if required
    )

    # Set the path to the world file
    forklift_gym_env_pkg_name = 'forklift_gym_env'
    world_file_name = 'road.world'
    world_path = os.path.join(get_package_share_directory(forklift_gym_env_pkg_name), world_file_name)

    world = LaunchConfiguration('world')

    declare_world_arg = DeclareLaunchArgument(
    name='world',
    default_value=world_path,
    description='Full path to the world model file to load')




    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('gazebo_ros'), '/launch', '/gazebo.launch.py']
        ),
        launch_arguments={'world': world}.items(),
    )

    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'forklift_bot',
                                   '-x', '10',
                                   '-y', '10',
                                   '-z', '0.30'],
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
        declare_world_arg, # Launch argument
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
    # The daemon process is terminated automatically before the main program exits,
    # to avoid leaving orphaned processes running
    process.daemon = True # refer to https://stackoverflow.com/questions/25391025/what-exactly-is-python-multiprocessing-modules-join-method-doing
    process.start()
    # process.join()

    return process


def generate_and_launch_ros_description_as_new_process():
    """
    Generates the Launch Description and starts it on a new process. Returns a reference to the started process.
    """
    # Start gazebo simulation, spawn forklift model, load ros controllers
    launch_desc = generateLaunchDescriptionForkliftEnv() # generate launch description
    launch_subp = startLaunchServiceProcess(launch_desc) # start the generated launch description on a subprocess
    return launch_subp