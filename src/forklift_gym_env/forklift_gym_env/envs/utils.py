import os
from ament_index_python.packages import get_package_share_directory
import xacro
import yaml
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from multiprocessing import Process
from enum import Enum


def set_GAZEBO_MODEL_PATH():
    """
    exports GAZEBO_MODEL_PATH environment variable by extending it with forklift_gym_env/models path.
    """
    forklift_gym_env_pkg_name = 'forklift_gym_env'
    # export pallet sdf models path to GAZEBO_MODEL_PATH environment variable so that it can be spawned with mesh files
    os.environ['GAZEBO_MODEL_PATH'] = os.environ['GAZEBO_MODEL_PATH'] + \
        os.path.join(get_package_share_directory(forklift_gym_env_pkg_name), "../../../../", \
            "src/forklift_gym_env/models") + ":"


def set_GAZEBO_PLUGIN_PATH():
    """
    exports GAZEBO_PLUGIN_PATH environment variable by extending it with build/ros_gazebo_plugins path.
    """
    ros_gazebo_plugins_pkg_name = 'ros_gazebo_plugins'
    # export build/ros_gazebo_plugins path so that custom plugins defined in that pkg can be loaded into gazebo
    os.environ['GAZEBO_PLUGIN_PATH'] = os.environ['GAZEBO_PLUGIN_PATH'] + \
        os.path.join(get_package_share_directory(ros_gazebo_plugins_pkg_name), "../../../../", \
            "build/ros_gazebo_plugins") + ":"


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


def get_pallet_model_description_raw(model_path):
    """
    Reads model (sdf) file for the pallet and returns it as string.
    Input:
        model_path (str): path of the sdf model file of the pallet model wrt build/forklift_gym_env/models/ .
    """
    # specify the name of the package and path to xacro file within the package

    model_file_path = os.path.join('build/forklift_gym_env/models', model_path)
    with open(model_file_path) as f:
        pallet_model_description_raw = f.read()

    return pallet_model_description_raw


def generateLaunchDescriptionForkliftEnv(config):
    """
    Generates Launch Description for starting gazebo, spawning forklift model, and loading controllers and returns it.
    Input:
        config (dict): config dict that corresponds to config/config.yaml configurations
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
    world_file_name = config['world_file_name']
    world_path = os.path.join(get_package_share_directory(forklift_gym_env_pkg_name), world_file_name)

    world = LaunchConfiguration('world')

    declare_world_arg = DeclareLaunchArgument(
    name = 'world',
    default_value = world_path,
    description = 'Full path to the world model file to load')



    gui = LaunchConfiguration('gui')
    declare_gui_arg = DeclareLaunchArgument(
    name = 'gui',
    default_value = str(config['gui']),
    description = 'Whether to launch gzclient. gzclient is run when it is True.')
    assert type(config['gui']) == bool

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('gazebo_ros'), '/launch', '/gazebo.launch.py']
        ),
        launch_arguments={'world': world, 'gui': gui}.items(),
    )



    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'forklift_bot',
                                   '-x', '10',
                                   '-y', '10',
                                   '-z', str(config['agent_pose_position_z'])],
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
        declare_gui_arg, # Launch argument
        # RegisterEventHandler(
        #   event_handler=OnProcessExit(
        #         target_action=spawn_entity,
        #         on_exit=[load_joint_state_controller],
        #     )
        # ),
        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=load_joint_state_controller,
        #         on_exit=[fork_joint_controller],
        #     )
        # ),
        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=fork_joint_controller,
        #         on_exit=[diff_controller],
        #     )
        # ),
        gazebo,
        node_robot_state_publisher,
        # spawn_entity,
    ])


def startLaunchServiceProcess(launchDesc):
    """Starts a Launch Description on a subprocess and returns it.
    Input:
         launchDesc : LaunchDescription obj.
    Return:
        started subprocess
    """
    # extend GAZEBO_MODEL_PATH with forklift_gym_env/models path.
    set_GAZEBO_MODEL_PATH()
    # extend GAZEBO_PLUGIN_PATH with ros_gazebo_plugins pkg path.
    set_GAZEBO_PLUGIN_PATH()

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


def generate_and_launch_ros_description_as_new_process(config):
    """
    Generates the Launch Description and starts it on a new process. Returns a reference to the started process.
    """
    # Start gazebo simulation, spawn forklift model, load ros controllers
    launch_desc = generateLaunchDescriptionForkliftEnv(config) # generate launch description
    launch_subp = startLaunchServiceProcess(launch_desc) # start the generated launch description on a subprocess
    return launch_subp


def read_yaml_config(config_path):
    """
    Inputs:
        config_path (str): path to config.yaml file
    Outputs:
        config (dict): parsed config.yaml parameters
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class ObservationType(Enum):
    """
    Specifies the observation space that is being used. Used in config.yaml
    """
    TF_ONLY = "tf_only"
    TF_AND_DEPTH_CAMERA_RAW = "tf and depth_camera_raw"


class RewardType(Enum):
    """
    Specifies the reward function being used. Used in config.yaml
    """
    L2_DIST = "L2_distance"


class ActionType(Enum):
    """
    Specifies the action space for the actions that will be taken by the agent. Used in config.yaml
    """
    DIFF_CONT = "differential_control"