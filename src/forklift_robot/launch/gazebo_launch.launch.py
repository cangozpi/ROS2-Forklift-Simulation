import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # Specify the name of the package and path to xacro file within the package
    pkg_name = 'forklift_robot'
    urdf_file_name = 'forklift.urdf.xacro'

    # Use xacro to process the file
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


    # rviz start node
    rviz_config_file_name = 'forklift_rviz.rviz'
    forklift_rviz_file_path = os.path.join(get_package_share_directory(pkg_name), rviz_config_file_name) #TODO: set this path to your .rviz configuration file path
    rviz_start_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        name='rviz2',
        arguments=['-d', forklift_rviz_file_path] 
    )



    # joint_state_publisher_gui start node
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
    )
    

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
        )


    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                    arguments=['-topic', 'robot_description',
                                '-entity', 'forklift_bot'],
                    output='screen')




    # Run the node
    return LaunchDescription([
        node_robot_state_publisher,
        rviz_start_node,
        joint_state_publisher_gui_node,
        gazebo,
        spawn_entity
    ])