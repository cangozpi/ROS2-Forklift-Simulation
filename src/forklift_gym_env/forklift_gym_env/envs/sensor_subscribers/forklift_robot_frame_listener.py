import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class FrameListener(Node):

    def __init__(self):
        super().__init__('forklift_robot_frame_listener')
        # Set ros node's clock to use simulation time (gazebo time)
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time_parameter])
        # print("KK", self.get_parameter('use_sim_time').get_parameter_value().bool_value, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # Declare and acquire `target_frame` parameter

        self.target_frame = "odom" #TODO: this should replaced with the world frame

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def get_transforms(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = self.target_frame
        to_frame_rel = 'chassis_bottom_link'

        # Look up for the transformation between target_frame and turtle2 frames
        # and send velocity commands for turtle2 to reach target_frame
        try:
            # t = self.tf_buffer.lookup_transform(
            #     to_frame_rel,
            #     from_frame_rel,
            #     rclpy.time.Time())

            t = self.tf_buffer.lookup_transform(
                from_frame_rel,
                to_frame_rel,
                rclpy.time.Time())
            return t

        except TransformException as ex:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return



def main():
    rclpy.init()
    node = FrameListener()
    try:
        # rclpy.spin(node)
        for i in range(5): 
            rclpy.spin_once(node)
            node.get_transforms()
            # rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == "__main__":
    main()