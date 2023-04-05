import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from gazebo_msgs.msg import ContactsState
from rclpy.qos import qos_profile_sensor_data

class CollisionDetectionSubscriber(Node):

    def __init__(self, cb, link_name):
        """
        Inputs:
            cb is the callback function passed as subscriber callback.
            link_name (str): name of the link that contacts msgs are being published for. For example for a given link_name
                of 'chassis_bottom_link', this will subscribe to ros topic: '/collision_detections/link_name'
        """
        super().__init__('collision_detection_' + link_name + '_subscriber')
        self.link_name = link_name
        # Set ros node's clock to use simulation time (gazebo time)
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time_parameter])

        self.subscription = self.create_subscription(
            ContactsState,
            '/collision_detection/' + link_name,
            cb,
            qos_profile = qos_profile_sensor_data
            )
        self.subscription  # prevent unused variable warning
    

    @staticmethod
    def get_non_ground_collisions(contactsState_msg):
        """
        Checks for collision detections with non-ground objects by checking the given msg. Returns the found ones as a dict.
        Inputs:
            contactsState_msg (ContactsState dict representation)
        Returns:
            unique_non_ground_contacts (dict): key is the foreign objects that the agent collided with. 
                Value is the corresponding ContactsState msg.
        """
        unique_non_ground_contacts = {}
        # Extract contact information
        for state in contactsState_msg["states"]:
            # find which collision_name is forklift's part and which is the foreign object it contacts
            if "forklift_bot" in state["collision1_name"]:
                forklift_contact_part_name = state["collision1_name"]
                foreign_contact_part_name = state["collision2_name"]
            else:
                forklift_contact_part_name = state["collision2_name"]
                foreign_contact_part_name = state["collision1_name"]
            # Check for non ground contact
            if foreign_contact_part_name != "ground_plane::link::collision":
                if foreign_contact_part_name not in unique_non_ground_contacts:
                    unique_non_ground_contacts[foreign_contact_part_name] = state

        return unique_non_ground_contacts
    

    @staticmethod
    def initialize_collision_detection_subscriber(env, link_name):
        """
        Returns an instance of class CollisionDetectionSubscriber which is is a ros node that has subscribed to collision topic
        associated with the given link_name.
        Inputs:
            link_name (str): name of the link that contacts msgs are being published for. For example for a given link_name
                of 'chassis_bottom_link', this will subscribe to ros topic: '/collision_detections/link_name'
        """
        # callback function for subscribing
        def collision_detection_cb(msg):
            env.collision_detection_states[link_name] = msg

        return CollisionDetectionSubscriber(collision_detection_cb, link_name)




def main(args=None):
    rclpy.init(args=args)

    collision_detection_subscriber = CollisionDetectionSubscriber()

    rclpy.spin(collision_detection_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    collision_detection_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()