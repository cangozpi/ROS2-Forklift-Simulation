#include "CollisionDetectionPlugin.hh"


using namespace gazebo;
GZ_REGISTER_SENSOR_PLUGIN(ContactPlugin)

/////////////////////////////////////////////////
ContactPlugin::ContactPlugin() : SensorPlugin()
{
}

/////////////////////////////////////////////////
ContactPlugin::~ContactPlugin()
{
}

/////////////////////////////////////////////////
void ContactPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/)
{
    // Create ROS2 Node to publish collision detections
    // rclcpp::init(argc, argv);
    rclcpp::init(0, nullptr);
    auto node = rclcpp::Node::make_shared("collision_detection_publisher");
    // this->ros_publisher = node->create_publisher<std_msgs::msg::String>("collision_detections", 10);
    this->ros_publisher = node->create_publisher<gazebo_msgs::msg::ContactsState>("collision_detections", 10);

    // Get the parent sensor.
    this->parentSensor =
        std::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);

    // Make sure the parent sensor is valid.
    if (!this->parentSensor)
    {
        gzerr << "ContactPlugin requires a ContactSensor.\n";
        return;
    }

    // Connect to the sensor update event.
    this->updateConnection = this->parentSensor->ConnectUpdated(
        std::bind(&ContactPlugin::OnUpdate, this));

    // Make sure the parent sensor is active.
    this->parentSensor->SetActive(true);
}

/////////////////////////////////////////////////
void ContactPlugin::OnUpdate()
{
    // Get all the contacts.
    // const msgs::Contacts contacts;
    const msgs::Contacts contacts = this->parentSensor->Contacts();

    // Convert gazebo_msg to gazebo_ros_msg
    gazebo_msgs::msg::ContactsState message;
    // message.states = contacts

    gazebo_msgs::msg::ContactsState x = gazebo_ros::Convert<gazebo_msgs::msg::ContactsState>(contacts);
    std::cout << x.states[0].collision1_name << " CAAAAAAAAAAAAAAAAANNN" << "\n";
    for (unsigned int i = 0; i < contacts.contact_size(); ++i)
    {
        auto cur_ros_msg = gazebo_msgs::msg::ContactState();
        auto cur_gz_msg = contacts.contact(i);
        cur_ros_msg.set__collision1_name(cur_gz_msg.collision1());
        cur_ros_msg.set__collision2_name(cur_gz_msg.collision2());
        message.states.push_back(cur_ros_msg);
        // std::cout << "Time: " << contacts.contact(i).time().has_sec() << "]\n";
        // std::cout << "Time: " << contacts.contact(i).time().sec() << "]\n";
        std::cout << "Time (kNsecFieldNumber): " << contacts.contact(i).time().kNsecFieldNumber << "\n";
        std::cout << "Time: (ksecFieldNumber)" << contacts.contact(i).time().kSecFieldNumber << "\n";
        std::cout << "Time: (sec())" << contacts.contact(i).time().sec() << "\n";
        std::cout << "Time: (nsec())" << contacts.contact(i).time().nsec() << "\n";


        std::cout << "Collision between[" << contacts.contact(i).collision1()
                  << "] and [" << contacts.contact(i).collision2() << "]\n";

        for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j)
        {
            std::cout << j << "  Position:"
                      << contacts.contact(i).position(j).x() << " "
                      << contacts.contact(i).position(j).y() << " "
                      << contacts.contact(i).position(j).z() << "\n";
            std::cout << "   Normal:"
                      << contacts.contact(i).normal(j).x() << " "
                      << contacts.contact(i).normal(j).y() << " "
                      << contacts.contact(i).normal(j).z() << "\n";
            std::cout << "   Depth:" << contacts.contact(i).depth(j) << "\n";
        }
    }
}