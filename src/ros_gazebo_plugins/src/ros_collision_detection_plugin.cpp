#include "ros_collision_detection_plugin.hpp"


using namespace gazebo;
using namespace gazebo::ros_collision_detection_plugin;

/////////////////////////////////////////////////
ContactPlugin::ContactPlugin() : SensorPlugin()
{
}

/////////////////////////////////////////////////
ContactPlugin::~ContactPlugin()
{
    // Uncommenting below breaks reset() functionality of forklift_gym_env pkg's training loop by disrupting the ros nodes used by the env.
    // rclcpp::shutdown();
}

/////////////////////////////////////////////////
void ContactPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
{
    // similar to https://github.com/ros-simulation/gazebo_ros_pkgs/blob/melodic-devel/gazebo_plugins/src/gazebo_ros_bumper.cpp

    // Create ROS2 Node to publish collision detections
    try{
        rclcpp::init(0, nullptr);
    }catch(...){}

    // get ros topic name to use from sdf if given
    std::string ros_topic_name = "collision_detections";
    if (_sdf->HasElement("rosTopicName")){
        ros_topic_name = _sdf->GetElement("rosTopicName")->Get<std::string>();
    }

    this->node = rclcpp::Node::make_shared("collision_detection_publisher");
    this->ros_publisher = node->create_publisher<gazebo_msgs::msg::ContactsState>(ros_topic_name, 10);

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

    RCLCPP_INFO(this->node->get_logger(), "Loaded ros_collision_detection plugin. Publishing to topic: '%s'", (char*)ros_topic_name.c_str());
}

/////////////////////////////////////////////////
void ContactPlugin::OnUpdate()
{
    // Get contact sensor readings (i.e. ContactsState msg)
    const msgs::Contacts contacts = this->parentSensor->Contacts();
    // std::cout << contacts.contact_size() << " CANOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO " << "\n";

    // Convert gazebo_msg to gazebo_ros_msg
    gazebo_msgs::msg::ContactsState ros_contacts_msg = gazebo_ros::Convert<gazebo_msgs::msg::ContactsState>(contacts);

    // Publish msg to ROS topic
    this->ros_publisher->publish(ros_contacts_msg);

    if(rclcpp::ok()){
        rclcpp::spin_some(this->node);
    }

    // Uncomment below to print info from gazebo's contactsState msg
    // for (unsigned int i = 0; i < contacts.contact_size(); ++i)
    // {
    //     // std::cout << "Time: " << contacts.contact(i).time().has_sec() << "]\n";
    //     // std::cout << "Time: " << contacts.contact(i).time().sec() << "]\n";
    //     std::cout << "Time (kNsecFieldNumber): " << contacts.contact(i).time().kNsecFieldNumber << "\n";
    //     std::cout << "Time: (ksecFieldNumber)" << contacts.contact(i).time().kSecFieldNumber << "\n";
    //     std::cout << "Time: (sec())" << contacts.contact(i).time().sec() << "\n";
    //     std::cout << "Time: (nsec())" << contacts.contact(i).time().nsec() << "\n";


    //     std::cout << "Collision between[" << contacts.contact(i).collision1()
    //               << "] and [" << contacts.contact(i).collision2() << "]\n";

    //     for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j)
    //     {
    //         std::cout << j << "  Position:"
    //                   << contacts.contact(i).position(j).x() << " "
    //                   << contacts.contact(i).position(j).y() << " "
    //                   << contacts.contact(i).position(j).z() << "\n";
    //         std::cout << "   Normal:"
    //                   << contacts.contact(i).normal(j).x() << " "
    //                   << contacts.contact(i).normal(j).y() << " "
    //                   << contacts.contact(i).normal(j).z() << "\n";
    //         std::cout << "   Depth:" << contacts.contact(i).depth(j) << "\n";
    //     }
    // }
}


GZ_REGISTER_SENSOR_PLUGIN(ContactPlugin)