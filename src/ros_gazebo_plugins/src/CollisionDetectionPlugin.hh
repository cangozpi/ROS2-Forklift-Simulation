#ifndef _GAZEBO_COLLISION_DETECTION_PLUGIN_HH_
#define _GAZEBO_COLLISION_DETECTION_PLUGIN_HH_

#include <string>

#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>

#include "rclcpp/rclcpp.hpp"
#include "gazebo_msgs/msg/contacts_state.hpp"
#include "gazebo_msgs/msg/contact_state.hpp"
#include "gazebo_ros/conversions/gazebo_msgs.hpp"


namespace gazebo
{
    /// \brief An example plugin for a contact sensor.
    class ContactPlugin : public SensorPlugin
    {
        /// \brief Constructor.
    public:
        ContactPlugin();

        /// \brief Destructor.
    public:
        virtual ~ContactPlugin();

        /// \brief Load the sensor plugin.
        /// \param[in] _sensor Pointer to the sensor that loaded this plugin.
        /// \param[in] _sdf SDF element that describes the plugin.
    public:
        virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);

        /// \brief Callback that receives the contact sensor's update signal.
    private:
        virtual void OnUpdate();

        /// \brief Pointer to the contact sensor
    private:
        sensors::ContactSensorPtr parentSensor;

        /// \brief Connection that maintains a link between the contact sensor's
        /// updated signal and the OnUpdate callback.
    private:
        event::ConnectionPtr updateConnection;
    
    private:
        rclcpp::PublisherBase::SharedPtr ros_publisher;
    };
}
#endif