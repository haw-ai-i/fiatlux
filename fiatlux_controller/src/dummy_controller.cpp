#include "rclcpp/rclcpp.hpp"
int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("fiatlux_controller");
  RCLCPP_INFO(node->get_logger(), "Boilerplate controller node started.");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
