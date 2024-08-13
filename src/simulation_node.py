from isaacsim import SimulationApp

# Example ROS2 bridge sample showing manual control over messages
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})
import carb
import omni
import omni.graph.core as og
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.nucleus import get_assets_root_path
from pxr import Sdf

# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

import os
path_here = os.path.dirname(os.path.abspath(__file__))

usd_path = os.path.join(path_here, "..", "min_example.usd")
omni.usd.get_context().open_stage(usd_path, None)

# Wait two frames so that stage starts loading
simulation_app.update()
simulation_app.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    simulation_app.update()
print("Loading Complete")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

simulation_context.play()
simulation_context.step()

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class SimTimeNode(Node):
    def __init__(self):
        super().__init__("sim_time_node")
        self.publisher = self.create_publisher(Float64, "/sim_time", 10)

rclpy.init()
sim_time_node = SimTimeNode()
while True:
    simulation_context.step()
    t = simulation_context.current_time
    print(t)
    sim_time_msg = Float64()
    sim_time_msg.data = t
    sim_time_node.publisher.publish(sim_time_msg)