from omni.isaac.core import World
from pxr import Usd, UsdGeom
import omni

# Create default plane
world = World()
world.scene.add_default_ground_plane()

# Add a sphere
stage = omni.usd.get_context().get_stage()
xformPrim = UsdGeom.Xform.Define(stage, "/hello")
spherePrim = UsdGeom.Sphere.Define(stage, "/hello/sphere")
