# bash FoundationPose/run_container.sh

# From Container:
# python3 depthai-python/examples/install_requirements.py
# cd FoundationPose && bash build_all.sh

from Utils import *
from estimater import *
from datareader import *
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import tkinter as tk
from pathlib import Path
import depthai as dai

set_logging_format()
set_seed(0)

root = tk.Tk()
root.withdraw()  # Hide the root window

mesh_path = "/home/cnboonhan/workspaces/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj"
mesh = trimesh.load(mesh_path)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx)

