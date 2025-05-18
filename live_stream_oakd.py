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
from datetime import timedelta

# Initialize
set_logging_format()
set_seed(0)
root = tk.Tk()
root.withdraw()  # Hide the root window

# Configure 
mesh_path = "/home/cnboonhan/workspaces/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj"
mesh = trimesh.load(mesh_path)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

# FoundationPose
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx)

# Setup Oak-D
pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
color = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
sync = pipeline.create(dai.node.Sync)
xoutGrp = pipeline.create(dai.node.XLinkOut)
xoutGrp.setStreamName("xout")
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
color.setCamera("color")
sync.setSyncThreshold(timedelta(milliseconds=50))
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(sync.inputs["disparity"])
color.video.link(sync.inputs["video"])
sync.out.link(xoutGrp.input)
disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()

## Create Mask, to match size of the video frame
mask = np.ones((400, 640), dtype=np.uint8)

## Create K matrix, from running get_K.py under RGB Camera Default intrinsics
cam_K = np.array([[2297.10986328125, 0.0, 1938.3743896484375], [0.0, 2297.10986328125, 1100.5076904296875], [0.0, 0.0, 1.0]])


with dai.Device(pipeline) as device:
    queue = device.getOutputQueue("xout", 10, False)
    while True:
        msgGrp = queue.get()
        if msgGrp:
            for name, msg in msgGrp:
                if name == "video":
                    color = msg.getCvFrame().astype(np.uint8)
                if name == "disparity":
                    depth = msg.getCvFrame().astype(np.uint8)
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=3)
            cv2.waitKey(1)
            break

    while True:
        msgGrp = queue.get()
        for name, msg in msgGrp:
            if name == "video":
                color = msg.getCvFrame().astype(np.uint8)
            if name == "disparity":
                depth = msg.getCvFrame().astype(np.uint8)
        pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=3)
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        if cv2.waitKey(1) == ord("q"):
            break
