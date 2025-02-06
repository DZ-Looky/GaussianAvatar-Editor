#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import open3d as o3d
import copy

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set_cano(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh, cano_openmouth):

    if dataset.select_camera_id != -1:
        name = f"{name}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders_cano"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh_cano"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # fetch new Camera from json file
    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    
    # set flame_param to zero to get the neural pose and expression    
    # T = gaussians.num_timesteps
    T = 1
    device = gaussians.flame_param['expr'].device
    gaussians.flame_param['expr'] = torch.zeros([T, gaussians.flame_param['expr'].shape[1]]).to(device)
    gaussians.flame_param['rotation'] = torch.zeros([T, 3]).to(device)
    gaussians.flame_param['neck_pose'] = torch.zeros([T, 3]).to(device)
    gaussians.flame_param['jaw_pose'] = torch.zeros([T, 3]).to(device)
    gaussians.flame_param['eyes_pose'] = torch.zeros([T, 6]).to(device)
    gaussians.flame_param['translation'] = torch.zeros([T, 3]).to(device)
    gaussians.flame_param['dynamic_offset'] = torch.zeros([T, gaussians.flame_model.v_template.shape[0], 3]).to(device)
 
    if gaussians.binding != None: 
        gaussians.select_mesh_by_timestep(0)   # neural pose and expression only needs one timestep
        
    if cano_openmouth:  
        aa = torch.load('debug/ns_306_val_00991_flame_params.pth')
        val_idx = 974
        verts, verts_cano = gaussians.flame_model(
                gaussians.flame_param['shape'][None, ...],
                gaussians.flame_param['expr'],
                gaussians.flame_param['rotation'],
                gaussians.flame_param['neck_pose'],
                aa['jaw_pose'][val_idx][None, ...],
                gaussians.flame_param['eyes_pose'],
                gaussians.flame_param['translation'],
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=gaussians.flame_param['static_offset'],
                dynamic_offset=gaussians.flame_param['dynamic_offset'],
            )
        gaussians.update_mesh_properties(verts, verts_cano)

    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        if render_mesh:
  
            out_dict = mesh_renderer.render_from_camera(gaussians.verts_cano, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + rendering.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
            
            if idx == 0:
                import trimesh
                mesh_cano = trimesh.Trimesh(gaussians.verts_cano.squeeze().cpu().numpy(), gaussians.faces.cpu().numpy(), process=False)
                mesh_cano.export(iter_path / 'mesh_cano.obj')

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_cano.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh_cano.mp4")

    except Exception as e:
        print(e)


def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh, vis_teeth_mask):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):

        if gaussians.binding != None:  
            gaussians.select_mesh_by_timestep(view.timestep)
        
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        if idx == 0 or idx == 20 or idx == 42:
            import trimesh
            mesh_cano = trimesh.Trimesh(gaussians.verts.squeeze().cpu().numpy(), gaussians.faces.cpu().numpy(), process=False)
            mesh_cano.export(str(iter_path) + '/mesh_%d.obj'%idx)

        if render_mesh:
            
            if vis_teeth_mask:   
                masked_faces = gaussians.faces[gaussians.flame_model.teeth_mask]
                gaussians.faces = masked_faces
            
            plot_opacity_distrubute = False
            if plot_opacity_distrubute:
                x = gaussians._opacity.clone().detach().squeeze().cpu()
                min_val = torch.min(x)
                max_val = torch.max(x)
                hist = torch.histc(x, bins=100, min=min_val, max=max_val)
                bin_width = (max_val - min_val) / 100
                bin_centers = torch.linspace(min_val + bin_width / 2, max_val - bin_width / 2, 100)
                plt.bar(bin_centers, hist, width=bin_width)
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.savefig('debug/histogram_idx%02d.png'%idx)
                plt.clf()
                
                # thred = bin_width * 90.0 + min_val
                thred = 10
                thred_mask = x > thred  
                # print('thred', thred)
                
                aa = torch.unique(gaussians.binding * thred_mask.cuda()).long()
                face_mask = torch.zeros(len(gaussians.faces)).cuda()  # 10144
                face_mask.scatter_(0, aa, 1)
            
                masked_faces = gaussians.faces[face_mask.bool()]
                gaussians.faces = masked_faces
    
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.4
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + rendering * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
            
    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)


def render_flame_mask_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh, vis_teeth_mask):
    
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
        
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders_flame_mask" 
    gts_path = iter_path / "gt"   
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"  
        
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    
    mask_gs_dict = {}
    for label in ['teeth', 'lips_tight', 'lip_inside']:
        faces_id = gaussians.flame_model.mask.f.get_buffer(label).clone().detach()
        faces_id_ = torch.unique(faces_id).cuda()
        mask_gs = torch.zeros(len(gaussians.binding)).cuda()  
        for i in range(len(gaussians.binding)):
            if gaussians.binding[i] in faces_id_:
                mask_gs[i] = 1
        mask_gs = mask_gs.bool()    
        mask_gs_dict[label] = mask_gs

    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        
        if gaussians.binding != None:  # True 
            gaussians.select_mesh_by_timestep(view.timestep)
        
        time_idx = int(view.image_path.split('/')[-1].split('.')[0].split('_')[0])
        cam_idx = int(view.image_path.split('/')[-1].split('.')[0].split('_')[1])
        
        

        
        verts = gaussians.verts.squeeze()
        verts_id = gaussians.flame_model.mask.v.get_buffer('teeth').clone().detach()
        vert_mask = torch.ones(len(gaussians.verts.squeeze())).cuda()  
        vert_mask[verts_id] = 0.
        rgb = vert_mask[...,None].repeat(1,3)   
        point_cloud = Pointclouds(points=[verts], features=[rgb])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
        o3d.io.write_point_cloud("debug/point_cloud_verts.ply", pcd)
        
        R, T = torch.Tensor(view.R)[None], torch.Tensor(view.T)[None]
        R[:,:,1:3]=-R[:,:,1:3]
        cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=((2048.7333984375, 2048.7451171875),), principal_point=((400.8660583496094, 274.87823486328125),), image_size=((802, 550),), in_ndc=False)[0]

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
        raster_settings = PointsRasterizationSettings(
            image_size=(802, 550), 
            radius = 0.03,
            points_per_pixel = 10
        )

        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        images = renderer(point_cloud)
        # plt.imsave("debug/pytorch_3d_image_new.png", images[0, ..., :3].clamp(0,1).cpu().numpy())
        # plt.axis("off")
        
        for label, mask in mask_gs_dict.items():
          
            gaussians_cp = copy.deepcopy(gaussians)
            
            import pdb; pdb.set_trace()
            
            gaussians_cp._xyz[~mask] = torch.Tensor([-1000000000, -1000000000, -1000000000,]).cuda()
            rendering_k = render(view, gaussians_cp, pipeline, background)["render"]
            array = rendering_k.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((array * 255).astype('uint8'))
            image.save('debug/rendering_%s_%s_%s.png'%(label, time_idx, cam_idx))
            
            gt = view.original_image[0:3, :, :]
            array = gt.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((array * 255).astype('uint8'))
            image.save('debug/rendering_%s_%s_%s_gt.png'%(label, time_idx, cam_idx))
            
        import pdb; pdb.set_trace()
       
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        if render_mesh:
            
            if vis_teeth_mask:   # only vis teeth
                masked_faces = gaussians.faces[gaussians.flame_model.teeth_mask]
                gaussians.faces = masked_faces
            
            plot_opacity_distrubute = False
            if plot_opacity_distrubute:
                x = gaussians._opacity.clone().detach().squeeze().cpu()
                min_val = torch.min(x)
                max_val = torch.max(x)
                hist = torch.histc(x, bins=100, min=min_val, max=max_val)
                bin_width = (max_val - min_val) / 100
                bin_centers = torch.linspace(min_val + bin_width / 2, max_val - bin_width / 2, 100)
                plt.bar(bin_centers, hist, width=bin_width)
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.savefig('debug/histogram_idx%02d.png'%idx)
                plt.clf()
                
                thred = 10
                thred_mask = x > thred   
                print('thred', thred)
                
                aa = torch.unique(gaussians.binding * thred_mask.cuda()).long()
                face_mask = torch.zeros(len(gaussians.faces)).cuda()  
                face_mask.scatter_(0, aa, 1)
            
                masked_faces = gaussians.faces[face_mask.bool()]
                gaussians.faces = masked_faces
          
            
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
            
            if plot_opacity_distrubute:
                aa = rendering_mesh.clone().detach().cpu().permute(1,2,0).clamp(0,1) * 255 
                aa = aa.numpy().astype('uint8')
                plt.imsave('debug/aa_06f.png'%(bin_width * 50.0 + min_val), aa)
            
        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
            
    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool, render_cano: bool, cano_openmouth: bool, vis_teeth_mask: bool, render_flame_mask: bool):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, render_cano=render_cano)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, vis_teeth_mask)
        elif render_cano:
            render_set_cano(dataset, "cano", scene.loaded_iter, scene.getCanoCameras(), gaussians, pipeline, background, render_mesh, cano_openmouth)
        elif render_flame_mask:
            render_flame_mask_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, vis_teeth_mask)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, vis_teeth_mask)
            
            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh, vis_teeth_mask)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh, vis_teeth_mask)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    # lxy
    parser.add_argument("--render_cano", action="store_true")    
    parser.add_argument("--cano_openmouth", action="store_true")    
    parser.add_argument("--vis_teeth_mask", action="store_true")     
    parser.add_argument("--render_flame_mask", action="store_true")   
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh, args.render_cano, args.cano_openmouth, args.vis_teeth_mask, args.render_flame_mask)
    
