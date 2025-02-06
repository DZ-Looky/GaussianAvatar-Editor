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

import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch
from torchvision import transforms 
loader = transforms.Compose([transforms.ToTensor()])  
unloader = transforms.ToPILImage()
import numpy as np
from utils.pytorch3d_load_obj import write_ply
from PIL import Image, ImageDraw
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from torch_scatter import scatter_mean




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, different_data, fix_gs_number, using_mask_continue, use_in2n, text_prompt, use_grad_mask, use_mouth_loss, use_mouth_mask, filter_close_mouth_mask, source_path, use_grad_mask_alphaT, use_in2nssim, use_gan, use_densify):
    
    if using_mask_continue:
        print('only update the editable region in this step training')
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params, use_in2n=use_in2n, text_prompt=text_prompt, use_mouth_loss=use_mouth_loss, filter_close_mouth_mask=filter_close_mouth_mask, source_path=source_path, use_gan=use_gan)
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    
    if use_gan:
        gaussians.use_gan = True
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, different_data)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    
    if use_grad_mask:
        teeth_mask_faceID = (gaussians.flame_model.teeth_mask)  
        teeth_filter = []
        for item in gaussians.binding: 
            teeth_filter.append(teeth_mask_faceID[item].item())
        teeth_filter = torch.Tensor(teeth_filter).cuda().bool()
        teeth_filter = ~teeth_filter
    
    labels = ['teeth', 'lips_tight', 'lip_inside']
    mask_face_ids = []
    for label in labels: mask_face_ids.append(gaussians.flame_model.mask.f.get_buffer(label).clone().detach().tolist())
    new_list = [elem for sublist in mask_face_ids for elem in sublist]
    mouth_mask_face_ids = (torch.unique(torch.Tensor(new_list))).int().cuda()
    gaussians.mouth_mask_face_ids = mouth_mask_face_ids
    
    mouth_mask_gs = torch.zeros(len(gaussians.binding)).cuda()  
    for i in range(len(gaussians.binding)):
        if gaussians.binding[i] in mouth_mask_face_ids:
            mouth_mask_gs[i] = 1
    mouth_mask_gs = mouth_mask_gs.bool()    
        
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:  # Flase, not into this choose            
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree  # gaussians.max_sh_degree 3
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:  
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)    # TODO 

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, opacity_filter, gs_render_id, gs_render_alphaT = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity_filter"], render_pkg["gs_render_id"], render_pkg["gs_render_alphaT"]  # [3,802,550], [85958,3], [85958], [85958], [85958], [10, 802, 550], [10, 802, 550]
        
        assert use_grad_mask + use_grad_mask_alphaT != 2, "ERROR! ONLY USING ONE"
        if use_grad_mask:
    
                # using former gs of each pixel
                former_10_gs_id = torch.unique(gs_render_id.clone().detach())[1:]  
                assert former_10_gs_id.min() >= 0, "Illegal Gaussian id!!!!"
                gs_filter = torch.zeros_like(render_pkg["opacity_filter"])    
                
                for idx in range(len(former_10_gs_id)): 
                    gs_filter[former_10_gs_id[idx]] = True
            
                mouth_mask_gs_opt = ~mouth_mask_gs
                gs_filter_ = mouth_mask_gs_opt & gs_filter
                gs_filter = gs_filter_.clone().detach()
             
                gaussians.set_mask(gs_filter)
                gaussians.apply_grad_mask(gs_filter)

        elif use_grad_mask_alphaT:

            gs_render_alphaT_norm = (gs_render_alphaT - gs_render_alphaT.min(0)[0]) / (gs_render_alphaT.max(0)[0] - gs_render_alphaT.min(0)[0] + 1e-10)   
            output_gs = torch.zeros(len(render_pkg["opacity_filter"])).cuda().flatten()
            gs_render_alphaT_norm  = gs_render_alphaT_norm[gs_render_id != -1]
            gs_render_id= gs_render_id[gs_render_id != -1]
            out_w = scatter_mean(gs_render_alphaT_norm.clone().detach().flatten(), index=gs_render_id.clone().detach().type(torch.int64).flatten(), out=output_gs)
            
            if len(mouth_mask_gs) != len(out_w):  # new gs added
                mouth_mask_face_ids = gaussians.mouth_mask_face_ids.clone().detach()
                mouth_mask_gs = torch.zeros(len(gaussians.binding)).cuda() 
                for i in range(len(gaussians.binding)):
                    if gaussians.binding[i] in mouth_mask_face_ids:
                        mouth_mask_gs[i] = 1
                mouth_mask_gs = mouth_mask_gs.bool()    
            mouth_mask_gs_opt = ~mouth_mask_gs
            out_w = mouth_mask_gs_opt * out_w
            
            gaussians.set_mask(out_w)
            gaussians.apply_grad_mask(out_w)

        else:
            pass

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
    
        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)  
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim  
        
        if using_mask_continue:
            assert len(visibility_filter) == len(gaussians.gs_edit_filter), "ERROR!!! CHECK!!!"
            visibility_filter = gaussians.gs_edit_filter.to(visibility_filter.device)

        if gaussians.binding != None:  
            if opt.metric_xyz: 
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else: 
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz  

            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else: 
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale  

            if opt.lambda_dynamic_offset != 0:  
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:  
                ti = viewpoint_cam.timestep 
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:   
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
                
        if use_in2n:
            
            padded_tensor = F.pad(image.permute(1,2,0), (0,0,(802 - 550) // 2, (802 - 550) // 2), value=1.).permute(2, 0, 1)  
            image_resized = F.interpolate(padded_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True)       # rendered 
            
            padded_tensor_gt = F.pad(gt_image.permute(1,2,0), (0,0,(802 - 550) // 2, (802 - 550) // 2), value=1.).permute(2, 0, 1)   
            gt_image_resized = F.interpolate(padded_tensor_gt.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True)   # origin
            
            result = gaussians.ip2p(image_resized.permute(0,2,3,1), gt_image_resized.permute(0,2,3,1), gaussians.prompt_utils,)   
            
            view_index = viewpoint_cam.timestep
            gaussians.edit_frames[view_index] = result["edit_images"].detach().clone() 

            lambda_l1 = 10.
            lambda_p = 10.
            losses['in2n_l1'] = lambda_l1 * torch.nn.functional.l1_loss(image_resized.permute(0,2,3,1),  gaussians.edit_frames[view_index])  
            
            if use_in2nssim:
                if iteration > round(0.9 * (opt.iterations - first_iter)):
                    losses['in2n_ssim'] = lambda_p * (1.0 - ssim(image_resized.contiguous().squeeze(), gaussians.edit_frames[view_index].permute(0, 3, 1, 2).contiguous().squeeze()))   
                else:  
                    losses['in2n_perceptual'] = lambda_p * gaussians.perceptual_loss(image_resized.contiguous(), 
                                                                                    gaussians.edit_frames[view_index].permute(0, 3, 1, 2).contiguous(), ).sum()           
            else:
                losses['in2n_perceptual'] = lambda_p * gaussians.perceptual_loss(image_resized.contiguous(), 
                                                                                    gaussians.edit_frames[view_index].permute(0, 3, 1, 2).contiguous(), ).sum()           
            
            # del losses with origin imgs
            del losses['l1']
            del losses['ssim']
            
        motionID = viewpoint_cam.image_path.split('/')[-3].split('_')[1]
        time_idx = int(viewpoint_cam.image_path.split('/')[-1].split('_')[0])
        cam_idx = int(viewpoint_cam.image_path.split('/')[-1].split('_')[-1].split('.')[0])
        mouth_mask = gaussians.mouth_langsam_mask_dicts[motionID][time_idx, cam_idx,].cuda()
        losses['l1_mouth'] = l1_loss(image * mouth_mask, gt_image * mouth_mask) * 1000    
        losses['ssim_mouth'] = (1.0 - ssim(image * mouth_mask, gt_image * mouth_mask)) * 1000 
            
        if use_gan:
            if iteration > round(5/6 * (opt.iterations - first_iter)):  
                gt_rgb = gaussians.edit_frames[view_index].permute(0, 3, 1, 2)  
                pred_rgb = image_resized
                input_real = torch.cat([gt_rgb, gt_rgb - gt_rgb], dim=1)   
                input_fake = torch.cat([gt_rgb, pred_rgb - gt_rgb], dim=1)
                
                pred_real = gaussians.discriminator(input_real)
                pred_fake = gaussians.discriminator(input_fake)
                
                discrim_loss = gaussians.discriminator_loss(pred_real, pred_fake) 
                generator_loss = gaussians.generator_loss(pred_fake)  
                
                gan_loss = 1e-2 * discrim_loss + 1e-2 * generator_loss  
                losses['gan_loss'] = gan_loss
     
        losses['total'] = sum([v for k, v in losses.items()])  # ['l1', 'ssim', 'xyz', 'scale'] 
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                if 'in2n_l1' in losses:
                    postfix["in2n_l1"] = f"{losses['in2n_l1']:.{7}f}"
                if 'in2n_perceptual' in losses:
                    postfix["in2n_perceptual"] = f"{losses['in2n_perceptual']:.{7}f}"
                if 'l1_mouth' in losses:
                    postfix["l1_mouth"] = f"{losses['l1_mouth']:.{7}f}"
                if "ssim_mouth" in losses:
                    postfix["ssim_mouth"] = f"{losses['ssim_mouth']:.{7}f}"
                if 'gan_loss' in losses:
                    postfix["gan_loss"] = f"{losses['gan_loss']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:  
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations): 
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if not fix_gs_number:   
                # Densification
                if iteration < opt.densify_until_iter: 
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])  # visibility_filter 
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  
                        print("Gaussian densify before", )
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None 
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)  
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):  
                        gaussians.reset_opacity()
                
            # Optimizer step
            if iteration < opt.iterations:  
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) or iteration == 10:  
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:        
        if 'l1_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        if 'ssim_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        if 'in2n_perceptual' in losses:
            tb_writer.add_scalar('train_loss_patches/in2n_perceptual', losses['in2n_perceptual'].item(), iteration)
        if 'in2n_l1' in losses:
            tb_writer.add_scalar('train_loss_patches/in2n_l1', losses['in2n_l1'].item(), iteration)
        if "l1_mouth" in losses:
            tb_writer.add_scalar('train_loss_patches/l1_mouth', losses['l1_mouth'].item(), iteration)
        if "ssim_mouth" in losses:
            tb_writer.add_scalar('train_loss_patches/ssim_mouth', losses['ssim_mouth'].item(), iteration)
        if 'gan_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/gan_loss', losses['gan_loss'].item(), iteration)
            
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if hasattr(scene.gaussians, "select_mesh_by_timestep"):
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
    
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def training_cano(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, different_data, edit_mask_dir):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    import torch
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, different_data)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # load cano space camera
    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
               
    if os.path.exists(scene.model_path + "/gs_edit_filter_init.pth"):
        if gaussians.gs_edit_filter is None:   # only for training edit from begin, other will load from ckpt
            gaussians.gs_edit_filter = torch.load(scene.model_path + "/gs_edit_filter_init.pth")
            gaussians.gs_edit_face_filter = torch.load(scene.model_path + "/gs_edit_face_filter_init.pth")
        
    else: # load all img mask to get the gs_edit_filter_init.pth
        if edit_mask_dir is not None:
            print('loading the editing mask of the image')
            edit_mask = []
            for j, mask_path in enumerate(sorted(glob.glob(edit_mask_dir + '/*.*'))):
                mask_pth = torch.load(mask_path)   
                edit_mask.append(mask_pth)
            edit_mask = torch.stack(edit_mask)  
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:              
            try:
                # receive data
                net_image = None
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree  # gaussians.max_sh_degree 3
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None: 
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)    # TODO 

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]  # [3,802,550], [85958,3], [85958], [85958]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
 
        # cumulating Gaussians's gs_filter until all views is done  
        if not os.path.exists(scene.model_path + "/gs_edit_filter_init.pth"):
            with torch.no_grad():
                image_name = viewpoint_cam.image_name    
                if image_name not in gaussians.gs_edit_filter_views:
                    gaussians.gs_edit_filter_views.append(image_name)
            
                    out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, viewpoint_cam)

                    rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                    rgb_mesh = rgba_mesh[:3, :, :]
                    alpha_mesh = rgba_mesh[3:, :, :]
                    mesh_opacity = 0.5
                    
                    rendering = image.clone().detach()
                    rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + rendering.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                    
                    # save img
                    if not os.path.isdir('debug'): os.makedirs('debug')
                    unloader(image.clone().detach().cpu().squeeze(0)).save('debug/render.png')
                    unloader(rendering_mesh.clone().detach().cpu().squeeze(0)).save('debug/render_mesh.png')
                    unloader(gt_image.clone().detach().cpu().squeeze(0)).save('debug/gt.png')
                    
                    # reproject mesh verts to img
                    xyz_cam = out_dict['verts_camera'].squeeze()  
                    
                    cx = viewpoint_cam.image_width / 2.
                    cy = viewpoint_cam.image_height / 2.
                    fx = viewpoint_cam.image_width / (2. * np.tan(viewpoint_cam.FoVx / 2))
                    fy = viewpoint_cam.image_height / (2. * np.tan(viewpoint_cam.FoVy / 2))
                    K = np.array([[-fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
                    xy_hom = torch.bmm(torch.Tensor(K[None]).repeat(xyz_cam.shape[0],1,1).to(xyz_cam.device), xyz_cam[..., None]).squeeze()
                    xy = (xy_hom[:, :2] / xy_hom[:, 2][...,None].repeat(1,2)).round().long()      
                                
                    image_2d = Image.open('debug/gt.png').convert("RGB")
                    draw = ImageDraw.Draw(image_2d)
                    for point in xy: 
                        draw.rectangle([tuple(point), tuple(point)], fill="red")
                    image_2d.save('debug/render_means2d.png')          
                    
                    if edit_mask_dir is not None:
                        img_mask = edit_mask[int(image_name)]  
                    else:
                        import pdb; pdb.set_trace()  
                    
                    edit_filter = []
                    for point in xy:
                        try:
                            edit_filter.append(img_mask[point[1], point[0]])
                        except:
                            edit_filter.append(torch.tensor(False))   
                    edit_filter = torch.stack(edit_filter)   
                
                    image_2d = Image.open('debug/gt.png').convert("RGB")
                    draw = ImageDraw.Draw(image_2d)
                    for idx in range(xy.shape[0]):
                        if edit_filter[idx]:
                            point = xy[idx]
                            draw.rectangle([tuple(point), tuple(point)], fill="green")
                    image_2d.save('debug/render_edit_filter_%s.png'%image_name)   

                    face_filter = []
                    for face in gaussians.faces:
                        if edit_filter[face[0]] or edit_filter[face[1]] or edit_filter[face[2]]:
                            face_filter.append(True)
                        else:
                            face_filter.append(False)
                    face_filter = torch.tensor(face_filter)    
                    
                    if gaussians.gs_edit_face_filter is None:
                        gaussians.gs_edit_face_filter = face_filter
                    else:
                        gaussians.gs_edit_face_filter += face_filter
                        print('using the union faces', gaussians.gs_edit_face_filter.int().sum().item(), 'percent:', (gaussians.gs_edit_face_filter.int().sum()/gaussians.gs_edit_face_filter.shape[0]).item())
                    
                    gs_filter = []
                    for bindingid in gaussians.binding:
                        if face_filter[bindingid]:
                            gs_filter.append(True)
                        else:
                            gs_filter.append(False)
                    gs_filter = torch.tensor(gs_filter)
                            
                    if gaussians.gs_edit_filter is None: 
                        gaussians.gs_edit_filter = gs_filter   
                    else:
                        gaussians.gs_edit_filter += gs_filter 
                        print('using the union region from multi views', gaussians.gs_edit_filter.int().sum().item(), 'percent:', (gaussians.gs_edit_filter.int().sum()/gaussians.gs_edit_filter.shape[0]).item())
                        
                else:
                    print('all view include, saving to %s'%(scene.model_path + "/gs_edit_filter_init.pth"))    
                    torch.save(gaussians.gs_edit_filter, scene.model_path + "/gs_edit_filter_init.pth")
                    torch.save(gaussians.gs_edit_face_filter, scene.model_path + "/gs_edit_face_filter_init.pth")
                    
        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)  
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim  
        
        # to double check if this can only update the gs_edit_filter's Gaussians   
        assert len(visibility_filter) == len(gaussians.gs_edit_filter), "ERROR HAPPENED!!! CHECK!!!"
        visibility_filter = gaussians.gs_edit_filter.to(visibility_filter.device)
        
        if gaussians.binding != None:  
            if opt.metric_xyz: 
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else: 
                
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz 
        
            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else: 
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale  

            if opt.lambda_dynamic_offset != 0:  
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:  
                ti = viewpoint_cam.timestep 
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:  
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
            
        losses['total'] = sum([v for k, v in losses.items()])  
        losses['total'].backward()

        import pdb; pdb.set_trace()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:  
                progress_bar.close()

            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations): 
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter: 
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])  
                
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % opt.densification_interval == 0 or iteration==600001:  
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None 
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)  
 
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):  
                    gaussians.reset_opacity()
            
            print('Gaussian0000:, ', gaussians._xyz.shape)

            # Optimizer step
            if iteration < opt.iterations: 
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) or iteration == 10:  # [10k, 20k, ..., 660k]
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--different_data", action='store_true', default=False, help="If using different training datas of start_checkpoint and now")  
    parser.add_argument("--training_cano_only", action='store_true', default=False, help="If only training neural cano space to do editing")  
    parser.add_argument("--edit_mask_dir", type=str, default = None, help="If none, then editing the whole Gaussians, if not none, then put the img mask path")   
    parser.add_argument("--fix_gs_number", action='store_true', default=False, help="If using then no more new Gaussians")  
    parser.add_argument("--using_mask_continue", action='store_true', default=False, help="If using then no more new Gaussians in training new exp+pose data")  
    parser.add_argument('--use_gan', action='store_true', default=False, help="adding GAN loss between rendered imgs with 2d edited imgs")  
    parser.add_argument("--use_in2n", action='store_true', default=False, help="Weather add the in2n loss")    
    parser.add_argument('--text_prompt', type=str, default="")    
    parser.add_argument("--use_grad_mask", action='store_true', default=False, help="Weather use grad mask")     
    parser.add_argument("--use_mouth_loss", action='store_true', default=False, help="Weather use L1 loss in mouth region")     
    parser.add_argument("--use_mouth_mask", action='store_true', default=False, help="Weather use grad mask on teeth, lips_tight, lip_inside")    
    parser.add_argument("--filter_close_mouth_mask", action='store_true', default=False, help="Weather to filter close mouth langSAM mask")     
    parser.add_argument("--use_grad_mask_alphaT", action='store_true', default=False, help="Weather use grad mask")    
    parser.add_argument("--use_in2nssim", action='store_true', default=False, help="use ssim instead perctual loss in in2n")     
    parser.add_argument("--use_densify", action='store_true', default=False, help="use gs densify")    
    
    args = parser.parse_args(sys.argv[1:])
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations, args.interval)) + [args.iterations])
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations, args.interval)) + [args.iterations])
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations, args.interval)) + [args.iterations])

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    if args.training_cano_only:
        training_cano(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.different_data, args.edit_mask_dir)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.different_data, args.fix_gs_number, args.using_mask_continue, args.use_in2n, args.text_prompt, args.use_grad_mask, args.use_mouth_loss, args.use_mouth_mask, args.filter_close_mouth_mask, args.source_path, args.use_grad_mask_alphaT, args.use_in2nssim, args.use_gan, args.use_densify)
        
    print("\nTraining complete.")


