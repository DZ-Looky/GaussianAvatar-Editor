# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.thmodel import Discriminator

from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
import glob

import os
def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

def get_device():
    return torch.device(f"cuda:{get_rank()}")


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100, use_in2n=False, text_prompt="", use_mouth_loss=False, filter_close_mouth_mask=False, source_path="", use_gan=False):
        super().__init__(sh_degree)
    
        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr
        
        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None
        
        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()
            
        self.gs_edit_filter = None    # lxy
        self.gs_edit_filter_views = []   # lxy
        self.gs_edit_face_filter = None   # lxy
        
        # # lxy  # 20240509
        # if use_gan:
        #     self.discriminator = Discriminator()
        #     self.discriminator.requires_grad_(True)
        
        self.prompt_utils = None
        self.ip2p = None
        self.edit_frames = {}
        self.mouth_langsam_mask_dicts = None
        self.use_gan = False
        self.mouth_mask_face_ids = None
        
        
        if use_in2n:
            assert text_prompt != "", 'ERROR!!!'
            # tmp1
            self.prompt_utils = StableDiffusionPromptProcessor(
                {
                    # "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                    # "pretrained_model_name_or_path": "/apdcephfs/share_1330077/emmafxliu/huggingface/stable-diffusion-2-1",
                    "pretrained_model_name_or_path": "/apdcephfs/share_1330077/emmafxliu/huggingface/stable-diffusion-v1-5",
                    # "pretrained_model_name_or_path": "/home/super/Desktop/8TDisk/huggingface/CompVis/stable-diffusion-v1-5",
                    "prompt": text_prompt,
                }
            )()
            
   
            from threestudio.models.guidance.instructpix2pix_guidance import (
                InstructPix2PixGuidance,
            )

            from omegaconf import OmegaConf

            self.ip2p = InstructPix2PixGuidance(
                OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98})
            )
            # cur_2D_guidance = self.ip2p
            print("using InstructPix2Pix!")
            
            from threestudio.utils.perceptual import PerceptualLoss
            self.perceptual_loss = PerceptualLoss().eval().to(get_device())
            
        # load from mask data, fast
        try:
            motion_list = []
            dicts_dir = None
            if "UNION10" in source_path:
                motion_list = ['EMO-1', 'EMO-2', 'EMO-3', 'EMO-4', 'EXP-2', 'EXP-3', 'EXP-4', 'EXP-5', 'EXP-8', 'EXP-9']
                ID = source_path.split('/')[-1].split('_')[1]
                # dicts_dir = 'utils_dict/%s_UNION10_mouth_mask.pth'%ID
                dicts_dir = 'gs_origin/UNION10EMOEXP_%s_eval_600k/%s_UNION10_mouth_mask.pth'%(ID, ID)
            else:
                motionID = source_path.split('/')[-1].split('_')[1]
                motion_list.append(motionID)
                ID = source_path.split('/')[-1].split('_')[0]
                # dicts_dir = 'utils_dict/%s_%s_mouth_mask.pth'%(ID, motionID)
                dicts_dir = 'gs_origin/UNION10EMOEXP_%s_eval_600k/%s_%s_mouth_mask.pth'%(ID, ID, motionID)
            
            if os.path.exists(dicts_dir):
                self.mouth_langsam_mask_dicts = torch.load(dicts_dir)
            else:
                mouth_langsam_mask_dicts = {}
                for motionID in motion_list:
                    langsam_masks_dir = '/apdcephfs/share_1330077/emmafxliu/Codes_langSAM/lang-segment-anything/outputs/%s_%s_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine_mouth/mouth/mask'%(ID, motionID)
                    
                    mouth_langsam_masks = []
                    for i, mask_pth_path in enumerate(sorted(glob.glob(langsam_masks_dir + '/*.*'))):
                        print(mask_pth_path)
                        mouth_langsam_masks.append(torch.load(mask_pth_path))
                    
                    last_pth_path = sorted(glob.glob(langsam_masks_dir + '/*.*'))[-1]
                    total_t = int(last_pth_path.split('/')[-1].split('.')[0].split('_')[0]) + 1
                    total_cam = int(last_pth_path.split('/')[-1].split('.')[0].split('_')[1]) + 1
                    mouth_langsam_masks_ = torch.stack(mouth_langsam_masks)    # [1168, 802, 550]]
                    mouth_langSAM_masks = mouth_langsam_masks_.reshape(total_t, total_cam, mouth_langsam_masks_.shape[-2], mouth_langsam_masks_.shape[-1])
                    
                    mouth_langsam_mask_dicts[motionID] = mouth_langSAM_masks
                    
                self.mouth_langsam_mask_dicts = mouth_langsam_mask_dicts
                torch.save(mouth_langsam_mask_dicts, dicts_dir)
        except:
            pass   # for inference

        if use_gan:
            from utils.thmodel import Discriminator, DiscriminatorLoss, GeneratorLoss
            
            self.discriminator = Discriminator().cuda()
            self.discriminator_loss = DiscriminatorLoss().cuda()
            self.generator_loss = GeneratorLoss().cuda()
            

    
    
    def set_mask(self, mask):
        self.mask = mask    
            
    def apply_grad_mask(self, mask):  # if mask =  True, then optimize
        assert len(mask) == self._xyz.shape[0]
        # self.set_mask(mask)

        def hook(grad):
            final_grad = grad * (
                self.mask[:, None] if grad.ndim == 2 else self.mask[:, None, None]
            )
            # final_grad = grad * (
            #     torch.sigmoid(self.mask)[:, None] if grad.ndim == 2 else torch.sigmoid(self.mask)[:, None, None]
            # )
            # print(final_grad.abs().max())
            # print(final_grad.abs().mean())
            return final_grad

        fields = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling"]

        self.hooks = []

        for field in fields:
            this_field = getattr(self, field)    # [87569, 3]...
            assert this_field.is_leaf and this_field.requires_grad
            self.hooks.append(this_field.register_hook(hook))
            
        
            
            
                

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            # print('*******', len(meshes), len(tgt_meshes))
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            # self.num_timesteps = 1119  # VERY HARD CODING!!! LXY # have revised by add --different_data
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param   # self.flame_param    # TODO 

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
        if False:  # ONLY FOR DEBUG
            # lxy # get the canonical space mesh, and posed mesh
            # triangles_cano = verts_cano[:, faces]   # [1, 10144, 3, 3]
            import trimesh
            mesh_cano = trimesh.Trimesh(verts_cano.squeeze().cpu().numpy(), faces.cpu().numpy(), process=False)
            mesh_cano.export('debug/mesh_cano.obj')
            mesh = trimesh.Trimesh(verts.squeeze().cpu().numpy(), faces.cpu().numpy(), process=False)
            mesh.export('debug/mesh.obj')
            print('save canonical space and posed mesh done, by lxy')
            import pdb; pdb.set_trace()
            
    
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        # if self.use_gan:
        #     self.flame_param['discriminator'].requires_grad = True
        #     param_discriminator = {'params': [self.flame_param['discriminator']], 'lr': 1e-5, "name": "discriminator"}
        #     self.optimizer.add_param_group(param_discriminator)

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param['translation'] = flame_param['translation']
            self.flame_param['rotation'] = flame_param['rotation']
            self.flame_param['neck_pose'] = flame_param['neck_pose']
            self.flame_param['jaw_pose'] = flame_param['jaw_pose']
            self.flame_param['eyes_pose'] = flame_param['eyes_pose']
            self.flame_param['expr'] = flame_param['expr']
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
