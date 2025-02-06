# GaussianAvatar-Editor: Photorealistic Animatable Gaussian Head Avatar Editor

### [Arxiv](https://arxiv.org/abs/2501.09978) | [Project Page](https://xiangyueliu.github.io/GaussianAvatar-Editor/) 
> [Xiangyue Liu](https://xiangyueliu.github.io/), [Kunming Luo](https://coolbeam.github.io/), [Heng Li](https://hengli.me/), [Qi Zhang](https://qzhang-cv.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Li Yi](https://ericyi.github.io/), [Ping Tan](https://ece.hkust.edu.hk/pingtan)<sup>†</sup>
>
> 3DV 2025
>
<div align=center>
<img src="assets/teaser_10M.gif" width="90%"/>
</div>


## Abstract
We introduce GaussianAvatar-Editor, an innovative framework for text-driven editing of animatable Gaussian head avatars that can be fully controlled in expression, pose, and viewpoint. Unlike static 3D Gaussian editing, editing animatable 4D Gaussian avatars presents challenges related to motion occlusion and spatial-temporal inconsistency. To address these issues, we propose the Weighted Alpha Blending Equation (WABE). This function enhances the blending weight of visible Gaussians while suppressing the influence on non-visible Gaussians, effectively handling motion occlusion during editing. Furthermore, to improve editing quality and ensure 4D consistency, we incorporate conditional adversarial learning into the editing process. This strategy helps to refine the edited results and maintain consistency throughout the animation. By integrating these methods, our GaussianAvatar-Editor achieves photorealistic and consistent results in animatable 4D Gaussian editing. We conduct comprehensive experiments across various subjects to validate the effectiveness of our proposed techniques, which demonstrates the superiority of our approach over existing methods. More results and code are available at: [https://xiangyueliu.github.io/GaussianAvatar-Editor/](https://xiangyueliu.github.io/GaussianAvatar-Editor/).

## Setup

### Environment

* Clone this repo
    ```shell
    git clone https://github.com/Lxiangyue/GaussianAvatar-Editor.git
    cd GaussianAvatar-Editor
    ```
* Install dependencies to setup a conda environment:
    ```shell
    conda create --name gsavatareditor -y python=3.10
    conda activate gsavatareditor
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    pip install -r requirements.txt
    ```
    We made our modifications on CUDA code in diff-gaussian-rasterization, so clone our version and install (compulsory): 
    ```shell
    cd submodules/diff-gaussian-rasterization && rm -r diff_gaussian_rasterization.egg-info && pip install . && cd ..
    cd nvdiffrast && rm -r nvdiffrast.egg-info && pip install . && cd ..
    cd simple-knn && rm -r simple-knn.egg-info && pip install . && cd ..
    ```
    

## Download

### Data
In our paper, we use part of the NeRSemble dataset. You can download the pre-processed data from
- [data](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xliufz_connect_ust_hk/EXgjFvCt3OZKnxbfi60aS5oBK3vIjYh8ybPz1bXPO9ZA4w?e=qGfxn9) (directly accessible).
- or from the official: [OneDrive](https://tumde-my.sharepoint.com/:f:/g/personal/shenhan_qian_tum_de/EtgO7DSNVzNKuYMRQeL4PE0BqMsTwdpQ09puewDLQBz87A) (request [here](https://forms.gle/dPEJx5DNvmhTm2Ry5)).

### Model

- Download the FLAME model: [flame_model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xliufz_connect_ust_hk/EfPdhVCFUstImv1gMtky8wgB0swp3bKFAnknP6_IyLFQKQ?e=cuohw1).

- Download Original Gaussian Avatars: [gs_origin](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xliufz_connect_ust_hk/EaPFenNhUwxIuPSyIzxBIPkBzmn_MIHlI61LAmQi5LlFAg?e=ewczZM).

- Download Our Trained Edited Gaussian Avatars Models (Optional, only for inference): [outputs](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xliufz_connect_ust_hk/EZlwAeZWRRlPr2JZwpUnXh4BfDV31shOa5K5y_NJ7PwdKQ?e=Kdfh3k).

* Path organized as:
    ```shell
    /GaussianAvatar-Editor
        /data
        /flame_model
        /gs_origin
        /outputs
    ```

## Reproducing Experiments
### Novel View Rendering
1. For example, using editing prompt *"Turn him into the Tolkien Elf"* on *306_EMO1*. 
<div align=center>
<img src="assets/ours_nv_01_7M.gif" width="60%"/>
</div>


* __Inference__ with our trained model: 
  ```shell
  python render.py -m outputs/306_EMO1_Elf  --select_camera_id 8
  # You could change the "select_camera_id" to one of the "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" to render the different view.
  ```
* __Train__:
  ```shell
  python train.py -s data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/306_EMO-1_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine  -m outputs/306_EMO1_Elf_train --port 60001 --eval --white_background --bind_to_mesh --iterations 601_000 --interval 500 --start_checkpoint gs_origin/UNION10EMOEXP_306_eval_600k/chkpnt600000.pth  --use_in2n  --text_prompt "Turn him into the Tolkien Elf" --different_data --use_grad_mask --use_gan  
  ```

2. For example, using editing prompt *"What would the human look like as a bearded man?"* on *104_EXP5*. 
<div align=center>
<img src="assets/ours_nv_02_4k_14M.gif" width="60%"/>
</div>


* __Inference__ with our trained model:
  ```shell
  python render.py -m outputs/104_EXP5_bearded  --select_camera_id 8
  ```
* __Train__:
  ```shell
  python train.py -s data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/104_EXP-5_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine  -m outputs/104_EXP5_bearded_train --port 60000 --eval --white_background --bind_to_mesh --iterations 601_000 --interval 500 --start_checkpoint gs_origin/UNION10EMOEXP_104_eval_600k/chkpnt600000.pth  --use_in2n  --text_prompt "What would she look like as a bearded man?" --different_data --use_grad_mask --use_gan 
  ```


### Self-reenactment
1. For example, using editing prompt *"Make it an Egyptian sculpture"* on *264_EXP5*. 
<div align=center>
<img src="assets/264_Egyptian_5M.gif" width="35%"/>
</div>


* __Inference__ with our trained model:
  ```shell
  python render.py -m outputs/264_EXP5_Egyptian  --select_camera_id 8
  ```
* __Train__:
  ```shell
  python train.py -s data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/264_EXP-5_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine  -m outputs/264_EXP5_Egyptian_train --port 60000 --eval --white_background --bind_to_mesh --iterations 601_000 --interval 500 --start_checkpoint gs_origin/UNION10EMOEXP_264_eval_600k/chkpnt600000.pth  --use_in2n  --text_prompt "Make it an Egyptian sculpture" --different_data --use_grad_mask --use_gan 
  ```

2. For example, using editing prompt *"The human should look 100 years old"* on *304_EXP2*. 
<div align=center>
<img src="assets/304_100_8M.gif" width="35%"/>
</div>

* __Inference__ with our trained model:
  ```shell
  python render.py -m outputs/304_EXP2_100  --select_camera_id 8
  ```
* __Train__:
  ```shell
  python train.py -s data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/304_EXP-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine  -m outputs/304_EXP2_100_train --port 60000 --eval --white_background --bind_to_mesh --iterations 601_000 --interval 500 --start_checkpoint gs_origin/UNION10EMOEXP_304_eval_600k/chkpnt600000.pth  --use_in2n  --text_prompt "The human should look 100 years old" --different_data --use_grad_mask --use_gan 
  ```

3. For example, using editing prompt *"Apply face paint"* on *304_EXP2*. 
<div align=center>
<img src="assets/304_facepaint_8M.gif" width="35%"/>
</div>

* __Inference__ with our trained model:
  ```shell
  python render.py -m outputs/304_EXP2_facepaint  --select_camera_id 8
  ```
* __Train__:
  ```shell
  python train.py -s data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/304_EXP-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine  -m outputs/304_EXP2_facepaint_train --port 60000 --eval --white_background --bind_to_mesh --iterations 601_000 --interval 500 --start_checkpoint gs_origin/UNION10EMOEXP_304_eval_600k/chkpnt600000.pth  --use_in2n  --text_prompt "Apply face paint" --different_data --use_grad_mask --use_gan 
  ```

### Cross-identy Reenactment 
* Please note that Cross-identy Reenactment does not require training.
<div align=center>
<img src="assets/ours_cross_2_10M.gif" width="60%"/>
</div>

1. For example, using the source actor 460_FREE to drive the trained edited avatars 306_EMO1_Elf to do the same actions. 
  ```shell
  python render.py -t /home/super/Desktop/8TDisk/Codes_GaussianAvatarEditor/GaussianAvatar-Editor/data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/460_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine -m outputs/306_EMO1_Elf --select_camera_id 8 
  ```

2. For example, using the source actor 460_FREE to drive the trained edited avatars 304_EXP2_100 to do the same actions. 
  ```shell
  python render.py -t /home/super/Desktop/8TDisk/Codes_GaussianAvatarEditor/GaussianAvatar-Editor/data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/460_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine -m outputs/304_EXP2_100 --select_camera_id 8 
  ```


## Citation
If you find our work useful in your research, please cite:
```
@article{liu2025gaussianavatar,
  title={GaussianAvatar-Editor: Photorealistic Animatable Gaussian Head Avatar Editor},
  author={Liu, Xiangyue and Luo, Kunming and Li, Heng and Zhang, Qi and Liu, Yuan and Yi, Li and Tan, Ping},
  journal={arXiv preprint arXiv:2501.09978},
  year={2025}
}
```

## Acknowledgements
The implementation of GaussianAvatar-Editor are based on [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars). Thanks to these authors for releasing the code.


