import sys
sys.path.append(".")

from jackson_import import *
from mdm.utils.fixseed import fixseed
import os
import argparse
import json
import numpy as np
import torch
# from gthmr.emp_train.training_loop import TrainLoop
from mdm.utils.parser_util import generate_args, train_args, train_emp_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from gthmr.emp_train.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.utils.model_util import create_emp_model_and_diffusion
import shutil
from mdm.data_loaders.tensors import collate
from gthmr.lib.utils.mdm_utils import viz_motions
from VIBE.lib.dataset.vibe_dataset import rotate_about_D
from gthmr.emp_train.get_data import get_dataset_loader_dict_new2
from gthmr.lib.utils import data_utils
import ipdb
import yaml
from utils.misc import updata_ns_by_missing_keys
from mdm.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_aa, axis_angle_to_6d, rotation_6d_to_matrix
import generative_infill.asset_library as asset_library
import generative_infill.dataset_gen_summary_statistics as dataset_gen
from generative_infill.reloader import reload
from generative_infill.to_meshes import to_meshes
import tqdm
import scipy.ndimage
import pickle 

sys.path.append("/home/zhiyin/tml-fencing")
from src.render import glue, render_smpl_frame

#########################
# Load pretrained MR-DM #
#########################
extra_parser = argparse.ArgumentParser(add_help=False)
extra_parser.add_argument('--input_motion_path', type=str,
                            default='/home/zhiyin/tml-fencing/OUTPUT.npy',
                            help="Path to the input motion numpy file.")
extra_parser.add_argument('--mask_path', type=str,
                            default='/home/zhiyin/tml-fencing/MASK.npy',
                            help="Path to the input mask numpy file.")
extra_parser.add_argument('--output_dir', type=str,
                            default='/home/zhiyin/iterative-editing-release/',
                            help="Directory to save the output motion.")
extra_parser.add_argument('--num_fill', type=int,
                            default=20,
                            help="Number of fills for inpainting.")
extra_args, remaining_argv = extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv
args = train_emp_args()
args.input_motion_path = extra_args.input_motion_path
args.mask_path = extra_args.mask_path
args.output_dir = extra_args.output_dir
args.num_fill = extra_args.num_fill
fixseed(args.seed  + 1)
out_path = args.output_dir
name = os.path.basename(os.path.dirname(args.model_path))
niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
fps = 30

dist_util.setup_dist(args.device)
if out_path == '':
    out_path = os.path.join(os.path.dirname(args.model_path),
                            'edit_{}_{}_{}_seed{}_numfill{}'.format(name, niter, "in_between", args.seed, args.num_fill))
else:
    out_path = os.path.join(out_path, 'edit_{}_{}_{}_seed{}_numfill{}'.format(name, niter, "in_between", args.seed, args.num_fill))

device = 'cuda'
path_model_args = os.path.join(os.path.dirname(args.model_path), "args.json")
if not os.path.exists(path_model_args):
    raise ValueError(f"Model path [{args.model_path}] must be in the same" \
                    "directory as its model args file: [args.json]")
with open(path_model_args, 'r') as f:
    args_pretrained_model = argparse.Namespace(**json.load(f))

args_pretrained_model.total_batch_size = 1

# Overwrite save_dir
args_pretrained_model.save_dir = args.save_dir

args_pretrained_model.dataset = "amass_hml_keyframe"

# Backward comp
args_pretrained_model = updata_ns_by_missing_keys(args_pretrained_model, args)

# Load Pretrained Model
print("creating model and diffusion...")
model, diffusion = create_emp_model_and_diffusion(args_pretrained_model, None)
model.to(device)
model.rot2xyz.smpl_model.eval()

print(f"Loading checkpoints from [{args.model_path}]...")
state_dict = torch.load(args.model_path)
load_model_wo_clip(model, state_dict)
start_motion = torch.tensor(np.load(args.input_motion_path))#[..., :70]

num_samples =  1
max_frames = start_motion.shape[-1]
print("Input motion shape: ", start_motion.shape, "with max frames: ", max_frames)
args.batch_size = num_samples
data = get_dataset_loader(name=args_pretrained_model.dataset,
                                  batch_size=num_samples,
                                  num_frames=max_frames,
                                  data_rep = args_pretrained_model.data_rep,
                                  split='db',
                                  hml_mode='train',
                                  shuffle=False
                                  )

iterator = iter(data)
sample, model_kwargs = next(iterator) # never use input_motions 

start_motion_clone = start_motion.clone()
start_motion = reload(model, start_motion)

model_kwargs["y"]["inpainting_mask"] = torch.ones( start_motion.shape )
if args.mask_path: # Load the mask if provided
    mask_np = np.load(args.mask_path)
    assert mask_np.shape == (max_frames,), \
        f"Mask shape {mask_np.shape} != input frames {(max_frames, )}"
    mask_t = torch.tensor(mask_np, dtype=torch.bool, device=start_motion.device)
    model_kwargs["y"]["inpainting_mask"][..., :] = mask_t
else:
    # ( BATCH_SIZE, 236, 1, 60) = > BATCH_SIZE x POSE_DIM x 1 x NUM_FRAMES
    model_kwargs["y"]["inpainting_mask"][..., 20:40] = 0

model_kwargs["y"]["inpainted_motion"] = start_motion


gt_sample = sample.clone()
N = sample.shape[0]
T = max_frames # TODO: this could only handle 60 frames

sample = sample.to(dist_util.dev())

model_kwargs['y']['text'] = [''] * sample.shape[0]
guidance_param = 0.

model_kwargs["y"]["keyframe_mask"] = torch.tensor(model_kwargs["y"]["keyframe_mask"]).to(device)[:sample.shape[0]]
model_kwargs["y"]["features"] = torch.zeros(N, T, 2048)

noise = torch.randn(sample.shape).to(sample.device)
noise = torch.where(model_kwargs["y"]["keyframe_mask"], sample, noise)
model_kwargs["y"]["noise_mask"] = model_kwargs["y"]["keyframe_mask"].clone()
model_kwargs["y"]["original"] = sample.clone()

model_kwargs["y"]["action_text"] = np.array(model_kwargs["y"]["action_text"])


model_kwargs["y"]["inpainting_mask"] = model_kwargs["y"]["inpainting_mask"].to(dist_util.dev())
model_kwargs["y"]["inpainted_motion"] = model_kwargs["y"]["inpainted_motion"].to(dist_util.dev())

model.eval()
with torch.no_grad():
    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
                    model,
                    (args.batch_size, model.njoints, model.nfeats, max_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None, #noise,
                    const_noise=False,
                    const_t_noise=False,
                    grad_model=model
    )
    sample = sample.detach()

    condition = torch.tensor(model_kwargs["y"]["inpainted_motion"]).to(sample.device)

if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)

j_dic = model.forward_kinematics(condition, None)
smpl_joints_condition = j_dic['kp_45_joints'][:, :22].reshape(N, T, 22, 3) + j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
smpl_joints_condition = smpl_joints_condition.cpu().numpy()
# Save MDM input (which is linear interpolated)
np.save(os.path.join(out_path, 'results_linear.npy'), smpl_joints_condition)
with open(os.path.join(out_path, 'j_dic_linear.pkl'), 'wb') as f:
    pickle.dump(j_dic, f)

j_dic = model.forward_kinematics(sample, None)
diffs = torch.diff(mask_t) != 0
discontinuous_intervals = torch.nonzero(diffs, as_tuple=False).squeeze() + 1
print("Discontinuous intervals: ", discontinuous_intervals)
for idx in discontinuous_intervals:
    pred_trans1 = j_dic["pred_trans"][:idx].squeeze()
    pred_rotmat1 = j_dic["pred_rotmat"][:idx].squeeze()
    pred_trans2 = j_dic["pred_trans"][idx:].squeeze()
    pred_rotmat2 = j_dic["pred_rotmat"][idx:].squeeze()

    glue(
        j_dic["pred_trans"][:idx].squeeze(),
        j_dic["pred_rotmat"][:idx].squeeze(),
        j_dic["pred_trans"][idx:].squeeze(),
        j_dic["pred_rotmat"][idx:].squeeze(),
    )
# from VIBE.lib.models.smpl import SMPL
# smpl = SMPL("./body_models/smpl/", batch_size=64, create_transl=False)
# pred_rotmat = j_dic["pred_rotmat"]
# betas = torch.zeros((N * T, 10)).to(pred_rotmat.device)
# smpl_output = smpl(betas=betas.to(pred_rotmat.device),
#                    body_pose=pred_rotmat[:, 1:],
#                    global_orient=pred_rotmat[:, [0]],
#                    pose2rot=False)
smpl_joints =  j_dic['kp_45_joints'][:, :22].reshape(N, T, 22, 3) + j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
smpl_joints = smpl_joints.cpu().numpy()

# Save MDM output
np.save(os.path.join(out_path, 'results.npy'), smpl_joints)
with open(os.path.join(out_path, 'j_dic.pkl'), 'wb') as f:
    pickle.dump(j_dic, f)
# np.save(os.path.join(out_path, "results_tinyviz.npy"),
#         np.concatenate([smpl_joints_condition, smpl_joints], axis=0))
# with open(os.path.join(out_path, 'smpl_output.pkl'), 'wb') as f:
#     pickle.dump(smpl_output, f)

# visualization
import sys, os
repo_root = "/home/zhiyin/motion-diffusion-model"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)      # ← inserted in front of existing paths
from data_loaders.humanml.utils.plot_script import plot_3d_motion

print(f"saving visualizations to [{out_path}]...")
save_file = 'samples_{:02d}_to_{:02d}.mp4'.format(0, 0)
animation_save_path = os.path.join(out_path, save_file)
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
skeleton = t2m_kinematic_chain
motion = smpl_joints[0]
caption = 'Edit [{}] unconditioned'.format("in_between")
if args.mask_path:
    gt_frames = np.where(mask_np == 1)[0].tolist()
else:
    gt_frames = list(range(0, 20)) + list(range(40, T))

animation = plot_3d_motion(animation_save_path, 
                            skeleton, motion, dataset=args.dataset, title=caption, 
                            fps=fps, gt_frames=gt_frames,
                            global_coords=True)
animation.write_videofile(
    animation_save_path,
    fps=fps,          # mandatory for raw VideoClip objects
    codec="libx264",  # good default (H.264)
    preset="medium",  # faster→"fast"/"ultrafast", smaller file→"slow"
    logger=None       # drop this line if you want ffmpeg progress messages
)

print(f"saving  (linear interpolated) visualizations to [{out_path}]...")
save_file = 'samples_{:02d}_to_{:02d}_linear.mp4'.format(0, 0)
animation_save_path = os.path.join(out_path, save_file)
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
skeleton = t2m_kinematic_chain
motion = smpl_joints_condition[0]
caption = 'Edit [{}] unconditioned'.format("in_between")
gt_frames = list(range(0, T))

animation = plot_3d_motion(animation_save_path, 
                            skeleton, motion, dataset=args.dataset, title=caption, 
                            fps=fps, gt_frames=gt_frames,
                            global_coords=True)
animation.write_videofile(
    animation_save_path,
    fps=fps,          # mandatory for raw VideoClip objects
    codec="libx264",  # good default (H.264)
    preset="medium",  # faster→"fast"/"ultrafast", smaller file→"slow"
    logger=None       # drop this line if you want ffmpeg progress messages
)