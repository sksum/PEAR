import os
import argparse
import numpy as np
import cv2
import torch
import imageio
import decord
import lightning
from scipy.signal import savgol_filter
from huggingface_hub import hf_hub_download
from torchvision import transforms
from pytorch3d.renderer import PointLights

from models.modules.ehm import EHM_v2
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.pipeline_utils import to_tensor
from utils.graphics_utils import GS_Camera
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from utils.general_utils import ConfigDict, device_parser, add_extra_cfgs


# ── helpers (from inference_images.py) ──────────────────────────────────

def pad_and_resize(img, target_size=512):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    return padded_img


def build_cameras_kwargs(batch_size, focal_length):
    screen_size = torch.tensor([1024, 1024]).float()[None].repeat(batch_size, 1)
    return {
        'principal_point': torch.zeros(batch_size, 2).float(),
        'focal_length': focal_length,
        'image_size': screen_size,
        'device': 'cuda',
    }


def polynomial_smooth(sequence, window_size=5, polyorder=2):
    seq = np.asarray(sequence.cpu())
    return savgol_filter(seq, window_length=window_size, polyorder=polyorder, axis=0, mode='interp')


# ── main inference ──────────────────────────────────────────────────────

@torch.no_grad()
def inference(input_path, output_path, config_name, devices):
    os.makedirs(output_path, exist_ok=True)

    # model setup (same as inference_images.py)
    meta_cfg = ConfigDict(model_config_path=os.path.join('configs', f'{config_name}.yaml'))
    meta_cfg = add_extra_cfgs(meta_cfg)
    lightning.fabric.seed_everything(10)

    body_renderer = BodyRenderer("assets/SMPLX", 1024, focal_length=24.0).cuda()
    lights = PointLights(device='cuda:0', location=[[0.0, -1.0, -10.0]])

    ehm_basemodel = hf_hub_download(repo_id="BestWJH/PEAR_models", filename="ehm_model_stage1.pt", repo_type="model")
    ehm_model = Ehm_Pipeline(meta_cfg)
    _state = torch.load(ehm_basemodel, map_location='cpu', weights_only=True)
    ehm_model.backbone.load_state_dict(_state['backbone'], strict=False)
    ehm_model.head.load_state_dict(_state['head'], strict=False)
    ehm_model = ehm_model.cuda()

    ehm = EHM_v2("assets/FLAME", "assets/SMPLX").cuda()

    # read video
    video_reader = decord.VideoReader(input_path)
    print(f"Processing {len(video_reader)} frames from {input_path}")

    # per-frame inference — collect raw params
    body_sequence, flame_sequence, cam_sequence = [], [], []
    for i in range(len(video_reader)):
        frame = video_reader[i].asnumpy()
        resized = pad_and_resize(frame, target_size=256)
        img_patch = to_tensor(resized, 'cuda')
        img_patch = torch.permute(img_patch / 255, (2, 0, 1)).unsqueeze(0)

        outputs = ehm_model(img_patch)
        body_sequence.append(outputs['body_param'])
        flame_sequence.append(outputs['flame_param'])
        cam_sequence.append(outputs['pd_cam'])

    # temporal smoothing
    body_fields = ["global_pose", "body_pose", "left_hand_pose", "right_hand_pose",
                   "hand_scale", "head_scale", "exp", "shape"]
    smoothed_body = {}
    for key in body_fields:
        data = torch.cat([s[key] for s in body_sequence], dim=0)
        smoothed_body[key] = torch.tensor(polynomial_smooth(data, window_size=7, polyorder=2)).cuda()

    flame_fields = ["eye_pose_params", "pose_params", "jaw_params",
                    "eyelid_params", "expression_params", "shape_params"]
    smoothed_flame = {}
    for key in flame_fields:
        data = torch.cat([s[key] for s in flame_sequence], dim=0)
        smoothed_flame[key] = torch.tensor(polynomial_smooth(data, window_size=5, polyorder=2)).cuda()

    cam_smoothed = torch.tensor(
        polynomial_smooth(torch.cat(cam_sequence, dim=0), window_size=7, polyorder=2)
    ).cuda()

    # re-render with smoothed params
    num_frames = smoothed_body["global_pose"].shape[0]
    all_meshes_img = []
    vertices_list = []

    for idx in range(num_frames):
        body_dict = {k: smoothed_body[k][idx:idx+1] for k in body_fields}
        body_dict.update({'eye_pose': None, 'jaw_pose': None, 'joints_offset': None})

        flame_dict = {k: smoothed_flame[k][idx:idx+1] for k in flame_fields}

        pd_smplx_dict = ehm(body_dict, flame_dict, pose_type='aa')
        pd_camera = GS_Camera(**build_cameras_kwargs(1, 24),
                              R=cam_smoothed[idx:idx+1, :3, :3],
                              T=cam_smoothed[idx:idx+1, :3, 3])
        pd_mesh_img = body_renderer.render_mesh(pd_smplx_dict['vertices'][None, 0, ...], pd_camera, lights=lights)
        pd_mesh_img = (pd_mesh_img[:, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
        all_meshes_img.append(pd_mesh_img)
        vertices_list.append(pd_smplx_dict['vertices'][0].detach().cpu().numpy())

    # save mesh video
    mesh_video_path = os.path.join(output_path, "mesh_video.mp4")
    writer = imageio.get_writer(mesh_video_path, fps=30, codec="libx264",
                                pixelformat="yuv420p",
                                ffmpeg_params=["-movflags", "faststart"],
                                macro_block_size=None)
    for img in all_meshes_img:
        h, w = img.shape[:2]
        writer.append_data(img[:h - (h % 2), :w - (w % 2)])
    writer.close()

    # save npz
    faces = body_renderer.faces[0].detach().cpu().numpy()
    vertices = np.stack(vertices_list, axis=0)
    np.savez_compressed(os.path.join(output_path, "results.npz"), vertices=vertices, faces=faces)

    print(f"Done. Saved mesh video to {mesh_video_path}")
    print(f"Saved mesh data to {os.path.join(output_path, 'results.npz')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='example/example_1.mp4', type=str)
    parser.add_argument('--output_path', default='example/video_output', type=str)
    parser.add_argument('--config_name', '-c', default='infer', type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    inference(args.input_path, args.output_path, args.config_name, args.devices)
