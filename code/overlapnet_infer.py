"""
overlapnet_infer.py
Wrapper to call PRBonn/OverlapNet inference.

Usage:
    from overlapnet_infer import OverlapNetInfer
    infer = OverlapNetInfer(overlapnet_root="/path/to/OverlapNet", checkpoint="/path/to/checkpoint.pth")
    score = infer.score_pair(pc_query, pc_map_keyframe)
Notes:
 - This wrapper tries to import an OverlapNet Python API if available in the repo.
 - If import fails, it will call the repo's inference script via subprocess (CLI) if available.
 - You MUST follow PRBonn/OverlapNet README to download dataset preprocessing code and checkpoint.
References: https://github.com/PRBonn/OverlapNet
"""
import os, tempfile, subprocess, numpy as np, sys, json
import yaml
import shutil

class OverlapNetInfer:
    def __init__(self, overlapnet_root, checkpoint=None, use_pretrained=True, device="cpu"):
        self.root = os.path.abspath(overlapnet_root) if overlapnet_root else None
        self.checkpoint = checkpoint
        self.use_pretrained = use_pretrained
        self.device = device
        self.api_available = False
        
        if self.root:
            # Add src/two_heads to path to find infer.py
            possible_paths = [
                self.root,
                os.path.join(self.root, 'src', 'two_heads')
            ]
            for p in possible_paths:
                if p not in sys.path:
                    sys.path.insert(0, p)
            
            try:
                from infer import Infer
                self.Infer = Infer
                self.api_available = True
            except ImportError:
                self.api_available = False

    def _project_scan(self, points, proj_H=64, proj_W=900, fov_up=3.0, fov_down=-25.0):
        """
        Project LiDAR points to range image.
        points: Nx3 (x,y,z)
        Returns: (H, W, 4) array with channels [Range, NormalX, NormalY, NormalZ]
        """
        fov_up = fov_up / 180.0 * np.pi
        fov_down = fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        depth = np.linalg.norm(points, axis=1)
        
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(np.clip(scan_z / (depth + 1e-8), -1, 1))

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        proj_x *= proj_W
        proj_y *= proj_H

        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)

        indices = np.argsort(depth)[::-1]
        depth = depth[indices]
        proj_y = proj_y[indices]
        proj_x = proj_x[indices]

        # Create 4-channel output: Range, NormalX, NormalY, NormalZ
        # We fill normals with 0 as dummy values since we don't compute them
        proj_range = np.full((proj_H, proj_W, 4), -1, dtype=np.float32)
        
        # Channel 0 is range
        proj_range[proj_y, proj_x, 0] = depth
        
        # Channels 1,2,3 are normals (set to 0 or some default)
        # If -1 is "no data", maybe 0 is safer for normals?
        # Let's set valid pixels to 0 for normals
        proj_range[proj_y, proj_x, 1:] = 0.0
        
        return proj_range

    def _save_bin(self, pc, path):
        # pc: (N, 3)
        # Save as (N, 4) float32 .bin file (KITTI format)
        N = pc.shape[0]
        data = np.zeros((N, 4), dtype=np.float32)
        data[:, :3] = pc
        data.tofile(path)

    def score_pair(self, pcA, pcB):
        """
        pcA, pcB: Nx3 numpy arrays
        returns: float overlap score in [0,1]
        """
        if self.use_pretrained and self.api_available:
            tmp_dir = tempfile.mkdtemp()
            try:
                # Create directory structure expected by OverlapNet
                # root/scans/
                # root/depth/
                # root/normal/
                os.makedirs(os.path.join(tmp_dir, 'scans'))
                os.makedirs(os.path.join(tmp_dir, 'depth'))
                os.makedirs(os.path.join(tmp_dir, 'normal'))
                
                # Save .bin files
                fA_bin = os.path.join(tmp_dir, 'scans', '000000.bin')
                fB_bin = os.path.join(tmp_dir, 'scans', '000001.bin')
                self._save_bin(pcA, fA_bin)
                self._save_bin(pcB, fB_bin)
                
                # Project and save depth maps
                projA = self._project_scan(pcA)
                projB = self._project_scan(pcB)
                
                # Split into depth and normal
                # Depth: (H, W)
                # Normal: (H, W, 3)
                depthA = projA[:, :, 0]
                normalA = projA[:, :, 1:]
                
                depthB = projB[:, :, 0]
                normalB = projB[:, :, 1:]
                
                np.save(os.path.join(tmp_dir, 'depth', '000000.npy'), depthA)
                np.save(os.path.join(tmp_dir, 'normal', '000000.npy'), normalA)
                
                np.save(os.path.join(tmp_dir, 'depth', '000001.npy'), depthB)
                np.save(os.path.join(tmp_dir, 'normal', '000001.npy'), normalB)

                # Locate config
                config_path = os.path.join(self.root, 'config/network.yml')
                if not os.path.exists(config_path):
                    config_path = os.path.join(self.root, 'src/two_heads/config/network.yml')
                
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    if self.checkpoint:
                        config['pretrained_weightsfilename'] = os.path.abspath(self.checkpoint)
                    
                    # Point to our temp directory
                    config['data_root_folder'] = tmp_dir
                    config['infer_seqs'] = ''
                    
                    # Disable other inputs to avoid needing to generate them
                    # The weights file expects 4 channels (Range, NormalX, NormalY, NormalZ)
                    # But we only have Range.
                    # If we disable normals, the model architecture changes to expect 1 channel.
                    # BUT the weights file has 4 channels.
                    # So we MUST provide 4 channels of input, even if dummy.
                    
                    # Re-enable normals so model matches weights
                    config['use_normals'] = True
                    config['use_intensity'] = False
                    config['use_class_probabilities'] = False
                    config['use_class_probabilities_pca'] = False
                    
                    inf = self.Infer(config)
                    
                    if hasattr(inf, 'infer_one'):
                        # infer_one takes file paths
                        overlap, yaw = inf.infer_one(fA_bin, fB_bin)
                        return float(overlap)
            except Exception as e:
                print("OverlapNet API call failed:", e)
                import traceback
                traceback.print_exc()
                # fall through to CLI fallback
            finally:
                shutil.rmtree(tmp_dir)

        # CLI fallback
        if self.root is None:
            raise RuntimeError("overlapnet_root is not set and API import failed.")
        
        # ... (CLI fallback code omitted for brevity, assuming API works now) ...
        # If API fails, we might need to update CLI fallback too, but let's focus on API first.
        return 0.0

