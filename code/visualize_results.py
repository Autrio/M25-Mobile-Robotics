import os
import argparse
import numpy as np
import glob
import sys

def visualize_matplotlib(map_pc, query_pc, reg_pc, mask):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 7))
    
    # Plot 1: Map vs Registered Query
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Map (Gray) vs Registered Query (Blue)")
    
    # Subsample for faster plotting if needed
    step = 10
    ax1.scatter(map_pc[::step,0], map_pc[::step,1], map_pc[::step,2], c='gray', s=1, alpha=0.5, label='Map')
    if reg_pc is not None:
        ax1.scatter(reg_pc[::step,0], reg_pc[::step,1], reg_pc[::step,2], c='blue', s=1, alpha=0.5, label='Registered')
    
    ax1.legend()
    
    # Plot 2: Inliers vs Outliers
    if reg_pc is not None and mask is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Inliers (Green) vs Outliers (Red)")
        
        inliers = reg_pc[mask]
        outliers = reg_pc[~mask]
        
        ax2.scatter(map_pc[::step,0], map_pc[::step,1], map_pc[::step,2], c='gray', s=1, alpha=0.1)
        if len(outliers) > 0:
            ax2.scatter(outliers[::step,0], outliers[::step,1], outliers[::step,2], c='red', s=1, label='Outliers')
        if len(inliers) > 0:
            ax2.scatter(inliers[::step,0], inliers[::step,1], inliers[::step,2], c='green', s=1, label='Inliers')
        ax2.legend()
    
    plt.show()

def visualize_open3d(map_pc, query_pc, reg_pc, mask):
    import open3d as o3d
    
    geoms = []
    
    # Map
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(map_pc)
    pcd_map.paint_uniform_color([0.5, 0.5, 0.5]) # Gray
    geoms.append(pcd_map)
    
    if reg_pc is not None:
        if mask is not None:
            # Inliers Green, Outliers Red
            inliers = reg_pc[mask]
            outliers = reg_pc[~mask]
            
            if len(inliers) > 0:
                pcd_in = o3d.geometry.PointCloud()
                pcd_in.points = o3d.utility.Vector3dVector(inliers)
                pcd_in.paint_uniform_color([0.0, 1.0, 0.0])
                geoms.append(pcd_in)
            
            if len(outliers) > 0:
                pcd_out = o3d.geometry.PointCloud()
                pcd_out.points = o3d.utility.Vector3dVector(outliers)
                pcd_out.paint_uniform_color([1.0, 0.0, 0.0])
                geoms.append(pcd_out)
        else:
            # Just Blue
            pcd_reg = o3d.geometry.PointCloud()
            pcd_reg.points = o3d.utility.Vector3dVector(reg_pc)
            pcd_reg.paint_uniform_color([0.0, 0.0, 1.0])
            geoms.append(pcd_reg)
            
    o3d.visualization.draw_geometries(geoms, window_name="Registration Result")

def main():
    parser = argparse.ArgumentParser(description="Visualize registration results")
    parser.add_argument("--dir", default="out_run", help="Output directory")
    parser.add_argument("--backend", choices=["auto", "matplotlib", "open3d"], default="auto", help="Visualization backend")
    parser.add_argument("--kitti_query", type=int, help="Specific KITTI query index to visualize (e.g. 150)")
    args = parser.parse_args()
    
    map_path = os.path.join(args.dir, "map.npy")
    if not os.path.exists(map_path):
        print(f"Error: {map_path} not found.")
        return

    print(f"Loading map from {map_path}...")
    map_pc = np.load(map_path)
    
    # Determine which query/reg files to load
    reg_pc = None
    mask = None
    query_pc = None
    
    if args.kitti_query is not None:
        # Look for specific KITTI files
        reg_path = os.path.join(args.dir, f"query_{args.kitti_query:010d}_reg.npy")
        mask_path = os.path.join(args.dir, f"query_{args.kitti_query:010d}_mask.npy")
    else:
        # Try synthetic first
        reg_path = os.path.join(args.dir, "registered_query.npy")
        mask_path = os.path.join(args.dir, "overlap_mask.npy")
        query_path = os.path.join(args.dir, "query.npy")
        
        if not os.path.exists(reg_path):
            # Try to find any KITTI result
            reg_files = glob.glob(os.path.join(args.dir, "query_*_reg.npy"))
            if reg_files:
                # Sort to get the latest or first
                reg_files.sort()
                reg_path = reg_files[0]
                base = reg_path.replace("_reg.npy", "")
                mask_path = base + "_mask.npy"
                print(f"Found KITTI result: {reg_path}")
            else:
                print("No registered query found.")
                reg_path = None

    if reg_path and os.path.exists(reg_path):
        print(f"Loading registered query from {reg_path}...")
        reg_pc = np.load(reg_path)
    
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask from {mask_path}...")
        mask = np.load(mask_path)
        
    # Try to load query.npy if it exists (synthetic)
    if os.path.exists(os.path.join(args.dir, "query.npy")):
        query_pc = np.load(os.path.join(args.dir, "query.npy"))

    # Select backend
    use_open3d = False
    if args.backend == "open3d":
        use_open3d = True
    elif args.backend == "auto":
        try:
            import open3d
            use_open3d = True
        except ImportError:
            print("Open3D not found, falling back to Matplotlib.")
            use_open3d = False
            
    if use_open3d:
        visualize_open3d(map_pc, query_pc, reg_pc, mask)
    else:
        visualize_matplotlib(map_pc, query_pc, reg_pc, mask)

if __name__ == "__main__":
    main()
