import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
dst = "data"

def copy_recursive(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):

        rel_path = os.path.relpath(root, src_dir)       # Get local path: "src absolute path" - "root" = "local path"
        local_dir = os.path.join(dst_dir, rel_path)     # new local path: "dst" + "local path"

        # Create local directory
        os.makedirs(local_dir, exist_ok=True)

        # copy files
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(local_dir, file)
            shutil.copy2(src_file, dst_file)

        print("\nroot:", root)
        print("dirs:", dirs)
        print("files:", files)

    print(f"\n✅ Copied {src_dir} в {dst_dir}")

copy_recursive(path, dst)