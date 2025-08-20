# Point Cloud Segmentation Dataset Generation

This folder contains scripts to process raw point cloud data and generate datasets for segmentation tasks. The main entry point is `generate_dataset.py`, which builds new datasets according to the parameters specified in `config.yaml`.

## General Procedure

1. **Configure Parameters**  
   Edit the `config.yaml` file to set all required parameters, such as:
   - Paths to raw and processed data
   - Number of points per object
   - Target object name
   - Number of orientations and objects per sample
   - Other dataset and training parameters

2. **Prepare Raw Data**  
   Ensure your raw point cloud files (`.pcd`) are organized in subfolders under the directory specified by `RAW_DATASET_DIR`. Each subfolder should correspond to a different object.

3. **Run Dataset Generation**  
   Execute the following command in the terminal from this directory:

the file generate_dataset.py:

This will:
- Process each object's `.pcd` files into HDF5 datasets with the specified number of points (with padding or subsampling as needed).
- Generate segmentation datasets by combining target and alien objects, applying random transformations and orientations as configured.

4. **Output**  
The processed datasets will be saved in the directories specified in `config.yaml` under `POINTCLOUD_SAVE_DIR`. Segmentation datasets will be stored in a subfolder named `segmentation_dataset`.

---

**Note:**  
- Make sure all dependencies (e.g., Open3D, h5py, numpy, PyYAML) are installed.
- Adjust paths and parameters in `config.yaml` as needed for your dataset and experiment.