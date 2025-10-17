import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import openslide
import pandas as pd
import numpy as np
from scipy.ndimage import generic_filter

from data.dataloader import WSI_patch_Dataset

def min_if_center_exists(values):
    """
    Return the minimum value inside the filter, if a center value exists within the filter.
    """
    center_value = values[len(values) // 2]
    if np.isnan(center_value):
        return np.nan
    else:
        valid_values = values[~np.isnan(values)]
        return np.min(valid_values) if valid_values.size > 0 else np.nan

def draw_heatmap(raw_WSI_path, patch_Ascores_csv_path, patch_size, scale_factor, erosion_filter=(3, 3)):
    """
    Perform heatmap generation and erosion on a downsampled WSI.

    Args:
        raw_WSI_path (str): Path to the raw WSI (.svs file).
        Ascore_df (pd.DataFrame): DataFrame containing patch names and corresponding anomaly scores.
        patch_size (int): Size of each patch in pixels.
        scale_factor (int): Downsampling factor used for WSI visualization.

    Notes:
        In the experiments presented in the paper, the `scale_factor` was set equal to the `patch_size`.
        The `scale_factor` must be less than or equal to the `patch_size`,
        and `patch_size` must be divisible by `scale_factor` with no remainder.
    """
    assert scale_factor <= patch_size, \
        f"Error: scale_factor ({scale_factor}) must be less than or equal to patch_size ({patch_size})."
    assert patch_size % scale_factor == 0, \
        f"Error: patch_size ({patch_size}) must be divisible by scale_factor ({scale_factor}) with no remainder."

    slide = openslide.OpenSlide(raw_WSI_path)
    slide_width, slide_height = slide.dimensions
    thumb_width, thumb_height = slide_width // scale_factor, slide_height // scale_factor
    heatmap = np.full((thumb_height, thumb_width), np.nan)

    Ascore_df = pd.read_csv(patch_Ascores_csv_path)
    Ascores = []
    for index, row in Ascore_df.iterrows():
        Ascore = float(row['Ascore'])
        Ascores.append(Ascore)

        patch_name = str(row['patch_name'])
        split_list = patch_name.split('_')
        w_coord = int(split_list[-3])
        h_coord = int(split_list[-2])

        w0 = w_coord // scale_factor
        h0 = h_coord // scale_factor
        if 0 <= w0 < thumb_width and 0 <= h0 < thumb_height:
            w3 = (w_coord + patch_size) // scale_factor
            h3 = (h_coord + patch_size) // scale_factor

            if np.isnan(heatmap[h0:h3, w0:w3]):
                heatmap[h0:h3, w0:w3] = Ascore
            else: print(f'[Error] {raw_WSI_path} Ascores are overlapped.')

    eroded_heatmap = generic_filter(heatmap, min_if_center_exists, size=erosion_filter, mode='constant', cval=np.nan)

    return eroded_heatmap, Ascores

def WSI_level_Ascore_erosion_true(eroded_heatmap):
    Ascore_max = np.nanmax(eroded_heatmap)

    valid_values = eroded_heatmap[~np.isnan(eroded_heatmap)]
    sorted_values = np.sort(valid_values)[::-1]
    top_1_percent_count = max(1, len(sorted_values) // 100)
    top_1_percent_values = sorted_values[:top_1_percent_count]
    Ascore_top1pct = np.mean(top_1_percent_values)

    return Ascore_max, Ascore_top1pct

def WSI_level_Ascore_erosion_false(Ascores):
    Ascore_max = np.nanmax(Ascores)

    sorted_scores = sorted(Ascores, reverse=True)
    top_1_percent_count = max(1, len(sorted_scores) // 100)
    top_1_percent_values = sorted_scores[:top_1_percent_count]
    Ascore_top1pct = np.mean(top_1_percent_values)

    return Ascore_max, Ascore_top1pct

def test(args):
    device = args["device"]
    prefix = args["prefix"]
    logger = args["logger"]
    VLM_img_preprocess = args["VLM_img_preprocess"]
    test_batch_size = args["test_batch_size"]
    model = args["trained_model"]

    patch_Ascores_path = f'./exp/{prefix}/patch_Ascores'
    os.makedirs(patch_Ascores_path, exist_ok=True)

    # Patch-level anomaly scores
    patches_WSIs_top_path = args["test_patches_WSIs_path"]
    for WSI_name in os.listdir(patches_WSIs_top_path):
        folder_path = os.path.join(patches_WSIs_top_path, WSI_name)
        if os.path.isdir(folder_path):
            patch_Ascores_csv_path = os.path.join(patch_Ascores_path, f"{WSI_name}_Ascores.csv")
            if os.path.exists(patch_Ascores_csv_path):
                logger.info(f"{WSI_name}_Ascores.csv already exists. Skipping...")
                continue

            WSI_dataset = WSI_patch_Dataset(folder_path=folder_path, transform=VLM_img_preprocess)
            WSI_dataloader = DataLoader(WSI_dataset, batch_size=test_batch_size, shuffle=False)

            patch_Ascores = []
            for batch in tqdm(WSI_dataloader, desc=f"Testing {WSI_name}"):
                imgs, img_names = batch
                imgs = imgs.to(device)

                d_n, d_a = model.compute_distances(imgs)
                Ascore = (d_n + d_a).cpu().numpy()

                patch_Ascores.extend(zip(img_names, Ascore))

            Ascore_df = pd.DataFrame(patch_Ascores, columns=['patch_name', 'Ascore'])
            Ascore_df.to_csv(patch_Ascores_csv_path, index=False)
            logger.info(f"Saved Ascore CSV: {patch_Ascores_csv_path}")

    # WSI-level anomaly scores
    if args["WSI_level"]:
        raw_WSIs_path = args["test_raw_WSIs_path"]
        test_patch_size = args["test_patch_size"]
        WSI_downsampling_factor = args["WSI_downsampling_factor"]

        WSI_level_Ascore_erosion_true_df = pd.DataFrame(columns=['WSI_name', 'Ascore_max', 'Ascore_top1pct'])
        erosion_true_path = f'./exp/{prefix}/WSI_level_Ascore_erosion_true.csv'
        WSI_level_Ascore_erosion_false_df = pd.DataFrame(columns=['WSI_name', 'Ascore_max', 'Ascore_top1pct'])
        erosion_false_path = f'./exp/{prefix}/WSI_level_Ascore_erosion_false.csv'

        for file in os.listdir(patch_Ascores_path):
            patch_Ascores_csv_path = os.path.join(patch_Ascores_path, file)
            WSI_name = file.replace('_Ascores.csv', '')
            raw_WSI_path = os.path.join(raw_WSIs_path, f"{WSI_name}.svs")
            if not os.path.exists(raw_WSI_path):
                logger.error(f'{raw_WSI_path} does not exist')
                continue

            eroded_heatmap, Ascores = draw_heatmap(raw_WSI_path, patch_Ascores_csv_path, test_patch_size, scale_factor=WSI_downsampling_factor)

            Ascore_max, Ascore_top1pct = WSI_level_Ascore_erosion_true(eroded_heatmap)
            WSI_level_Ascore_erosion_true_df.loc[len(WSI_level_Ascore_erosion_true_df)] = [WSI_name, Ascore_max, Ascore_top1pct]
            Ascore_max, Ascore_top1pct = WSI_level_Ascore_erosion_false(Ascores)
            WSI_level_Ascore_erosion_false_df.loc[len(WSI_level_Ascore_erosion_false_df)] = [WSI_name, Ascore_max, Ascore_top1pct]

        WSI_level_Ascore_erosion_true_df.to_csv(erosion_true_path, index=False)
        logger.info(f"Saved erosion-true WSI-level Ascore CSV: {erosion_true_path}")
        WSI_level_Ascore_erosion_false_df.to_csv(erosion_false_path, index=False)
        logger.info(f"Saved erosion-false WSI-level Ascore CSV: {erosion_false_path}")