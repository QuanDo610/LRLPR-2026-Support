"""MultiFrameDataset for license plate recognition with multi-frame input."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_degradation_transforms,
    get_light_transforms,
    get_hr_sync_transforms,
    get_lr_post_degrade_transforms,
)


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame license plate recognition.
    
    Handles both real LR images and synthetic LR (degraded HR) images.
    Implements Scenario-B specific validation splitting logic.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
        return_hr: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            img_height: Target image height.
            img_width: Target image width.
            char2idx: Character to index mapping.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
            augmentation_level: 'full' or 'light' augmentation for training.
            is_test: If True, load test data without labels (for submission).
            full_train: If True, use all tracks for training (no val split).
            return_hr: If True, also returns the HR tensor (for joint SR training).
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        self.return_hr = return_hr
        self.hr_height = 32
        self.hr_width = 128
        
        if mode == 'train':
            # Training: apply augmentation on the fly
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
            self.hr_sync_transform = (
                get_hr_sync_transforms(self.hr_height, self.hr_width)
                if augmentation_level != "light"
                else None
            )
            self.lr_post_transform = get_lr_post_degrade_transforms(
                img_height, img_width, apply_dropout=(augmentation_level != "light")
            )
            self.hr_norm_transform = get_val_transforms(self.hr_height, self.hr_width)
        else:
            # Validation or test: only resize and normalize
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None
            self.hr_sync_transform = None
            self.lr_post_transform = None
            self.hr_norm_transform = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        # Handle test mode differently
        if is_test:
            print(f"[TEST] Loaded {len(all_tracks)} tracks.")
            self._index_test_samples(all_tracks)
            print(f"-> Total: {len(self.samples)} test samples.")
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            
            selected_tracks = train_tracks if mode == 'train' else val_tracks
            print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
            
            self._index_samples(selected_tracks)
            print(f"-> Total: {len(self.samples)} samples.")

    @staticmethod
    def get_track_unique_id(track_path: str) -> str:
        """Get globally unique identifier for a track including its scenario/subset.
        
        Examples:
            'dataset/data/train/Scenario-A/Mercosur/track_04378' -> 'Scenario-A/Mercosur/track_04378'
            'dataset/data/train/Scenario-B/track_00001' -> 'Scenario-B/track_00001'
        """
        norm = os.path.normpath(track_path)
        for sc in ["Scenario-A", "Scenario-B"]:
            if sc in norm:
                idx = norm.find(sc)
                return norm[idx:]
        parts = norm.split(os.sep)
        return os.path.join(parts[-2], parts[-1]) if len(parts) >= 2 else os.path.basename(track_path)

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one with Scenario-B priority."""
        # If full_train mode, return all tracks as training
        if self.full_train:
            print("📌 FULL TRAIN MODE: Using all tracks for training (no validation split).")
            return all_tracks, []
        
        train_tracks, val_tracks = [], []
        
        # 1. Load split file if exists
        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                unique_id = self.get_track_unique_id(t)
                base_name = os.path.basename(t)
                # Match by unique_id, or if legacy split, match base_name ONLY if from Scenario-B
                if unique_id in val_ids or (base_name in val_ids and "Scenario-B" in t):
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            # Check consistency: If val empty or no Scenario-B, recreate
            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split file invalid or missing Scenario-B. Recreating...")
                val_tracks = []  # Reset to trigger new split logic

        # 2. Create new split if needed
        if not val_tracks:
            print("⚠️ Creating new split (Taking Val only from Scenario-B)...")
            
            # Filter Scenario-B tracks
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
            
            if not scenario_b_tracks:
                print("⚠️ Warning: No 'Scenario-B' folder found. Using random from all.")
                scenario_b_tracks = all_tracks
            
            # Val size = (1 - split_ratio) * total_scenario_b
            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            
            # Shuffle and take from beginning as val
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            
            # Train = (All) - (Val)
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]
            
            # Save unique track IDs (scenario + folder)
            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([self.get_track_unique_id(t) for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Index all samples from selected tracks."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                
                track_id = os.path.basename(track_path)
                
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                # Real LR samples
                self.samples.append({
                    'paths': lr_files,
                    'label': label,
                    'is_synthetic': False,
                    'track_id': track_id
                })
                
                # Synthetic LR samples (only in training mode)
                if self.mode == 'train':
                    self.samples.append({
                        'paths': hr_files,
                        'label': label,
                        'is_synthetic': True,
                        'track_id': track_id
                    })
            except Exception:
                pass

    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test samples without labels."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            track_id = os.path.basename(track_path)
            
            # Load all LR images (sorted by frame number)
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png")) +
                glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            
            if lr_files:
                self.samples.append({
                    'paths': lr_files,
                    'label': '',  # No label for test data
                    'is_synthetic': False,
                    'track_id': track_id
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """Load exactly 5 frames (guaranteed by dataset structure).
        
        For training: applies degradation (if synthetic) then augmentation.
        For validation: applies degradation (if synthetic) then clean transform.
        For test: only applies clean transform, returns dummy targets.
        """
        item = self.samples[idx]
        img_paths = item['paths']
        label = item['label']
        is_synthetic = item['is_synthetic']
        track_id = item['track_id']
        
        # Standardize synthetic HR frames to canonical (32, 128) canvas before degradation
        # so Downscale(0.3, 0.5) behaves identically for 16x64 (Flow 2/3) and 32x128 (Flow 1)
        load_h = max(self.img_height, 32) if (is_synthetic and self.degrade) else self.img_height
        load_w = max(self.img_width, 128) if (is_synthetic and self.degrade) else self.img_width

        # Load all frames for this track
        raw_images = []
        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((load_h, load_w, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (load_w, load_h))
            raw_images.append(img)

        num_frames = len(raw_images)
        if num_frames == 0:
            raw_images = [np.zeros((load_h, load_w, 3), dtype=np.uint8) for _ in range(5)]
            num_frames = 5

        def _apply_multi_frame(t, imgs):
            if num_frames > 1 and hasattr(t, 'additional_targets') and len(t.additional_targets) >= num_frames - 1:
                t_dict = {f'image{i}': imgs[i] for i in range(1, num_frames)}
                res = t(image=imgs[0], **t_dict)
                return [res['image']] + [res[f'image{i}'] for i in range(1, num_frames)]
            return [t(image=img)['image'] for img in imgs]

        if self.return_hr:
            if is_synthetic and self.degrade:
                # 1. Clean HR synchronized geometric & photometric transform on (32, 128)
                if self.hr_sync_transform is not None:
                    clean_hr = _apply_multi_frame(self.hr_sync_transform, raw_images)
                else:
                    clean_hr = [cv2.resize(img, (self.hr_width, self.hr_height)) for img in raw_images]

                # 2. Normalize clean HR to [-1, 1] tensor
                hr_norm_list = _apply_multi_frame(self.hr_norm_transform, clean_hr)
                hr_tensor = torch.stack(hr_norm_list, dim=0)
                has_hr = True

                # 3. Degrade clean HR
                deg_hr = _apply_multi_frame(self.degrade, clean_hr)

                # 4. Downsample to target LR size (16, 64) and apply dropout
                lr_list = _apply_multi_frame(self.lr_post_transform, deg_hr)
                images_tensor = torch.stack(lr_list, dim=0)
            else:
                images_list = _apply_multi_frame(self.transform, raw_images)
                images_tensor = torch.stack(images_list, dim=0)
                hr_tensor = torch.zeros((num_frames, 3, self.hr_height, self.hr_width), dtype=torch.float32)
                has_hr = False
        else:
            # 1. Apply degradation if synthetic (coherently across multi-frame if supported)
            if is_synthetic and self.degrade:
                raw_images = _apply_multi_frame(self.degrade, raw_images)

            # 2. Apply transform (synchronously across frames to strictly preserve temporal coherence)
            images_list = _apply_multi_frame(self.transform, raw_images)
            images_tensor = torch.stack(images_list, dim=0)
        
        # Handle test mode (no labels)
        if self.is_test:
            target = [0]  # Dummy target
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if len(target) == 0:
                target = [0]
            target_len = len(target)
            
        if self.return_hr:
            return images_tensor, torch.tensor(target, dtype=torch.long), target_len, label, track_id, hr_tensor, has_hr
        return images_tensor, torch.tensor(target, dtype=torch.long), target_len, label, track_id

    @staticmethod
    def collate_fn(batch: List[Tuple]):
        """Custom collate function for DataLoader (supports 5-tuple and 7-tuple with paired HR)."""
        if len(batch[0]) == 7:
            images, targets, target_lengths, labels_text, track_ids, hr_images, has_hr = zip(*batch)
            images = torch.stack(images, 0)
            targets = torch.cat(targets)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            hr_images = torch.stack(hr_images, 0)
            has_hr_mask = torch.tensor(has_hr, dtype=torch.bool)
            return images, targets, target_lengths, labels_text, track_ids, hr_images, has_hr_mask

        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text, track_ids
