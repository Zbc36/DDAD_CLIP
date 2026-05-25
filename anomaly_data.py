import os
import time
import csv
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils import data
import json
from joblib import Parallel, delayed


def _coerce_record(record):
    if isinstance(record, str):
        return {"image": record, "mask": None}
    if isinstance(record, dict):
        if "image" not in record:
            raise KeyError("Dataset record dict must contain an 'image' field.")
        normalized = dict(record)
        normalized["image"] = str(record["image"])
        mask_name = record.get("mask")
        normalized["mask"] = None if mask_name in {None, ""} else str(mask_name)
        return normalized
    raise TypeError("Unsupported dataset record type: {}".format(type(record)))


def _record_image_name(record):
    return _coerce_record(record)["image"]


def _record_mask_name(record):
    return _coerce_record(record)["mask"]


def _record_img_id(record):
    normalized = _coerce_record(record)
    if "img_id" in normalized and normalized["img_id"] not in {None, ""}:
        return str(normalized["img_id"])
    return os.path.splitext(normalized["image"])[0]


def _record_group_id(record, group_key=None):
    normalized = _coerce_record(record)
    candidate_keys = []
    if group_key not in {None, "", "auto"}:
        candidate_keys.append(str(group_key))
    candidate_keys.extend([
        "group_id",
        "patient_id",
        "patient_uid",
        "study_id",
        "study_uid",
        "subject_id",
        "subject_uid",
        "series_id",
        "series_uid",
        "uid",
    ])

    metadata = normalized.get("meta")
    for key in candidate_keys:
        value = normalized.get(key)
        if value not in {None, ""}:
            return str(value)
        if isinstance(metadata, dict):
            meta_value = metadata.get(key)
            if meta_value not in {None, ""}:
                return str(meta_value)
    return _record_img_id(normalized)


def _load_grayscale_image(img_dir, img_name, img_size):
    with Image.open(os.path.join(img_dir, img_name)) as image:
        return image.convert("L").resize((img_size, img_size), resample=Image.BILINEAR)


def parallel_load(img_dir, records, img_size, verbose=0):
    image_names = [_record_image_name(record) for record in records]
    return Parallel(n_jobs=-1, verbose=verbose)(
        delayed(_load_grayscale_image)(img_dir, image_name, img_size) for image_name in image_names
    )


class AnomalyDetectionDataset(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", extra_data=0, ar=0.):
        super(AnomalyDetectionDataset, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_normal = list(data_dict["train"]["0"])

            normal_l = list(data_dict["train"]["unlabeled"]["0"])
            abnormal_l = list(data_dict["train"]["unlabeled"]["1"])
            if extra_data > 0:
                abnormal_num = int(extra_data * ar)
                normal_num = extra_data - abnormal_num
            else:
                abnormal_num = 0
                normal_num = 0

            train_l = train_normal + normal_l[:normal_num] + abnormal_l[:abnormal_num]
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_l, img_size)
            self.labels += (len(train_normal) + normal_num) * [0] + abnormal_num * [1]
            self.img_id += [_record_img_id(record) for record in train_l]
            print("Loaded {} normal images, "
                  "{} (unlabeled) normal images, "
                  "{} (unlabeled) abnormal images. {:.3f}s".format(len(train_normal), normal_num, abnormal_num,
                                                                   time.time() - t0))

        else:  # test
            test_normal = list(data_dict["test"]["0"])
            test_abnormal = list(data_dict["test"]["1"])

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [_record_img_id(record) for record in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]
        return img, label, img_id

    def __len__(self):
        return len(self.slices)


class SelfAnomalyDataset(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None):
        super(SelfAnomalyDataset, self).__init__()
        self.root = main_path
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x
        self.anomaly_transform = self.transform

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        t0 = time.time()
        train_normal = list(data_dict["train"]["0"])
        self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
        print("Loaded {} normal images. {:.3f}s".format(len(train_normal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)
        if np.random.rand() > 0.5:
            img, mask = self.generate_anomaly(img, index, core_percent=0.8)
            label = 1
        else:
            mask = torch.zeros_like(img).squeeze().long()
            label = 0
        return img, label, mask

    def __len__(self):
        return len(self.slices)

    def generate_anomaly(self, image, index, core_percent=0.8):
        dims = np.array(np.shape(image)[1:])  # H x W
        core = core_percent * dims  # width of core region
        offset = (1 - core_percent) * dims / 2  # offset to center core

        min_width = np.round(0.05 * dims[1])
        max_width = np.round(0.2 * dims[1])  # make sure it is less than offset

        center_dim1 = np.random.randint(offset[0], offset[0] + core[0])
        center_dim2 = np.random.randint(offset[1], offset[1] + core[1])
        patch_center = np.array([center_dim1, center_dim2])
        patch_width = np.random.randint(min_width, max_width)

        coor_min = patch_center - patch_width
        coor_max = patch_center + patch_width

        # clip coordinates to within image dims
        coor_min = np.clip(coor_min, 0, dims)
        coor_max = np.clip(coor_max, 0, dims)

        alpha = torch.rand(1)  #
        mask = torch.zeros_like(image).squeeze()
        mask[coor_min[0]:coor_max[0], coor_min[1]:coor_max[1]] = alpha
        mask_inv = 1 - mask

        # mix
        anomaly_source_index = np.random.randint(0, len(self.slices))
        while anomaly_source_index == index:
            anomaly_source_index = np.random.randint(0, len(self.slices))
        anomaly_source = self.slices[anomaly_source_index]
        anomaly_source = self.anomaly_transform(anomaly_source)
        image_synthesis = mask_inv * image + mask * anomaly_source

        return image_synthesis, (mask > 0).long()


def split_train_normal_records(data_dict, val_ratio=0.1, val_seed=0):
    train_normal = list(data_dict["train"]["0"])
    if val_ratio <= 0 or len(train_normal) <= 1:
        return train_normal, [], []

    rng = np.random.RandomState(val_seed)
    indices = np.arange(len(train_normal))
    rng.shuffle(indices)
    holdout_count = max(1, int(round(len(train_normal) * val_ratio)))
    holdout_indices = indices[:holdout_count]
    train_indices = indices[holdout_count:]

    holdout_records = [train_normal[idx] for idx in holdout_indices]
    train_records = [train_normal[idx] for idx in train_indices]

    midpoint = max(1, len(holdout_records) // 2)
    val_records = holdout_records[:midpoint]
    test_records = holdout_records[midpoint:] if midpoint < len(holdout_records) else holdout_records
    return train_records, val_records, test_records


def split_labeled_records(records, val_ratio=0.1, val_seed=0):
    records = list(records)
    if val_ratio <= 0 or len(records) <= 1:
        return records, []

    grouped_indices = {}
    for index, (_, label) in enumerate(records):
        grouped_indices.setdefault(int(label), []).append(index)

    rng = np.random.RandomState(val_seed)
    train_indices, val_indices = [], []
    for label in sorted(grouped_indices.keys()):
        label_indices = np.array(grouped_indices[label], dtype=np.int64)
        rng.shuffle(label_indices)
        if len(label_indices) <= 1:
            train_indices.extend(label_indices.tolist())
            continue

        val_count = max(1, int(round(len(label_indices) * val_ratio)))
        val_count = min(len(label_indices) - 1, val_count)
        val_indices.extend(label_indices[:val_count].tolist())
        train_indices.extend(label_indices[val_count:].tolist())

    train_indices.sort()
    val_indices.sort()
    return [records[index] for index in train_indices], [records[index] for index in val_indices]


def split_grouped_records(records, val_ratio=0.1, val_seed=0, group_key=None):
    records = list(records)
    if val_ratio <= 0 or len(records) <= 1:
        return records, []

    grouped_records = {}
    for record in records:
        group_id = _record_group_id(record, group_key=group_key)
        grouped_records.setdefault(group_id, []).append(record)

    group_ids = list(grouped_records.keys())
    if len(group_ids) <= 1:
        return records, []

    rng = np.random.RandomState(val_seed)
    rng.shuffle(group_ids)
    target_val_count = max(1, int(round(len(records) * val_ratio)))

    selected_val_groups = []
    selected_val_count = 0
    for group_id in group_ids:
        remaining_groups = len(group_ids) - len(selected_val_groups) - 1
        if selected_val_count >= target_val_count and len(selected_val_groups) > 0:
            break
        if remaining_groups < 1 and len(selected_val_groups) > 0:
            break
        selected_val_groups.append(group_id)
        selected_val_count += len(grouped_records[group_id])

    val_group_ids = set(selected_val_groups)
    val_records, train_records = [], []
    for group_id, group_records in grouped_records.items():
        if group_id in val_group_ids:
            val_records.extend(group_records)
        else:
            train_records.extend(group_records)

    if len(train_records) == 0:
        last_group_id = selected_val_groups[-1]
        train_records.extend(grouped_records[last_group_id])
        val_records = [
            record
            for group_id in selected_val_groups[:-1]
            for record in grouped_records[group_id]
        ]
    return train_records, val_records


class MultiBranchDataset(data.Dataset):
    """
    Unified dataset for the multi-branch anomaly detection pipeline.

    Returned sample keys:
      - image_64:  1 x image_size x image_size tensor for DDAD / diffusion
      - image_224: 3 x clip_image_size x clip_image_size tensor for CLIP branch
                   (legacy key name kept for runtime compatibility)
      - label_img: image-level anomaly label (0 or 1)
      - mask_syn:  1 x image_size x image_size synthetic mask
      - mask_real: 1 x image_size x image_size real lesion mask when available
      - has_real_mask: boolean tensor indicating whether mask_real is meaningful
      - img_id:    file stem
    """

    def __init__(
        self,
        main_path,
        subset="train_normal",
        image_size=64,
        clip_image_size=336,
        include_clip=True,
        val_ratio=0.1,
        val_seed=0,
        synthetic_probability=0.5,
        synthetic_mode_probs=None,
        synthetic_shape_probs=None,
        cache_images=True,
        cache_clip_images=None,
        force_deterministic=False,
        deterministic_seed=1337,
        clean_val_ratio=None,
        unlabeled_val_ratio=None,
        group_key=None,
    ):
        super(MultiBranchDataset, self).__init__()
        self.root = main_path
        self.subset = subset
        self.image_size = int(image_size)
        self.clip_image_size = int(clip_image_size)
        self.include_clip = bool(include_clip)
        self.synthetic_probability = float(synthetic_probability)
        self.cache_images = bool(cache_images)
        self.cache_clip_images = bool(self.cache_images if cache_clip_images is None else cache_clip_images)
        self.force_deterministic = bool(force_deterministic)
        self.deterministic_seed = int(deterministic_seed)
        self.group_key = None if group_key in {None, "", "auto"} else str(group_key)
        self.clean_val_ratio = float(val_ratio if clean_val_ratio is None else clean_val_ratio)
        self.unlabeled_val_ratio = float(val_ratio if unlabeled_val_ratio is None else unlabeled_val_ratio)
        self.synthetic_mode_probs = synthetic_mode_probs or {
            "copy_paste": 0.4,
            "intensity_shift": 0.2,
            "blur_or_sharpen": 0.2,
        }
        supported_modes = {"copy_paste", "intensity_shift", "blur_or_sharpen"}
        self.synthetic_mode_probs = {
            key: float(value)
            for key, value in self.synthetic_mode_probs.items()
            if key in supported_modes and float(value) > 0.0
        }
        if len(self.synthetic_mode_probs) == 0:
            self.synthetic_mode_probs = {
                "copy_paste": 0.4,
                "intensity_shift": 0.2,
                "blur_or_sharpen": 0.2,
            }
        self.synthetic_shape_probs = synthetic_shape_probs or {
            "rectangle": 0.15,
            "ellipse": 0.25,
            "blob": 0.25,
            "multi_blob": 0.20,
            "streak": 0.15,
        }

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        train_normal, synthetic_val, synthetic_test = split_train_normal_records(
            data_dict,
            val_ratio=val_ratio,
            val_seed=val_seed,
        )
        train_normal = [_coerce_record(record) for record in train_normal]
        synthetic_val = [_coerce_record(record) for record in synthetic_val]
        synthetic_test = [_coerce_record(record) for record in synthetic_test]
        unlabeled_normal = [_coerce_record(record) for record in list(data_dict["train"]["unlabeled"]["0"])]
        unlabeled_abnormal = [_coerce_record(record) for record in list(data_dict["train"]["unlabeled"]["1"])]
        clean_train_normal, clean_val_normal = split_grouped_records(
            list(data_dict["train"]["0"]),
            val_ratio=self.clean_val_ratio,
            val_seed=val_seed,
            group_key=self.group_key,
        )
        clean_train_normal = [_coerce_record(record) for record in clean_train_normal]
        clean_val_normal = [_coerce_record(record) for record in clean_val_normal]
        unlabeled_pool_records = []
        for record in list(data_dict["train"]["unlabeled"]["0"]):
            normalized = _coerce_record(record)
            normalized["_hidden_label"] = 0
            unlabeled_pool_records.append(normalized)
        for record in list(data_dict["train"]["unlabeled"]["1"]):
            normalized = _coerce_record(record)
            normalized["_hidden_label"] = 1
            unlabeled_pool_records.append(normalized)
        unlabeled_train_pool, unlabeled_val_pool = split_grouped_records(
            unlabeled_pool_records,
            val_ratio=self.unlabeled_val_ratio,
            val_seed=val_seed + 97,
            group_key=self.group_key,
        )
        test_normal = [_coerce_record(record) for record in list(data_dict["test"]["0"])]
        test_abnormal = [_coerce_record(record) for record in list(data_dict["test"]["1"])]
        real_records = [(name, 0) for name in train_normal + unlabeled_normal]
        real_records.extend((name, 1) for name in unlabeled_abnormal)
        real_train_records, real_val_records = split_labeled_records(
            real_records,
            val_ratio=val_ratio,
            val_seed=val_seed,
        )

        if subset == "train_normal":
            self.records = [(name, 0) for name in train_normal]
            self.synthetic_enabled = self.synthetic_probability > 0.0
            self.deterministic = False
            self.source_records = [(name, 0) for name in train_normal] if len(train_normal) > 0 else list(self.records)
        elif subset == "clean_train_normal":
            self.records = [(name, 0) for name in clean_train_normal]
            self.synthetic_enabled = self.synthetic_probability > 0.0
            self.deterministic = False
            self.source_records = list(self.records)
        elif subset == "clean_val_normal":
            self.records = [(name, 0) for name in clean_val_normal]
            self.synthetic_enabled = self.synthetic_probability > 0.0
            self.deterministic = True
            self.source_records = list(self.records)
        elif subset == "unlabeled_train_pool":
            self.records = [(record, 0) for record in unlabeled_train_pool]
            self.synthetic_enabled = False
            self.deterministic = True
            self.source_records = list(self.records)
        elif subset == "clean_plus_unlabeled_train_pool":
            self.records = [(record, 0) for record in clean_train_normal]
            self.records.extend((record, 0) for record in unlabeled_train_pool)
            self.synthetic_enabled = False
            self.deterministic = False
            self.source_records = [(record, 0) for record in clean_train_normal] if len(clean_train_normal) > 0 else list(self.records)
        elif subset == "unlabeled_val_pool":
            self.records = [(record, 0) for record in unlabeled_val_pool]
            self.synthetic_enabled = False
            self.deterministic = True
            self.source_records = list(self.records)
        elif subset == "train_plus_unlabeled":
            self.records = real_records
            self.synthetic_enabled = False
            self.deterministic = False
            self.source_records = [(name, 0) for name in train_normal] if len(train_normal) > 0 else list(self.records)
        elif subset == "real_train":
            self.records = real_train_records
            self.synthetic_enabled = False
            self.deterministic = False
            self.source_records = [(name, 0) for name in train_normal] if len(train_normal) > 0 else list(self.records)
        elif subset == "real_val":
            self.records = real_val_records
            self.synthetic_enabled = False
            self.deterministic = True
            self.source_records = [(name, 0) for name in train_normal] if len(train_normal) > 0 else list(self.records)
        elif subset == "real_test":
            self.records = [(name, 0) for name in test_normal]
            self.records.extend((name, 1) for name in test_abnormal)
            self.synthetic_enabled = False
            self.deterministic = True
            self.source_records = list(self.records)
        elif subset == "official_test":
            self.records = [(name, 0) for name in test_normal]
            self.records.extend((name, 1) for name in test_abnormal)
            self.synthetic_enabled = False
            self.deterministic = True
            self.source_records = list(self.records)
        elif subset == "synthetic_val":
            self.records = [(name, 0) for name in synthetic_val]
            self.synthetic_enabled = True
            self.deterministic = True
            self.source_records = [(name, 0) for name in train_normal] if len(train_normal) > 0 else list(self.records)
        elif subset == "synthetic_test":
            self.records = [(name, 0) for name in synthetic_test]
            self.synthetic_enabled = True
            self.deterministic = True
            self.source_records = [(name, 0) for name in train_normal] if len(train_normal) > 0 else list(self.records)
        else:
            raise ValueError("Unsupported MultiBranchDataset subset: {}".format(subset))

        if len(self.records) == 0:
            raise RuntimeError("No records found for subset {}".format(subset))

        self.images_dir = os.path.join(self.root, "images")
        self.masks_dir = os.path.join(self.root, "masks")
        self._grayscale_cache = {}
        self._clip_cache = {}
        self._mask_cache = {}

    def __len__(self):
        return len(self.records)

    def _make_rng(self, index):
        if self.force_deterministic:
            return np.random.RandomState(index + self.deterministic_seed)
        if self.deterministic:
            return np.random.RandomState(index + 1337)
        return np.random.RandomState(np.random.randint(0, 10 ** 6))

    def _load_grayscale_tensor(self, img_name, size):
        cache_key = (img_name, int(size))
        should_cache = self.cache_images and (int(size) != self.clip_image_size or self.cache_clip_images)
        if should_cache and cache_key in self._grayscale_cache:
            return self._grayscale_cache[cache_key].clone()

        with Image.open(os.path.join(self.images_dir, img_name)) as image:
            image = image.convert("L")
            image = image.resize((size, size), resample=Image.BILINEAR)
            image = np.asarray(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        image_tensor = (image_tensor * 2.0 - 1.0).contiguous()
        if should_cache:
            self._grayscale_cache[cache_key] = image_tensor
            return image_tensor.clone()
        return image_tensor

    def _to_clip_tensor(self, grayscale_tensor):
        return grayscale_tensor.repeat(3, 1, 1)

    def _load_clip_tensor(self, img_name):
        if self.cache_clip_images and img_name in self._clip_cache:
            return self._clip_cache[img_name].clone()

        clip_tensor = self._to_clip_tensor(self._load_grayscale_tensor(img_name, self.clip_image_size)).contiguous()
        if self.cache_clip_images:
            self._clip_cache[img_name] = clip_tensor
            return clip_tensor.clone()
        return clip_tensor

    def _sample_source_name(self, index, rng):
        if len(self.source_records) == 1:
            return _record_image_name(self.source_records[0][0])

        source_index = rng.randint(0, len(self.source_records))
        current_name = _record_image_name(self.records[index][0])
        while _record_image_name(self.source_records[source_index][0]) == current_name:
            source_index = rng.randint(0, len(self.source_records))
        return _record_image_name(self.source_records[source_index][0])

    def _load_binary_mask_tensor(self, mask_name, size):
        if mask_name in {None, ""}:
            return torch.zeros(1, size, size, dtype=torch.long)

        cache_key = (str(mask_name), int(size))
        if self.cache_images and cache_key in self._mask_cache:
            return self._mask_cache[cache_key].clone()

        mask_path = os.path.join(self.masks_dir, str(mask_name))
        if not os.path.exists(mask_path):
            raise FileNotFoundError("Expected real anomaly mask at {}".format(mask_path))

        with Image.open(mask_path) as mask_image:
            mask_image = mask_image.convert("L")
            mask_image = mask_image.resize((size, size), resample=Image.NEAREST)
            mask_array = np.asarray(mask_image, dtype=np.uint8)

        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.int64)).unsqueeze(0)
        if self.cache_images:
            self._mask_cache[cache_key] = mask_tensor
            return mask_tensor.clone()
        return mask_tensor

    def _sample_patch_center(self, size, rng, core_percent=0.8):
        dims = np.array([size, size], dtype=np.float32)
        core = core_percent * dims
        offset = (1.0 - core_percent) * dims / 2.0
        center_y = rng.randint(int(offset[0]), max(int(offset[0]) + 1, int(offset[0] + core[0])))
        center_x = rng.randint(int(offset[1]), max(int(offset[1]) + 1, int(offset[1] + core[1])))
        return center_y, center_x

    def _ellipse_mask_from_params(self, size, center_y, center_x, radius_y, radius_x, angle_rad=0.0):
        yy, xx = np.ogrid[:size, :size]
        yy = yy.astype(np.float32) - float(center_y)
        xx = xx.astype(np.float32) - float(center_x)
        cos_theta = float(np.cos(angle_rad))
        sin_theta = float(np.sin(angle_rad))
        x_rot = xx * cos_theta + yy * sin_theta
        y_rot = -xx * sin_theta + yy * cos_theta
        radius_y = max(1.0, float(radius_y))
        radius_x = max(1.0, float(radius_x))
        mask = (((x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2) <= 1.0).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def _build_rectangular_mask(self, size, rng, core_percent=0.8):
        min_half = max(2, int(round(0.04 * size)))
        max_half = max(min_half + 1, int(round(0.14 * size)))
        center_y, center_x = self._sample_patch_center(size, rng, core_percent=core_percent)
        half_h = rng.randint(min_half, max_half)
        half_w = rng.randint(min_half, max_half)

        y0 = max(0, center_y - half_h)
        y1 = min(size, center_y + half_h)
        x0 = max(0, center_x - half_w)
        x1 = min(size, center_x + half_w)

        mask = torch.zeros(1, size, size, dtype=torch.float32)
        mask[:, y0:y1, x0:x1] = 1.0
        return mask

    def _build_elliptical_mask(self, size, rng, core_percent=0.8):
        center_y, center_x = self._sample_patch_center(size, rng, core_percent=core_percent)
        radius_y = rng.uniform(0.04, 0.12) * float(size)
        radius_x = rng.uniform(0.04, 0.14) * float(size)
        angle_rad = rng.uniform(0.0, np.pi)
        return self._ellipse_mask_from_params(size, center_y, center_x, radius_y, radius_x, angle_rad)

    def _build_blob_mask(self, size, rng, core_percent=0.8):
        base_y, base_x = self._sample_patch_center(size, rng, core_percent=core_percent)
        component_count = rng.randint(2, 5)
        jitter = max(2, int(round(0.08 * size)))
        mask = torch.zeros(1, size, size, dtype=torch.float32)
        for _ in range(component_count):
            center_y = int(np.clip(base_y + rng.randint(-jitter, jitter + 1), 0, size - 1))
            center_x = int(np.clip(base_x + rng.randint(-jitter, jitter + 1), 0, size - 1))
            radius_y = rng.uniform(0.03, 0.08) * float(size)
            radius_x = rng.uniform(0.03, 0.09) * float(size)
            angle_rad = rng.uniform(0.0, np.pi)
            mask = torch.maximum(
                mask,
                self._ellipse_mask_from_params(size, center_y, center_x, radius_y, radius_x, angle_rad),
            )
        return (mask > 0).float()

    def _build_multi_blob_mask(self, size, rng, core_percent=0.8):
        base_y, base_x = self._sample_patch_center(size, rng, core_percent=core_percent)
        mask = torch.zeros(1, size, size, dtype=torch.float32)
        focus_count = rng.randint(2, 4)
        radius = rng.uniform(0.10, 0.18) * float(size)
        start_angle = rng.uniform(0.0, 2.0 * np.pi)
        for focus_idx in range(focus_count):
            angle = start_angle + (2.0 * np.pi * focus_idx / max(1, focus_count))
            center_y = int(np.clip(base_y + radius * np.sin(angle), 0, size - 1))
            center_x = int(np.clip(base_x + radius * np.cos(angle), 0, size - 1))
            radius_y = rng.uniform(0.03, 0.07) * float(size)
            radius_x = rng.uniform(0.03, 0.08) * float(size)
            angle_rad = rng.uniform(0.0, np.pi)
            mask = torch.maximum(
                mask,
                self._ellipse_mask_from_params(size, center_y, center_x, radius_y, radius_x, angle_rad),
            )
        return (mask > 0).float()

    def _build_streak_mask(self, size, rng, core_percent=0.8):
        center_y, center_x = self._sample_patch_center(size, rng, core_percent=core_percent)
        radius_y = rng.uniform(0.02, 0.05) * float(size)
        radius_x = rng.uniform(0.12, 0.22) * float(size)
        angle_rad = rng.uniform(0.0, np.pi)
        mask = self._ellipse_mask_from_params(size, center_y, center_x, radius_y, radius_x, angle_rad)
        cap_mask = self._ellipse_mask_from_params(
            size,
            int(np.clip(center_y + rng.randint(-max(1, int(radius_x // 3)), max(2, int(radius_x // 3) + 1)), 0, size - 1)),
            int(np.clip(center_x + rng.randint(-max(1, int(radius_x // 3)), max(2, int(radius_x // 3) + 1)), 0, size - 1)),
            rng.uniform(0.03, 0.06) * float(size),
            rng.uniform(0.03, 0.06) * float(size),
            rng.uniform(0.0, np.pi),
        )
        return torch.maximum(mask, cap_mask)

    def _pick_synthetic_shape(self, rng):
        shapes = list(self.synthetic_shape_probs.keys())
        probs = np.array([self.synthetic_shape_probs[key] for key in shapes], dtype=np.float64)
        probs = probs / probs.sum()
        return shapes[rng.choice(len(shapes), p=probs)]

    def _build_synthetic_mask(self, size, rng):
        shape = self._pick_synthetic_shape(rng)
        if shape == "rectangle":
            return self._build_rectangular_mask(size, rng)
        if shape == "ellipse":
            return self._build_elliptical_mask(size, rng)
        if shape == "blob":
            return self._build_blob_mask(size, rng)
        if shape == "multi_blob":
            return self._build_multi_blob_mask(size, rng)
        if shape == "streak":
            return self._build_streak_mask(size, rng)
        raise ValueError("Unsupported synthetic shape: {}".format(shape))

    def _copy_paste(self, image, source, mask):
        return (1.0 - mask) * image + mask * source

    def _intensity_shift(self, image, mask, rng):
        delta = float(rng.uniform(0.15, 0.35))
        if rng.rand() > 0.5:
            delta = -delta
        shifted = torch.clamp(image + delta, -1.0, 1.0)
        return (1.0 - mask) * image + mask * shifted

    def _blur_or_sharpen(self, image, mask, rng):
        kernel = 5 if rng.rand() > 0.5 else 3
        blurred = F.avg_pool2d(image.unsqueeze(0), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(0)
        if rng.rand() > 0.5:
            processed = blurred
        else:
            processed = torch.clamp(image + 0.7 * (image - blurred), -1.0, 1.0)
        return (1.0 - mask) * image + mask * processed

    def _pick_synthetic_mode(self, rng):
        modes = list(self.synthetic_mode_probs.keys())
        probs = np.array([self.synthetic_mode_probs[key] for key in modes], dtype=np.float64)
        probs = probs / probs.sum()
        return modes[rng.choice(len(modes), p=probs)]

    def _apply_synthetic_anomaly(self, image_name, image_64, image_224, source_name, source_64, source_224, rng):
        mask_64 = self._build_synthetic_mask(self.image_size, rng)
        mode = self._pick_synthetic_mode(rng)

        if mode == "copy_paste":
            anomaly_64 = self._copy_paste(image_64, source_64, mask_64)
        elif mode == "intensity_shift":
            anomaly_64 = self._intensity_shift(image_64, mask_64, rng)
        elif mode == "blur_or_sharpen":
            anomaly_64 = self._blur_or_sharpen(image_64, mask_64, rng)
        else:
            raise ValueError("Unsupported synthetic anomaly mode: {}".format(mode))

        scale_factor = float(self.clip_image_size) / float(self.image_size)
        mask_224 = F.interpolate(mask_64.unsqueeze(0), size=(self.clip_image_size, self.clip_image_size), mode="nearest")
        mask_224 = mask_224.squeeze(0)

        if mode == "copy_paste":
            anomaly_224 = self._copy_paste(image_224, source_224, mask_224)
        elif mode == "intensity_shift":
            anomaly_224 = self._intensity_shift(image_224, mask_224, rng)
        elif mode == "blur_or_sharpen":
            anomaly_224 = self._blur_or_sharpen(image_224, mask_224, rng)
        else:
            upsampled = F.interpolate(anomaly_64.unsqueeze(0), size=(self.clip_image_size, self.clip_image_size), mode="bilinear", align_corners=False)
            anomaly_224 = upsampled.squeeze(0).repeat(3, 1, 1)

        del scale_factor
        return anomaly_64.clamp(-1.0, 1.0), anomaly_224.clamp(-1.0, 1.0), mask_64

    def __getitem__(self, index):
        record, base_label = self.records[index]
        normalized_record = _coerce_record(record)
        img_name = _record_image_name(record)
        rng = self._make_rng(index)
        hidden_label = normalized_record.get("_hidden_label")

        image_64 = self._load_grayscale_tensor(img_name, self.image_size)
        image_224 = self._load_clip_tensor(img_name) if self.include_clip else torch.empty(0, dtype=image_64.dtype)
        mask_syn = torch.zeros(1, self.image_size, self.image_size, dtype=torch.long)
        mask_real = torch.zeros(1, self.image_size, self.image_size, dtype=torch.long)
        has_real_mask = False
        label_img = int(base_label)

        if self.synthetic_enabled and rng.rand() < self.synthetic_probability:
            source_name = self._sample_source_name(index, rng)
            source_64 = self._load_grayscale_tensor(source_name, self.image_size)
            if self.include_clip:
                source_224 = self._load_clip_tensor(source_name)
                image_64, image_224, mask_float = self._apply_synthetic_anomaly(
                    img_name,
                    image_64,
                    image_224,
                    source_name,
                    source_64,
                    source_224,
                    rng,
                )
            else:
                mask_float = self._build_synthetic_mask(self.image_size, rng)
                mode = self._pick_synthetic_mode(rng)
                if mode == "copy_paste":
                    image_64 = self._copy_paste(image_64, source_64, mask_float)
                elif mode == "intensity_shift":
                    image_64 = self._intensity_shift(image_64, mask_float, rng)
                elif mode == "blur_or_sharpen":
                    image_64 = self._blur_or_sharpen(image_64, mask_float, rng)
                else:
                    raise ValueError("Unsupported synthetic anomaly mode: {}".format(mode))
            mask_syn = (mask_float > 0).long()
            label_img = 1
        else:
            real_mask_name = _record_mask_name(record)
            if real_mask_name is not None:
                mask_real = self._load_binary_mask_tensor(real_mask_name, self.image_size)
                has_real_mask = True

        sample = {
            "image_64": image_64,
            "image_224": image_224,
            "label_img": torch.tensor(label_img, dtype=torch.long),
            "mask_syn": mask_syn,
            "mask_real": mask_real,
            "has_real_mask": torch.tensor(has_real_mask, dtype=torch.bool),
            "img_id": _record_img_id(record),
            "group_id": _record_group_id(record, group_key=self.group_key),
            "image_name": img_name,
            "hidden_label": torch.tensor(-1 if hidden_label is None else int(hidden_label), dtype=torch.long),
        }
        return sample


class PseudoLabelDataset(data.Dataset):
    def __init__(
        self,
        main_path,
        manifest_path,
        image_size=64,
        clip_image_size=336,
        cache_images=True,
        cache_clip_images=None,
    ):
        super(PseudoLabelDataset, self).__init__()
        self.root = main_path
        self.image_size = int(image_size)
        self.clip_image_size = int(clip_image_size)
        self.cache_images = bool(cache_images)
        self.cache_clip_images = bool(self.cache_images if cache_clip_images is None else cache_clip_images)
        self.images_dir = os.path.join(self.root, "images")
        self._grayscale_cache = {}
        self._clip_cache = {}
        self.samples = []

        def row_float(row, key, default=0.0):
            value = row.get(key)
            if value in {None, ""}:
                return float(default)
            return float(value)

        def row_int(row, key, default=-1):
            value = row.get(key)
            if value in {None, ""}:
                return int(default)
            return int(value)

        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image_name") in {None, ""}:
                    raise KeyError("Pseudo-label manifest must contain an 'image_name' column.")
                self.samples.append({
                    "img_id": str(row.get("img_id") or os.path.splitext(str(row["image_name"]))[0]),
                    "image_name": str(row["image_name"]),
                    "group_id": str(row.get("group_id") or row.get("img_id") or os.path.splitext(str(row["image_name"]))[0]),
                    "pseudo_target": float(row["pseudo_target"]),
                    "pseudo_weight": float(row["pseudo_weight"]),
                    "pseudo_kind": str(row.get("pseudo_kind") or ""),
                    "teacher_score": row_float(row, "teacher_score", row["pseudo_target"]),
                    "ddad_score": row_float(row, "ddad_score", 0.0),
                    "ddad_percentile": row_float(row, "ddad_percentile", 0.0),
                    "refined_ddad_score": row_float(row, "refined_ddad_score", 0.0),
                    "refined_ddad_percentile": row_float(row, "refined_ddad_percentile", 0.0),
                    "abnormal_joint_score": row_float(row, "abnormal_joint_score", 0.0),
                    "normal_joint_score": row_float(row, "normal_joint_score", 0.0),
                    "localization_confidence": row_float(row, "localization_confidence", 0.0),
                    "background_consistency": row_float(row, "background_consistency", 0.0),
                    "hidden_label": row_int(row, "hidden_label", -1),
                })

        if len(self.samples) == 0:
            raise RuntimeError("Pseudo-label manifest is empty: {}".format(manifest_path))

    def __len__(self):
        return len(self.samples)

    def _load_grayscale_tensor(self, img_name, size):
        cache_key = (img_name, int(size))
        should_cache = self.cache_images and (int(size) != self.clip_image_size or self.cache_clip_images)
        if should_cache and cache_key in self._grayscale_cache:
            return self._grayscale_cache[cache_key].clone()

        with Image.open(os.path.join(self.images_dir, img_name)) as image:
            image = image.convert("L")
            image = image.resize((size, size), resample=Image.BILINEAR)
            image = np.asarray(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        image_tensor = (image_tensor * 2.0 - 1.0).contiguous()
        if should_cache:
            self._grayscale_cache[cache_key] = image_tensor
            return image_tensor.clone()
        return image_tensor

    def _load_clip_tensor(self, img_name):
        if self.cache_clip_images and img_name in self._clip_cache:
            return self._clip_cache[img_name].clone()
        clip_tensor = self._load_grayscale_tensor(img_name, self.clip_image_size).repeat(3, 1, 1).contiguous()
        if self.cache_clip_images:
            self._clip_cache[img_name] = clip_tensor
            return clip_tensor.clone()
        return clip_tensor

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample["image_name"]
        return {
            "image_64": self._load_grayscale_tensor(image_name, self.image_size),
            "image_224": self._load_clip_tensor(image_name),
            "label_img": torch.tensor(1 if sample["pseudo_target"] >= 0.5 else 0, dtype=torch.long),
            "pseudo_target": torch.tensor(sample["pseudo_target"], dtype=torch.float32),
            "pseudo_weight": torch.tensor(sample["pseudo_weight"], dtype=torch.float32),
            "pseudo_kind": str(sample["pseudo_kind"]),
            "teacher_score": torch.tensor(sample["teacher_score"], dtype=torch.float32),
            "ddad_score": torch.tensor(sample["ddad_score"], dtype=torch.float32),
            "ddad_percentile": torch.tensor(sample["ddad_percentile"], dtype=torch.float32),
            "refined_ddad_score": torch.tensor(sample["refined_ddad_score"], dtype=torch.float32),
            "refined_ddad_percentile": torch.tensor(sample["refined_ddad_percentile"], dtype=torch.float32),
            "abnormal_joint_score": torch.tensor(sample["abnormal_joint_score"], dtype=torch.float32),
            "normal_joint_score": torch.tensor(sample["normal_joint_score"], dtype=torch.float32),
            "localization_confidence": torch.tensor(sample["localization_confidence"], dtype=torch.float32),
            "background_consistency": torch.tensor(sample["background_consistency"], dtype=torch.float32),
            "img_id": str(sample["img_id"]),
            "group_id": str(sample["group_id"]),
            "image_name": str(image_name),
            "hidden_label": torch.tensor(int(sample["hidden_label"]), dtype=torch.long),
        }


class CachedFusionDataset(data.Dataset):
    def __init__(self, cache_dir):
        super(CachedFusionDataset, self).__init__()
        self.cache_dir = cache_dir
        if not os.path.isdir(cache_dir):
            raise RuntimeError("Fusion cache directory does not exist: {}".format(cache_dir))

        self.cache_files = sorted(
            [
                os.path.join(cache_dir, file_name)
                for file_name in os.listdir(cache_dir)
                if file_name.endswith(".pt")
            ]
        )
        if len(self.cache_files) == 0:
            raise RuntimeError("No cached fusion feature files found in {}".format(cache_dir))

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, index):
        payload = torch.load(self.cache_files[index], map_location="cpu")
        has_legacy_vector = "fusion_vector" in payload
        has_raw_primitives = "raw_fusion_maps" in payload and "clip_global_logit" in payload
        if not has_legacy_vector and not has_raw_primitives:
            raise RuntimeError(
                "Cached fusion sample is missing both fusion_vector and raw_fusion_maps/clip_global_logit. "
                "Please regenerate fusion cache with the current code."
            )
        if "clip_anchor" not in payload:
            raise RuntimeError(
                "Cached fusion sample is missing clip_anchor. Please regenerate fusion cache with the current code."
            )

        sample = {
            "clip_anchor": payload["clip_anchor"].float(),
            "label_img": payload["label_img"].long(),
            "img_id": str(payload["img_id"]),
        }
        if has_legacy_vector:
            sample["fusion_vector"] = payload["fusion_vector"].float()
        if "raw_fusion_maps" in payload:
            sample["raw_fusion_maps"] = payload["raw_fusion_maps"].float()
        if "clip_global_logit" in payload:
            sample["clip_global_logit"] = payload["clip_global_logit"].float()
        return sample
