import os
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms_v2
from timm.layers import to_2tuple

from util.transforms import (
    paired_random_crop_resize,
    paired_resize,
    random_crop_resize_img_and_mask,
    resize_img_and_mask
)

__all__ = ['imagenet', 'imagenet100', 'sen2venus', 'rapidai4eo', 'mados', 'm_cashew_plantation', 'm_sa_crop_type']


NUM_CLASSES = {
    'imagenet': 1000,
    'imagenet100': 100
}

class ImageListDataset(data.Dataset):
    def __init__(self, image_list, label_list, class_desc=None, transform=None):
        from torchvision.datasets.folder import default_loader
        self.loader = default_loader
        data.Dataset.__init__(self)
        self.samples = [(fn, lbl) for fn, lbl in zip(image_list, label_list)]
        self.transform = transform
        self.class_desc = class_desc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def imagenet(data_path, transform, train=True):
    import scipy.io
    meta = scipy.io.loadmat(os.path.join(data_path, 'anno', 'meta.mat'))['synsets']
    synsets = [m[0][1][0] for m in meta]
    descriptions = {m[0][1][0]: m[0][2][0] for m in meta}
    imagenet_ids = {m[0][1][0]: m[0][0][0][0]-1 for m in meta}
    if train:
        dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
        wnids = [fn.split('/')[-2] for fn, lbl in dataset.samples]
        inids = [imagenet_ids[w] for w in wnids]

        dataset.samples = [(fn, lbl) for (fn, _), lbl in zip(dataset.samples, inids)]
        dataset.imgs = dataset.samples
        dataset.targets = inids
    else:
        fns = [f"{data_path}/val/ILSVRC2012_val_{i:08d}.JPEG" for i in range(1,50001)]
        inids = [int(ln.strip())-1 for ln in open(os.path.join(data_path, 'anno', 'ILSVRC2012_validation_ground_truth.txt'))]

        dataset = ImageListDataset(fns, inids, transform=transform)
        dataset.imgs = dataset.samples
        dataset.targets = inids

    dataset.classes = synsets[:1000]
    dataset.descriptions = [descriptions[cls] for cls in dataset.classes]
    return dataset


def imagenet100(data_path, transform, train=True):
    # The list of classes from ImageNet-100, which are randomly sampled from the original ImageNet-1k dataset. This list can also be downloaded at: http://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
    in100_cls_list = ['n02869837','n01749939','n02488291','n02107142','n13037406','n02091831','n04517823','n04589890','n03062245','n01773797','n01735189','n07831146','n07753275','n03085013','n04485082','n02105505','n01983481','n02788148','n03530642','n04435653','n02086910','n02859443','n13040303','n03594734','n02085620','n02099849','n01558993','n04493381','n02109047','n04111531','n02877765','n04429376','n02009229','n01978455','n02106550','n01820546','n01692333','n07714571','n02974003','n02114855','n03785016','n03764736','n03775546','n02087046','n07836838','n04099969','n04592741','n03891251','n02701002','n03379051','n02259212','n07715103','n03947888','n04026417','n02326432','n03637318','n01980166','n02113799','n02086240','n03903868','n02483362','n04127249','n02089973','n03017168','n02093428','n02804414','n02396427','n04418357','n02172182','n01729322','n02113978','n03787032','n02089867','n02119022','n03777754','n04238763','n02231487','n03032252','n02138441','n02104029','n03837869','n03494278','n04136333','n03794056','n03492542','n02018207','n04067472','n03930630','n03584829','n02123045','n04229816','n02100583','n03642806','n04336792','n03259280','n02116738','n02108089','n03424325','n01855672','n02090622']

    def filter_dataset(db, cls_list):
        cls2lbl = {cls: lbl for lbl, cls in enumerate(cls_list)}
        idx = [i for i, lbl in enumerate(db.targets) if db.classes[lbl] in cls2lbl]
        samples = [db.samples[i] for i in idx]
        db.samples = [(fn, cls2lbl[db.classes[lbl]]) for fn, lbl in samples]
        db.targets = [dt[1] for dt in db.samples]
        db.imgs = db.samples
        descriptions = {cls: desc for cls, desc in zip(db.classes, db.descriptions)}
        db.classes = cls_list
        db.descriptions = [descriptions[cls] for cls in cls_list]

    dataset = imagenet(data_path, transform, train=train)
    filter_dataset(dataset, in100_cls_list)
    return dataset


class Sen2Venus(torch.utils.data.Dataset):
    """
    Sen2Venus dataset for self-supervised pretraining with latentMIM.
    Contains paired high-resolution (Venus) and low-resolution (Sentinel-2) satellite images.
    """
    # Normalization statistics computed from the dataset
    VENUS_MEANS = [444.3010, 716.1393, 813.6448, 2605.5037]
    VENUS_STDS = [279.9104, 385.4034, 648.5869, 797.1441]

    SENTINEL2_MEANS = [443.8853, 715.6812, 813.2707, 2603.5852]
    SENTINEL2_STDS = [283.9673, 389.3354, 651.2451, 811.8329]

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        img_size: int = 256,
        hr_img_size: int = None,
        use_hr_image: bool = True,
        load_both_images: bool = False,
        random_crop_resize: bool = True,
        min_crop = 0.2,
        normalize: bool = True,
    ):
        """
        Args:
            hdf5_file: Path to HDF5 file containing Sen2Venus data
            splits_file: Path to JSON file containing train/val/test split indices
            split: Which split to use ('train', 'val', or 'test')
            img_size: Size to resize LR (Sentinel-2) images to
            hr_img_size: Size to resize HR (Venus) images to. If None, same as img_size
            use_hr_image: If True, use HR (Venus) images; if False, use LR (Sentinel-2)
            load_both_images: If True, return both HR and LR images
            random_crop_resize: If True, apply random crop and resize for training
            normalize: If True, normalize images using precomputed statistics
        """
        self.hdf5_file = Path(data_path) / "sen2venus.hdf5"
        self.img_size = img_size
        self.hr_img_size = hr_img_size if hr_img_size is not None else img_size
        self.use_hr_image = use_hr_image
        self.load_both_images = load_both_images
        self.random_crop_resize = random_crop_resize
        self.min_crop = min_crop
        self.normalize = normalize

        # Load split indices from JSON file
        splits_file = Path(data_path) / "splits_v1.json"
        with open(splits_file, "r") as f:
            self.indices = json.load(f)[split]

    def __len__(self) -> int:
        return len(self.indices)

    def _paired_resize(self, venus_img, sentinel2_img):
        if self.random_crop_resize:
            return paired_random_crop_resize(
                hr_img=venus_img, lr_img=sentinel2_img,
                size=(self.img_size, self.img_size),
                hr_img_size=(self.hr_img_size, self.hr_img_size),
                scale=(self.min_crop, 1.0)
            )
        else:
            return paired_resize(
                hr_img=venus_img, lr_img=sentinel2_img,
                size=(self.img_size, self.img_size),
                hr_img_size=(self.hr_img_size, self.hr_img_size)
            )

    def _resize(self, img):
        """Apply transformation to a single image."""
        img_size = to_2tuple(self.img_size)

        if self.random_crop_resize:
            return transforms_v2.RandomResizedCrop(size=img_size, scale=(self.min_crop, 1.0))(img)
        else:
            return transforms_v2.Resize(size=img_size)(img)

    def _normalize(self, img, means, stds):
        img = (img - torch.tensor(means).view(-1, 1, 1)) / torch.tensor(stds).view(-1, 1, 1)
        return img

    def __getitem__(self, idx: int):
        # -- load images
        venus_img = None
        sentinel2_img = None
        with h5py.File(self.hdf5_file, "r") as data_file:
            sample_idx = self.indices[idx]
            if self.load_both_images:
                venus_img = torch.from_numpy(data_file["sentinel2"][sample_idx])
                sentinel2_img = torch.from_numpy(data_file["sentinel2"][sample_idx])
            elif self.use_hr_image:
                venus_img = torch.from_numpy(data_file["venus"][sample_idx])
            else:
                sentinel2_img = torch.from_numpy(data_file["sentinel2"][sample_idx])

        # -- resize images
        if self.load_both_images:
            venus_img, sentinel2_img = self._paired_resize(venus_img, sentinel2_img)
        elif self.use_hr_image:
            venus_img = self._resize(venus_img)
        else:
            sentinel2_img = self._resize(sentinel2_img)

        if venus_img is not None:
            venus_img = venus_img.float()
        if sentinel2_img is not None:
            sentinel2_img = sentinel2_img.float()

        # -- normalize
        if self.normalize:
            if venus_img is not None:
                venus_img = self._normalize(venus_img, self.SENTINEL2_MEANS, self.SENTINEL2_STDS)  # TODO!: Revert after experiment
            if sentinel2_img is not None:
                sentinel2_img = self._normalize(sentinel2_img, self.SENTINEL2_MEANS, self.SENTINEL2_STDS)

        if self.load_both_images:
            return venus_img, sentinel2_img, 0
        elif self.use_hr_image:
            return venus_img, 0
        else:
            return sentinel2_img, 0


def sen2venus(data_path: str, train: bool = True,
              img_size: int = 256, hr_img_size: int = None, use_hr_img: bool = True,
              load_both_images: bool = False, normalize: bool = True, min_crop=0.2):
    split = "train" if train else "val"
    dataset = Sen2Venus(
        data_path=data_path,
        split=split,
        img_size=img_size,
        hr_img_size=hr_img_size,
        use_hr_image=use_hr_img,
        load_both_images=load_both_images,
        random_crop_resize=train,  # Apply random crop/resize only during training
        min_crop=min_crop,
        normalize=normalize,
    )
    return dataset


class RapidAI4EO(torch.utils.data.Dataset):
    """
    RapidAI4EO dataset for self-supervised pretraining with latentMIM.
    Contains paired high-resolution (Planet, 200x200) and low-resolution (Sentinel-2, 60x60) satellite images.

    HDF5 structure:
        - sentinel2: (N, 12, 60, 60) - 12 bands at 10m resolution
        - planet: (N, 4, 200, 200) - 4 bands (BGRN) at 3m resolution
        - sample_ids: sample identifiers
        - dates: acquisition dates
    """
    # Normalization statistics (computed from dataset)
    # Sentinel-2: first 4 bands (B, G, R, NIR)
    SENTINEL2_MEANS = [557.5, 828.4, 900.5, 2652.0]
    SENTINEL2_STDS = [396.1, 477.6, 665.9, 946.3]

    # Planet: 4 bands (BGRN)
    PLANET_MEANS = [528.3, 744.1, 849.1, 2692.8]
    PLANET_STDS = [336.2, 418.1, 607.8, 886.1]

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        img_size: int = 60,
        hr_img_size: int = 200,
        use_hr_image: bool = True,
        load_both_images: bool = False,
        random_crop_resize: bool = True,
        min_crop: float = 0.2,
        normalize: bool = True,
    ):
        """
        Args:
            data_path: Path to directory containing rapidai4eo.h5 and rapidai4eo_splits.json
            split: Which split to use ('train', 'val', or 'test')
            img_size: Size to resize LR (Sentinel-2) images to
            hr_img_size: Size to resize HR (Planet) images to
            use_hr_image: If True, use HR (Planet) images; if False, use LR (Sentinel-2)
            load_both_images: If True, return both HR and LR images
            random_crop_resize: If True, apply random crop and resize for training
            min_crop: Minimum crop scale for random resized crop
            normalize: If True, normalize images using precomputed statistics
        """
        self.hdf5_file = Path(data_path) / "rapidai4eo_v1.h5"
        self.img_size = img_size
        self.hr_img_size = hr_img_size
        self.use_hr_image = use_hr_image
        self.load_both_images = load_both_images
        self.random_crop_resize = random_crop_resize
        self.min_crop = min_crop
        self.normalize = normalize

        # Load split indices from JSON file
        splits_file = Path(data_path) / "rapidai4eo_splits.json"
        with open(splits_file, "r") as f:
            self.indices = json.load(f)[split]

    def __len__(self) -> int:
        return len(self.indices)

    def _paired_resize(self, planet_img, sentinel2_img):
        """Resize both images with paired random crop."""
        if self.random_crop_resize:
            return paired_random_crop_resize(
                hr_img=planet_img, lr_img=sentinel2_img,
                size=(self.img_size, self.img_size),
                hr_img_size=(self.hr_img_size, self.hr_img_size),
                scale=(self.min_crop, 1.0)
            )
        else:
            return paired_resize(
                hr_img=planet_img, lr_img=sentinel2_img,
                size=(self.img_size, self.img_size),
                hr_img_size=(self.hr_img_size, self.hr_img_size)
            )

    def _resize(self, img):
        """Apply transformation to a single image."""
        size = to_2tuple(self.img_size)

        if self.random_crop_resize:
            return transforms_v2.RandomResizedCrop(size=size, scale=(self.min_crop, 1.0))(img)
        else:
            return transforms_v2.Resize(size=size)(img)

    def _normalize(self, img, means, stds):
        """Normalize image using provided statistics."""
        img = (img - torch.tensor(means).view(-1, 1, 1)) / torch.tensor(stds).view(-1, 1, 1)
        return img

    def __getitem__(self, idx: int):
        # Load images
        planet_img = None
        sentinel2_img = None

        with h5py.File(self.hdf5_file, "r") as data_file:
            sample_idx = self.indices[idx]
            if self.load_both_images:
                planet_img = torch.from_numpy(data_file["sentinel2"][sample_idx][:4].astype(np.float32))  # todo: revert after experiment
                sentinel2_img = torch.from_numpy(data_file["sentinel2"][sample_idx][:4].astype(np.float32))
            elif self.use_hr_image:
                planet_img = torch.from_numpy(data_file["planet"][sample_idx].astype(np.float32))
            else:
                sentinel2_img = torch.from_numpy(data_file["sentinel2"][sample_idx][:4].astype(np.float32))

        # Resize images
        if self.load_both_images:
            planet_img, sentinel2_img = self._paired_resize(planet_img, sentinel2_img)
        elif self.use_hr_image:
            planet_img = self._resize(planet_img)
        else:
            sentinel2_img = self._resize(sentinel2_img)

        # Normalize
        if self.normalize:
            if planet_img is not None:
                planet_img = self._normalize(planet_img, self.SENTINEL2_MEANS, self.SENTINEL2_STDS)  # todo: revert after experiment
            if sentinel2_img is not None:
                sentinel2_img = self._normalize(sentinel2_img, self.SENTINEL2_MEANS, self.SENTINEL2_STDS)

        if self.load_both_images:
            return planet_img, sentinel2_img, 0
        elif self.use_hr_image:
            return planet_img, 0
        else:
            return sentinel2_img, 0


def rapidai4eo(data_path: str, train: bool = True,
               img_size: int = 60, hr_img_size: int = 200, use_hr_img: bool = True,
               load_both_images: bool = False, normalize: bool = True, min_crop: float = 0.2):
    """
    Factory function for RapidAI4EO dataset.

    Args:
        data_path: Path to directory containing rapidai4eo.h5 and rapidai4eo_splits.json
        train: If True, use training split; else validation
        img_size: Size for Sentinel-2 images (default 60)
        hr_img_size: Size for Planet images (default 200)
        use_hr_img: If True, use Planet (HR); if False, use Sentinel-2 (LR)
        load_both_images: If True, return both Planet and Sentinel-2
        normalize: If True, normalize using precomputed statistics
        min_crop: Minimum crop scale for random resized crop

    Returns:
        RapidAI4EO dataset instance
    """
    split = "train" if train else "val"
    dataset = RapidAI4EO(
        data_path=data_path,
        split=split,
        img_size=img_size,
        hr_img_size=hr_img_size,
        use_hr_image=use_hr_img,
        load_both_images=load_both_images,
        random_crop_resize=train,
        min_crop=min_crop,
        normalize=normalize,
    )
    return dataset


def load_dataset(
    dataset, path, img_size=112, hr_img_size=None, use_hr_img=False, load_both_images=False, train=True, min_crop=0.2, transform=None):
    del transform

    if dataset == 'sen2venus':
        return sen2venus(
            data_path=path,
            train=train,
            img_size=img_size,
            hr_img_size=hr_img_size,
            use_hr_img=use_hr_img,
            load_both_images=load_both_images,
            normalize=True,
            min_crop=min_crop
        )
    elif dataset == 'rapidai4eo':
        return rapidai4eo(
            data_path=path,
            train=train,
            img_size=img_size,
            hr_img_size=hr_img_size,
            use_hr_img=use_hr_img,
            load_both_images=load_both_images,
            normalize=True,
            min_crop=min_crop
        )
    else:
        raise Exception(f"The dataset {dataset} is not supported")


from util import misc
from torch.utils import data
def create_loader(dataset, batch_size, num_workers=0, num_tasks=1, global_rank=0, pin_memory=True, drop_last=True):
    if misc.is_dist_avail_and_initialized():
        sampler_train = data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = data.RandomSampler(dataset)

    loader = data.DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader


class MADOSDataset(data.Dataset):
    """
    MADOS dataset.
    Used for evaluating learned representations on semantic segmentation tasks.
    """

    def __init__(self, data_path: str, split: str = "train", img_size: int = 80,
                 normalize: bool = True, filter_bands: list = [1, 2, 3, 6]):
        """
        Args:
            data_path: Path to directory containing mados.h5 and NORM_CONFIG.json
            split: Which split to use ('train', 'val', or 'test')
            img_size: Size to resize images to
            normalize: If True, normalize images using precomputed statistics
            filter_bands: If provided, only keep these band indices (e.g., [1, 2, 3, 6] for BGR+NIR)
        """
        self.h5_path = os.path.join(data_path, 'mados.h5')
        self.split = split
        self.img_size = to_2tuple(img_size)
        self.normalize = normalize
        self.filter_bands = filter_bands

        # Load normalization config
        norm_stats_path = os.path.join(data_path, 'NORM_CONFIG.json')
        self.norm_config = json.load(open(norm_stats_path, "r"))

        assert split in ["train", "val", "test"], \
            f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"

        # Load split indices (open and close file immediately to avoid pickling issues)
        with h5py.File(self.h5_path, "r") as f:
            all_splits = f["split"][:]
            self.indices = np.where(np.isin(all_splits, [split.encode('utf-8')]))[0]

    def __len__(self):
        return len(self.indices)

    def _normalize_bands(self, image):
        """Normalize image bands using mean and std from config."""
        means = torch.tensor(self.norm_config["mean"]).reshape(-1, 1, 1)
        stds = torch.tensor(self.norm_config["std"]).reshape(-1, 1, 1)
        return (image - means) / stds

    def _resize_img_and_mask(self, img, mask):
        """Resize image and mask with proper interpolation modes."""
        if self.split == "train":
            # Random crop and resize for training
            # Use NEAREST for mask to preserve label values
            return random_crop_resize_img_and_mask(img, mask, self.img_size, scale=(0.3, 1.0), ratio=(3/4, 4/3))
        else:
            # Just resize for val/test
            # Use NEAREST for mask to preserve label values
            return resize_img_and_mask(img, mask, self.img_size)

    def __getitem__(self, idx):
        # Open HDF5 file for each access (fast and avoids pickling issues)
        with h5py.File(self.h5_path, "r") as h5_file:
            # Load image and label
            image = torch.from_numpy(h5_file["images"][self.indices[idx]])  # (C, H, W)
            label = torch.from_numpy(h5_file["label"][self.indices[idx]]).long()  # (H, W)

        # Handle NaN pixels
        nan_mask = torch.isnan(image).any(dim=0)
        image = torch.where(torch.isnan(image), 0.0, image)
        label = torch.where(nan_mask, -1, label)

        # Set 0 (no-data labels) to -1 (ignored index)
        label = torch.where(label == 0, -1, label)
        # Shift labels 1-15 to 0-14 (only for non-ignored labels)
        label = torch.where(label > 0, label - 1, label)

        # Add channel dim to label
        label = label.unsqueeze(0)  # (1, H, W)

        # Normalize
        if self.normalize:
            image = self._normalize_bands(image)

        # Resize image and mask
        image, label = self._resize_img_and_mask(image, label)

        # Filter bands if specified
        if self.filter_bands is not None:
            image = image[self.filter_bands]

        # Ensure label is long type (resize operations may convert to float)
        label = label.long()

        return image, label


def mados(data_path: str, split: str = "train", img_size: int = 80,
          normalize: bool = True, filter_bands: list = None):
    """
    Create MADOS dataset for evaluation.

    Args:
        data_path: Path to directory containing mados.h5 and NORM_CONFIG.json
        split: Which split to use ('train', 'val', or 'test')
        img_size: Size to resize images to
        normalize: If True, normalize images using precomputed statistics
        filter_bands: If provided, only keep these band indices (e.g., [1, 2, 3, 6] for BGR+NIR)

    Returns:
        MADOSDataset instance
    """
    dataset = MADOSDataset(
        data_path=data_path,
        split=split,
        img_size=img_size,
        normalize=normalize,
        filter_bands=filter_bands
    )
    return dataset


class GeobenchDataset(data.Dataset):
    """
    Geobench dataset wrapper for segmentation tasks.
    Supports m-cashew-plantation and m-SA-crop-type datasets.

    Logical correspondence to ijepa GeobenchDataset (lines 233-316):
    - Uses geobench library to load data
    - Applies normalization using dataset statistics
    - Resizes with proper interpolation (NEAREST for masks)
    - Returns (image, mask) tuples
    """

    def __init__(
        self,
        split: str,
        partition: str,
        dataset_name: str,
        data_path: str = None,
        norm_operation: str = 'standardize',
        benchmark_name: str = "segmentation_v0.9.1",
        band_names: list = None,
        img_size: int = 64
    ):
        """
        Args:
            split: 'train', 'valid', or 'test'
            partition: Partition name (e.g., "1.00x_train", "0.50x_train")
            dataset_name: 'm-cashew-plantation' or 'm-SA-crop-type'
            data_path: Path to geobench data directory (if None, uses $GEO_BENCH_DIR)
            norm_operation: 'standardize', 'norm_yes_clip', 'norm_no_clip', or 'satlas'
            benchmark_name: Geobench benchmark name
            band_names: List of band names to use (default: Blue, Green, Red, NIR)
            img_size: Size to resize images to
        """
        import geobench
        import os

        assert split in ["train", "valid", "test"], \
            f"split must be 'train', 'valid', or 'test', not {split}"
        assert dataset_name in ["m-SA-crop-type", "m-cashew-plantation"], \
            f"dataset_name must be 'm-SA-crop-type' or 'm-cashew-plantation', not {dataset_name}"

        self.split = split
        self.partition = partition
        self.dataset_name = dataset_name
        self.norm_operation = norm_operation
        self.img_size = to_2tuple(img_size)

        # Default to Sentinel-2 BGR+NIR bands (same as ijepa line 246)
        if band_names is None:
            band_names = ["02 - Blue", "03 - Green", "04 - Red", "08 - NIR"]
        self.band_names = band_names

        # Set benchmark directory if data_path is provided
        if data_path is not None:
            benchmark_dir = os.path.join(data_path, benchmark_name)
        else:
            benchmark_dir = None

        # Load geobench task (same as ijepa lines 263-265)
        for task in geobench.task_iterator(benchmark_name=benchmark_name, benchmark_dir=benchmark_dir):
            if task.dataset_name == dataset_name:
                break

        # Get dataset from task (same as ijepa line 267)
        self.dataset = task.get_dataset(
            split=self.split,
            partition_name=self.partition,
            band_names=band_names
        )

        print(f"Loaded {dataset_name} {split} set with partition {partition}: {len(self.dataset)} samples")

        # Get band indices (same as ijepa lines 272-276)
        original_band_names = [
            self.dataset[0].bands[i].band_info.name
            for i in range(len(self.dataset[0].bands))
        ]
        self.band_indices = [original_band_names.index(name) for name in self.band_names]

        # Cache normalization stats (same as ijepa line 279)
        self.norm_stats = self.dataset.normalization_stats()

    def __len__(self):
        return len(self.dataset)

    def _normalize_bands(self, image):
        """
        Normalize bands using dataset statistics.
        Corresponds to ijepa normalize_bands function (lines 319-351).
        """
        if self.norm_operation == "satlas":
            image = image / 8160
            image = np.clip(image, 0, 1)
            return image

        means, stds = self.norm_stats[0], self.norm_stats[1]
        means = np.array(means)
        stds = np.array(stds)

        if self.norm_operation == "standardize":
            image = (image - means) / stds
        else:
            min_value = means - stds
            max_value = means + stds
            image = (image - min_value) / (max_value - min_value)

            if self.norm_operation == "norm_yes_clip":
                image = np.clip(image, 0, 1)
            elif self.norm_operation == "norm_no_clip":
                pass
            else:
                raise ValueError(
                    f"norm_operation must be 'standardize', 'norm_yes_clip', 'norm_no_clip', or 'satlas', not {self.norm_operation}"
                )

        return image

    def __getitem__(self, idx):
        """
        Load and preprocess a sample.
        Corresponds to ijepa __getitem__ (lines 281-313).
        """
        # Load sample from geobench (same as ijepa lines 282-283)
        sample = self.dataset[idx]
        label = sample.label

        # Load bands (same as ijepa line 286)
        # Stack selected bands: (H, W, C)
        image = np.stack([sample.bands[band_idx].data for band_idx in self.band_indices], axis=2)
        assert image.shape[-1] == len(self.band_names), \
            f"Expected {len(self.band_names)} channels, got {image.shape[-1]}"

        # Normalize (same as ijepa line 288)
        image = torch.tensor(self._normalize_bands(image))

        # Handle label (same as ijepa lines 291-296)
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            label = np.array(list(label))

        target = torch.tensor(label, dtype=torch.long)

        # Permute to channel-first: (C, H, W) (same as ijepa line 302)
        image = image.permute(2, 0, 1)
        # Add channel dimension to target: (1, H, W) (same as ijepa line 303)
        target = target.unsqueeze(0)

        # Resize with proper interpolation (same as ijepa lines 305-309)
        # CRITICAL: Use NEAREST interpolation for masks to preserve labels
        if self.split == 'train':
            image, target = random_crop_resize_img_and_mask(
                img=image, mask=target, size=self.img_size,
                scale=(0.3, 1.0), ratio=(3/4, 4/3)
            )
        else:
            image, target = resize_img_and_mask(
                img=image, mask=target, size=self.img_size
            )

        # Convert to float (same as ijepa line 311)
        image = image.float()

        return image, target


def m_cashew_plantation(
    split: str = "train",
    partition: str = "default",
    data_path: str = None,
    norm_operation: str = 'standardize',
    band_names: list = None,
    img_size: int = 64
):
    """
    Create m-cashew-plantation dataset.

    Corresponds to ijepa make_m_cashew_plant_dataset (lines 25-126).
    Returns dataset instance (dataloader creation handled separately).

    Args:
        split: 'train', 'valid', or 'test'
        partition: Partition name (e.g., "1.00x_train", "0.50x_train", "0.20x_train")
        data_path: Path to geobench data directory (if None, uses $GEO_BENCH_DIR)
        norm_operation: Normalization operation ('standardize', 'norm_yes_clip', etc.)
        band_names: List of Sentinel-2 band names (default: Blue, Green, Red, NIR)
        img_size: Target image size

    Returns:
        GeobenchDataset instance
    """
    if band_names is None:
        band_names = ["02 - Blue", "03 - Green", "04 - Red", "08 - NIR"]

    dataset = GeobenchDataset(
        split=split,
        partition=partition,
        dataset_name="m-cashew-plantation",
        data_path=data_path,
        norm_operation=norm_operation,
        benchmark_name="segmentation_v0.9.1",
        band_names=band_names,
        img_size=img_size
    )

    return dataset


def m_sa_crop_type(
    split: str = "train",
    partition: str = "default",
    data_path: str = None,
    norm_operation: str = 'standardize',
    band_names: list = None,
    img_size: int = 64
):
    """
    Create m-SA-crop-type dataset.

    Corresponds to ijepa make_m_sa_crop_type_dataset (lines 129-230).
    Returns dataset instance (dataloader creation handled separately).

    Args:
        split: 'train', 'valid', or 'test'
        partition: Partition name (e.g., "1.00x_train", "0.50x_train", "0.20x_train")
        data_path: Path to geobench data directory (if None, uses $GEO_BENCH_DIR)
        norm_operation: Normalization operation ('standardize', 'norm_yes_clip', etc.)
        band_names: List of Sentinel-2 band names (default: Blue, Green, Red, NIR)
        img_size: Target image size

    Returns:
        GeobenchDataset instance
    """
    if band_names is None:
        band_names = ["02 - Blue", "03 - Green", "04 - Red", "08 - NIR"]

    dataset = GeobenchDataset(
        split=split,
        partition=partition,
        dataset_name="m-SA-crop-type",
        data_path=data_path,
        norm_operation=norm_operation,
        benchmark_name="segmentation_v0.9.1",
        band_names=band_names,
        img_size=img_size
    )

    return dataset