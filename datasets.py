import os
import json
from pathlib import Path

import h5py
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms_v2
from timm.layers import to_2tuple

from util.transforms import paired_random_crop_resize, paired_resize

__all__ = ['imagenet', 'imagenet100', 'sen2venus']


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
                venus_img = torch.from_numpy(data_file["venus"][sample_idx])
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
                venus_img = self._normalize(venus_img, self.VENUS_MEANS, self.VENUS_STDS)
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


def load_dataset(
    dataset, path, img_size=112, hr_img_size=None, use_hr_img=False, load_both_images=False, train=True, min_crop=0.2, transform=None):
    del transform

    if dataset != 'sen2venus':
        raise Exception(f"The dataset {dataset} is not supported")

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