"""
Mask utilities for LatentMIM.
"""

import numpy as np
import torch


class RectangularPatchMaskCollator:
    """
    Collator that generates random rectangular patch masks for Sen2Venus/RapidAI4EO datasets.

    The mask indicates which patches fall within a randomly selected rectangular region.
    This is useful for masked image modeling where you want to mask contiguous regions.
    """

    def __init__(
        self,
        grid_size: int = 14,
        min_rect_size: float = 0.3,
        max_rect_size: float = 0.7,
        aspect_ratio_range: tuple = (0.5, 2.0),
    ):
        """
        Args:
            grid_size: Number of patches along each dimension (e.g., 14 for 14x14 grid)
            min_rect_size: Minimum size of rectangle as fraction of grid_size (0.0 to 1.0)
            max_rect_size: Maximum size of rectangle as fraction of grid_size (0.0 to 1.0)
            aspect_ratio_range: (min_ratio, max_ratio) for rectangle aspect ratio (width/height)
        """
        self.grid_size = grid_size
        self.min_rect_size = max(1, int(min_rect_size * grid_size))
        self.max_rect_size = max(1, int(max_rect_size * grid_size))
        self.aspect_ratio_range = aspect_ratio_range

    def _generate_rect_mask(self, batch_size: int, device: torch.device = None):
        """
        Generate random rectangular masks for a batch.

        Args:
            batch_size: Number of samples in the batch
            device: Device to create tensors on

        Returns:
            mask: (B, grid_size, grid_size) boolean tensor where True = inside rectangle
            rect_params: dict with 'top', 'left', 'height', 'width' tensors of shape (B,)
        """
        min_area = self.min_rect_size ** 2
        max_area = self.max_rect_size ** 2

        tops = []
        lefts = []
        heights = []
        widths = []
        masks = []

        for _ in range(batch_size):
            # Sample aspect ratio
            aspect_ratio = np.random.uniform(*self.aspect_ratio_range)

            # Sample area and compute dimensions
            area = np.random.randint(min_area, max_area + 1)

            # width = sqrt(area * aspect_ratio), height = sqrt(area / aspect_ratio)
            width = int(np.clip(np.sqrt(area * aspect_ratio), self.min_rect_size, self.max_rect_size))
            height = int(np.clip(np.sqrt(area / aspect_ratio), self.min_rect_size, self.max_rect_size))

            # Ensure dimensions don't exceed grid
            width = min(width, self.grid_size)
            height = min(height, self.grid_size)

            # Sample top-left corner position
            max_top = self.grid_size - height
            max_left = self.grid_size - width
            top = np.random.randint(0, max_top + 1)
            left = np.random.randint(0, max_left + 1)

            # Create mask for this sample
            mask = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool)
            mask[top:top+height, left:left+width] = True

            tops.append(top)
            lefts.append(left)
            heights.append(height)
            widths.append(width)
            masks.append(mask)

        # Stack into batch tensors
        mask = torch.stack(masks, dim=0)  # (B, grid_size, grid_size)
        if device is not None:
            mask = mask.to(device)

        rect_params = {
            'top': torch.tensor(tops, dtype=torch.long),
            'left': torch.tensor(lefts, dtype=torch.long),
            'height': torch.tensor(heights, dtype=torch.long),
            'width': torch.tensor(widths, dtype=torch.long),
        }

        return mask, rect_params

    def _get_mask_indices(self, mask: torch.Tensor):
        """
        Convert 2D boolean mask to flat indices.

        Args:
            mask: (B, grid_size, grid_size) boolean tensor

        Returns:
            indices: List of tensors, one per sample, containing flat indices of True positions
            num_masked: (B,) tensor with count of masked patches per sample
        """
        batch_size = mask.shape[0]
        flat_mask = mask.view(batch_size, -1)  # (B, grid_size^2)

        indices_list = []
        num_masked = []
        for b in range(batch_size):
            idx = torch.where(flat_mask[b])[0]
            indices_list.append(idx)
            num_masked.append(len(idx))

        return indices_list, torch.tensor(num_masked, dtype=torch.long)

    def __call__(self, batch):
        """
        Collate batch and add rectangular patch mask indices.

        Args:
            batch: List of (image, label) or (hr_image, lr_image, label) tuples

        Returns:
            tuple: (*collated_batch, mask_indices)
                - collated_batch: output of default_collate on the original batch
                - mask_indices: (B, min_keep) tensor of flat indices, truncated to min length in batch
        """
        from torch.utils.data import default_collate

        collated = default_collate(batch)
        batch_size = len(batch)

        # Generate rectangular masks and get indices
        rect_mask, _ = self._generate_rect_mask(batch_size)
        mask_indices, _ = self._get_mask_indices(rect_mask)

        # Truncate all masks to minimum length in batch
        min_keep = min(len(m) for m in mask_indices)
        mask_indices = torch.stack([m[:min_keep] for m in mask_indices], dim=0)

        return *collated, mask_indices
