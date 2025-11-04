# Copyright © Niantic, Inc. 2022.

import logging
import math
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

_logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerACE:
    def __init__(self, options):
        self.options = options

        self.device = torch.device('cuda')

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.dataset = CamLocDataset(
            root_dir=self.options.scene / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.options.cluster_idx,    # Optional clustering for Cambridge experiments.
        )

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.options.scene,
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2]))

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            intrinsics_fusion_mode=getattr(self.options, 'intrinsics_fusion', 'add'),
        )
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Freeze the pretrained encoder (backbone) so that only the fusion module and head are
        # optimised during training. This keeps feature extraction fixed while allowing the
        # intrinsics fusion layers to adapt.
        for param in self.regressor.encoder.parameters():
            param.requires_grad_(False)
        self.regressor.encoder.eval()

        # Buffer chunk configuration controls how often we refresh the feature buffer so that
        # newly optimised fusion weights are reflected in future training batches.
        preferred_chunk_size = getattr(self.options, "buffer_chunk_size", None)
        self.auto_buffer_chunk_size = None
        if preferred_chunk_size is None or preferred_chunk_size <= 0:
            self.auto_buffer_chunk_size = self._recommend_buffer_chunk_size(
                int(self.options.training_buffer_size),
                int(self.options.batch_size),
            )
            preferred_chunk_size = self.auto_buffer_chunk_size

        self.buffer_chunk_sizes = self._build_buffer_chunk_schedule(preferred_chunk_size)
        self.total_buffer_chunks = len(self.buffer_chunk_sizes)

        # Setup optimization parameters (only trainable parameters are included).
        trainable_params = [p for p in self.regressor.parameters() if p.requires_grad]
        if not self.buffer_chunk_sizes:
            raise ValueError("Training buffer schedule is empty; check buffer_chunk_size and batch_size options.")

        self.optimizer = optim.AdamW(trainable_params, lr=self.options.learning_rate_min)

        # Setup learning rate scheduler.
        steps_per_epoch = sum(chunk_size // self.options.batch_size for chunk_size in self.buffer_chunk_sizes)
        if steps_per_epoch == 0:
            raise ValueError("Batch size is larger than every configured training buffer chunk.")
        self.steps_per_epoch = steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * steps_per_epoch
        self.iterations_output = 100 # print loss every n iterations, and (optionally) write a visualisation frame

        if self.auto_buffer_chunk_size is not None:
            _logger.info(
                "Auto-selected buffer chunk size: %d (total buffer %d, batch %d, %d chunks per epoch)",
                self.auto_buffer_chunk_size,
                int(self.options.training_buffer_size),
                int(self.options.batch_size),
                self.total_buffer_chunks,
            )

        if self.total_buffer_chunks > 1:
            preview_count = min(3, self.total_buffer_chunks)
            chunk_preview = ", ".join(str(size) for size in self.buffer_chunk_sizes[:preview_count])
            if self.total_buffer_chunks > preview_count:
                chunk_preview += ", ..."
            _logger.info(
                "Training buffer will refresh %d chunks per epoch (first chunks: %s)",
                self.total_buffer_chunks,
                chunk_preview,
            )

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = None

        # Generate video of training process
        if self.options.render_visualization:
            # infer rendering folder from map file name
            target_path = vutil.get_rendering_target_path(
                self.options.render_target_path,
                self.options.output_map_file)
            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.options.render_flipped_portrait,
                self.options.render_map_depth_filter,
                mapping_vis_error_threshold=self.options.render_map_error_threshold)
        else:
            self.ace_visualizer = None

    def _build_buffer_chunk_schedule(self, preferred_chunk_size):
        """Return a list of buffer sizes to process sequentially each epoch."""
        total = int(self.options.training_buffer_size)
        batch = int(self.options.batch_size)

        if preferred_chunk_size is None or preferred_chunk_size <= 0:
            preferred_chunk_size = self._recommend_buffer_chunk_size(total, batch)

        preferred_chunk_size = int(preferred_chunk_size)
        schedule = []

        if preferred_chunk_size >= total:
            schedule.append(total)
            return schedule

        chunk_multiple = max(preferred_chunk_size // batch, 1) * batch

        while total > chunk_multiple:
            schedule.append(chunk_multiple)
            total -= chunk_multiple

        if total > 0:
            if total < batch and schedule:
                schedule[-1] += total
            else:
                schedule.append(total)

        return schedule

    def _recommend_buffer_chunk_size(self, total_size, batch_size):
        """Heuristic chunk sizing that keeps ~2M samples per refresh."""
        target_features_per_chunk = 2_000_000

        if total_size <= target_features_per_chunk:
            return total_size

        approx_chunks = max(1, math.ceil(total_size / target_features_per_chunk))
        chunk_size = math.ceil(total_size / approx_chunks)

        chunk_size = max(batch_size, math.ceil(chunk_size / batch_size) * batch_size)

        return min(chunk_size, total_size)

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:

            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_files,
                self.dataset.rgb_files,
                self.iterations // self.iterations_output + 1,
                self.options.render_camera_z_offset
            )

        creating_buffer_time = 0.
        training_time = 0.

        self.training_start = time.time()

        # Train the regression head and fusion modules, refreshing the buffer after each chunk so
        # that updated fusion parameters influence subsequent feature batches.
        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            epoch_buffer_time = 0.
            epoch_training_time = 0.
            # Iterate once for each planned buffer refill; chunk_idx therefore runs from
            # 1..self.total_buffer_chunks (len(self.buffer_chunk_sizes)).
            for chunk_idx, chunk_size in enumerate(self.buffer_chunk_sizes, start=1):
                buffer_start_time = time.time()
                self.create_training_buffer(chunk_size, chunk_index=chunk_idx)
                buffer_end_time = time.time()
                buffer_duration = buffer_end_time - buffer_start_time
                creating_buffer_time += buffer_duration
                epoch_buffer_time += buffer_duration
                _logger.info(
                    "Filled training buffer chunk %d/%d (size=%d) in %.1fs.",
                    chunk_idx,
                    self.total_buffer_chunks,
                    chunk_size,
                    buffer_duration,
                )

                chunk_training_start = time.time()
                self.run_epoch(chunk_size)
                chunk_training_duration = time.time() - chunk_training_start
                training_time += chunk_training_duration
                epoch_training_time += chunk_training_duration

                # Release the chunk to free GPU memory before the next refill.
                self.training_buffer = None

            _logger.info(
                "Completed epoch %03d/%03d in %.1fs (buffers %.1fs, training %.1fs)",
                self.epoch + 1,
                self.options.epochs,
                time.time() - epoch_start_time,
                epoch_buffer_time,
                epoch_training_time,
            )

        # Save trained model.
        self.save_model()

        end_time = time.time()
        _logger.info(f'Done without errors. '
                     f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {end_time - self.training_start:.1f} seconds.')

        if self.ace_visualizer is not None:

            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                root_dir=self.options.scene / "train",
                mode=0,
                use_half=self.options.use_half,
                image_height=self.options.image_resolution,
                augment=False) # No data augmentation when visualizing the map

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False, # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers)

            self.ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)

    def create_training_buffer(self, buffer_size, chunk_index=None):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1,
                                             drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        if chunk_index is None:
            _logger.info("Starting creation of the training buffer.")
        else:
            _logger.info(
                "Starting creation of the training buffer chunk %d/%d.",
                chunk_index,
                self.total_buffer_chunks,
            )
        progress_bar = None
        if tqdm is not None:
            progress_bar = tqdm(
                total=buffer_size,
                desc="Filling training buffer" if chunk_index is None else f"Buffer chunk {chunk_index}",
                unit="sample",
                leave=False,
            )

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((buffer_size, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((buffer_size, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((buffer_size, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((buffer_size, 3, 3), dtype=torch.float32,
                                          device=self.device)
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            try:
                while buffer_idx < buffer_size:
                    dataset_passes += 1
                    for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:

                        # Copy to device.
                        image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                        image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                        gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                        intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                        intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                        # Compute image features.
                        with autocast(enabled=self.options.use_half):
                            features_BCHW = self.regressor.get_features(image_B1HW, intrinsics_B33)

                        # Dimensions after the network's downsampling.
                        B, C, H, W = features_BCHW.shape

                        # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                        image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                        image_mask_B1HW = image_mask_B1HW.bool()

                        # If the current mask has no valid pixels, continue.
                        if image_mask_B1HW.sum() == 0:
                            continue

                        # Create a tensor with the pixel coordinates of every feature vector.
                        pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                        pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                        pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                        # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                        gt_pose_inv = gt_pose_inv_B44[:, :3]
                        gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                        # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                        intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                        intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                        def normalize_shape(tensor_in):
                            """Bring tensor from shape BxCxHxW to NxC"""
                            return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                        batch_data = {
                            'features': normalize_shape(features_BCHW),
                            'target_px': normalize_shape(pixel_positions_B2HW),
                            'gt_poses_inv': gt_pose_inv,
                            'intrinsics': intrinsics,
                            'intrinsics_inv': intrinsics_inv
                        }

                        # Turn image mask into sampling weights (all equal).
                        image_mask_B1HW = image_mask_B1HW.float()
                        image_mask_N1 = normalize_shape(image_mask_B1HW)

                        # Over-sample according to image mask.
                        features_to_select = self.options.samples_per_image * B
                        features_to_select = min(features_to_select, buffer_size - buffer_idx)

                        # Sample indices uniformly, with replacement.
                        sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                        features_to_select,
                                                        replacement=True,
                                                        generator=self.sampling_generator)

                        # Select the data to put in the buffer.
                        for k in batch_data:
                            batch_data[k] = batch_data[k][sample_idxs]

                        # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                        buffer_offset = buffer_idx + features_to_select
                        for k in batch_data:
                            self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                        buffer_idx = buffer_offset
                        if progress_bar is not None:
                            progress_bar.update(features_to_select)
                        if buffer_idx >= buffer_size:
                            break
            finally:
                if progress_bar is not None:
                    progress_bar.close()

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(
            "Created buffer chunk (%d samples) of %.2fGB with %d passes over the training data.",
            buffer_size,
            buffer_memory,
            dataset_passes,
        )
        self.regressor.train()
        self.regressor.encoder.eval()

    def run_epoch(self, buffer_size):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer['features'][random_batch_indices].contiguous(),
                self.training_buffer['target_px'][random_batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].contiguous()
            )
            self.iteration += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file. When
        # using intrinsic feature fusion, we also need to persist the learned fusion parameters.

        def _to_half_state(state_dict):
            half_state = state_dict.__class__()
            for key, value in state_dict.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    half_state[key] = value.half()
                else:
                    half_state[key] = value
            return half_state

        head_state_dict = _to_half_state(self.regressor.heads.state_dict())
        intrinsics_state_dict = _to_half_state(self.regressor.intrinsics_fusion.state_dict())

        save_state = {
            "heads": head_state_dict,
            "intrinsics_fusion": intrinsics_state_dict,
            "intrinsics_fusion_mode": self.regressor.intrinsics_fusion_mode,
        }

        torch.save(save_state, self.options.output_map_file)
        _logger.info(
            "Saved trained head and intrinsics fusion weights to: %s",
            self.options.output_map_file,
        )
