from torch import nn
import torch
import torch.nn.functional as F

REGISTERED_MODELS_CLASSES = {}

def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


class ConditionFusionCompressor(nn.Module):
    def __init__(
        self,
        image_channels=64,
        partial_channels=24,
        hidden_channels=44,
        out_channels=16,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.partial_channels = partial_channels
        self.in_channels = image_channels + partial_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        # self.projection = nn.Sequential(
        #     nn.Conv3d(self.in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        #     nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        # )
        self.projection = nn.Sequential(
                nn.Conv3d(self.in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
    @staticmethod
    def _validate_condition(condition, name, expected_channels):
        if condition.dim() != 5:
            raise ValueError(f"{name} condition expects [B, C, W, L, H] input")
        if condition.shape[1] != expected_channels:
            raise ValueError(
                f"{name} condition expected {expected_channels} channels, got {condition.shape[1]}"
            )

    def forward(self, partial_condition, image_condition):
        self._validate_condition(image_condition, "image", self.image_channels)
        self._validate_condition(partial_condition, "partial", self.partial_channels)
        if image_condition.shape[0] != partial_condition.shape[0]:
            raise ValueError("Image and partial condition batch sizes must match")
        if image_condition.shape[2:] != partial_condition.shape[2:]:
            raise ValueError("Image and partial condition spatial shapes must match")

        condition = torch.cat((image_condition.float(), partial_condition.float()), dim=1)
        return self.projection(condition)


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 model_part,
                 sparse_shape,
                 channels_list,
                 num_input_features,
                 init_size,
                 voxel_channel,
                 condition_in_channels=88,
                 image_condition_channels=64,
                 partial_condition_channels=24,
                 condition_mid_channels=44,
                 condition_channels=16,
                 condition_dropout_prob=0.1,
                 guidance_scale=3.0,
                 num_inference_steps=100,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.sparse_shape = sparse_shape
        self.model_part = model_part
        self.condition_in_channels = condition_in_channels
        self.image_condition_channels = image_condition_channels
        self.partial_condition_channels = partial_condition_channels
        self.condition_mid_channels = condition_mid_channels
        self.condition_channels = condition_channels
        self.condition_dropout_prob = condition_dropout_prob
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        expected_condition_in_channels = image_condition_channels + partial_condition_channels
        if condition_in_channels != expected_condition_in_channels:
            raise ValueError(
                f"condition_in_channels must equal image_condition_channels + "
                f"partial_condition_channels ({expected_condition_in_channels}), got {condition_in_channels}"
            )
        self.condition_compressor = ConditionFusionCompressor(
            image_channels=image_condition_channels,
            partial_channels=partial_condition_channels,
            hidden_channels=condition_mid_channels,
            out_channels=condition_channels,
        )

    def compress_conditions(self, partial_condition, image_condition):
        return self.condition_compressor(
            partial_condition,
            image_condition,
        )

    def forward(self, batch_size, val_VAE_features_change, val_partial_features_change, val_image_features_change, train=True):
        val_condition_features_change = self.compress_conditions(
            val_partial_features_change,
            val_image_features_change,
        )

        if train == True:
            noise_complete = torch.randn_like(val_VAE_features_change)
            ###------------------------------------sample a random timestep for each voxel scene-------
            noise_steps = 1000
            timesteps_complete = torch.randint(
                noise_steps,
                (val_VAE_features_change.shape[0],),
                device=val_VAE_features_change.device,
                dtype=torch.long,
            )
            x_t_complete = self.model_part.add_noise(
                val_VAE_features_change,
                timesteps_complete,
                noise_complete,
            ).to(dtype=torch.float32)
            pred_noise = self.model_part.pred_noise(x_t_complete,
                                                    val_condition_features_change,
                                                    timesteps_complete,
                                                    guidance_scale=1.0,
                                                    train=train,
                                                    condition_dropout_prob=self.condition_dropout_prob)

            return noise_complete, pred_noise
        else:
            noise_complete = torch.randn_like(val_VAE_features_change)
            ###------------------------------------sample a random timestep for each voxel scene-------
            noise_steps = 1000
            timesteps_complete = torch.randint(
                noise_steps,
                (val_VAE_features_change.shape[0],),
                device=val_VAE_features_change.device,
                dtype=torch.long,
            )
            x_t_complete = self.model_part.add_noise(
                val_VAE_features_change,
                timesteps_complete,
                noise_complete,
            ).to(dtype=torch.float32)
            pred_noise_cfg, _, pred_noise_conditional = self.model_part.pred_noise(
                x_t_complete,
                val_condition_features_change,
                timesteps_complete,
                guidance_scale=self.guidance_scale,
                train=train,
                return_cfg_components=True,
            )

            loss_val_conditional = F.mse_loss(
                pred_noise_conditional.float(),
                noise_complete.float(),
                reduction="mean",
            )
            loss_val_cfg = F.mse_loss(
                pred_noise_cfg.float(),
                noise_complete.float(),
                reduction="mean",
            )
            x_yuan_recover = self.model_part.sample(noise_complete,
                                                    val_condition_features_change,
                                                    guidance_scale=self.guidance_scale,
                                                    num_inference_steps=self.num_inference_steps,
                                                    train=False)
            x_yuan_recover_raw = self.model_part.denormalize_latent(x_yuan_recover)
            recon_voxel = self.model_part.autoencoder.decode(x_yuan_recover_raw)

            return loss_val_conditional, loss_val_cfg, recon_voxel
