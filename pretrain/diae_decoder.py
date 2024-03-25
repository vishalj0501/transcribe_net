import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit_pytorch.vit import Transformer
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class DIAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        image_size,
        patch_size,
        dim,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,

    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        channels = 3
        patch_height, patch_width = pair(patch_size)
        image_height, image_width = pair(image_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.deg_patch_to_emb = nn.Linear(patch_dim, dim)
        self.blur_patch_to_emb = nn.Linear(patch_dim, dim)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]

        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        
        
        self.clean_to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.deg_to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.blur_to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)


    def forward(self, img, dist_img , dist_img_blur):
        device = img.device

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]


        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        masked_patches = patches[batch_range, masked_indices]
        encoded_tokens = self.encoder.transformer(tokens)

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.clean_to_pixels(mask_tokens)

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        dist_patches = self.to_patch(dist_img)
        dist_batch, dist_num_patches, *_ = dist_patches.shape

        dist_tokens = self.deg_patch_to_emb(dist_patches)

        dist_tokens = dist_tokens + self.encoder.pos_embedding[:, 1:(dist_num_patches + 1)]

        dist_encoded_tokens = self.encoder.transformer(dist_tokens)

        dist_decoder_tokens = self.enc_to_dec(dist_encoded_tokens)

        dist_decoded_tokens = self.decoder(dist_decoder_tokens)

        dist_pred_pixel_values = self.deg_to_pixels(dist_decoded_tokens)
        
        enh_loss = F.mse_loss(dist_pred_pixel_values, patches)

        dist_patches_blur = self.to_patch(dist_img_blur)
        dist_batch, dist_num_patches, *_ = dist_patches.shape

        dist_tokens_blur = self.deg_patch_to_emb(dist_patches_blur)

        dist_tokens_blur = dist_tokens_blur + self.encoder.pos_embedding[:, 1:(dist_num_patches + 1)]

        dist_encoded_tokens_blur = self.encoder.transformer(dist_tokens_blur)


        dist_decoder_tokens_blur = self.enc_to_dec(dist_encoded_tokens_blur)

        dist_decoded_tokens_blur = self.decoder(dist_decoder_tokens_blur)

        dist_pred_pixel_values_blur = self.blur_to_pixels(dist_decoded_tokens_blur)
        deblur_loss = F.mse_loss(dist_pred_pixel_values_blur, patches)


        return recon_loss, enh_loss, deblur_loss, patches, batch_range, masked_indices, pred_pixel_values, dist_patches 
