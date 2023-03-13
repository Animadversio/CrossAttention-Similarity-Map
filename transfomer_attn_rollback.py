#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lpips import LPIPS
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data
#%%
Dist = LPIPS(net='squeeze', spatial=True,)
Dist = Dist.cuda().eval()
Dist.requires_grad_(False)
#%%

_, BFEStats = load_neural_data()
#%%
Expi = 155
imgfps_arr0, resp_arr0, bsl_arr0, gen_arr0 = \
    load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="arr")
imgfps_arr1, resp_arr1, bsl_arr1, gen_arr1 = \
    load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="arr")
#%%
imgs_blk0 = [plt.imread(imgfp) for imgfp in imgfps_arr0[-3]]
imgs_blk1 = [plt.imread(imgfp) for imgfp in imgfps_arr1[-3]]
imgtsr_blk0 = torch.tensor(np.stack(imgs_blk0)).permute(0, 3, 1, 2).float() / 255.0
imgtsr_blk1 = torch.tensor(np.stack(imgs_blk1)).permute(0, 3, 1, 2).float() / 255.0
#%%
distmats = Dist.forward_distmat(imgtsr_blk0.cuda(), imgtsr_blk1.cuda()).cpu()
#%%
plt.figure()
plt.imshow(distmats.mean([0,1,2]).detach().numpy())
plt.colorbar()
plt.axis("image")
plt.show()
#%%
show_imgrid(imgtsr_blk0, nrow=8, figsize=(10,10))
show_imgrid(imgtsr_blk1, nrow=8, figsize=(10,10))

#%%
import timm
from timm import create_model
vit_model = create_model("vit_base_patch16_224", pretrained=True)
vit_model.eval()
#%%
img_rsz0 = F.interpolate(imgtsr_blk0, (224, 224), mode="bilinear", align_corners=True)
#%%
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
train_nodes, _ = get_graph_node_names(vit_model)
#%%
vit_model.cuda().requires_grad_(False)
feature_extractor = create_feature_extractor(vit_model, return_nodes={"blocks": "out"})
#%%
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

hugfc_preprocessing = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')
#%%
model.cuda().requires_grad_(False)
#%%
inputs = hugfc_preprocessing(images=list(imgtsr_blk1), return_tensors="pt")["pixel_values"]
with torch.no_grad():
    outputs = model(inputs.cuda()[:10], output_attentions=True, output_hidden_states=False)
last_hidden_states = outputs.last_hidden_state
#%%
attn_weights = torch.stack([attnmat.cpu() for attnmat in outputs.attentions])
del outputs
#%%
head_avg_attn = attn_weights.mean([2])

# head_avg_attn, _ = attn_weights.max(dim=2)
#%%
identity_mat = torch.eye(head_avg_attn.shape[-1]).repeat(head_avg_attn.shape[1], 1, 1)
rollback_mat = identity_mat.clone()  # torch.eye(head_avg_attn.shape[-1]).repeat(head_avg_attn.shape[1], 1, 1)
flow_mat     = identity_mat.clone()  # torch.eye(head_avg_attn.shape[-1]).repeat(head_avg_attn.shape[1], 1, 1)
for layeri in range(12):
    rollback_mat  = torch.bmm((head_avg_attn[layeri] + identity_mat), rollback_mat, )
    flow_mat  = torch.bmm((head_avg_attn[layeri]), flow_mat, )
#%%
cls_rollback_attn_mats = rollback_mat[:, 0, 1:]
cls_rollback_attn_mats = cls_rollback_attn_mats.reshape(-1, 28, 28)
cls_flow_attn_mats = flow_mat[:, 0, 1:]
cls_flow_attn_mats = cls_flow_attn_mats.reshape(-1, 28, 28)
#%%
cls_rollback_attn_img = F.interpolate(cls_rollback_attn_mats[:, None], (224, 224), mode="bilinear", align_corners=True)
#%%
# normalize min, max to 0,1 for visualization
val_min, _ = cls_rollback_attn_img.flatten(1).min(-1)
val_max, _ = cls_rollback_attn_img.flatten(1).max(-1)
cls_rollback_attn_img_norm = (cls_rollback_attn_img - val_min[:,None,None,None]) / \
                             (val_max[:,None,None,None] - val_min[:,None,None,None])

#%%
# cls_rollback_attn_img = (cls_rollback_attn_img - cls_rollback_attn_img.min(dim=(1,2,3), keepdim=True)) / \
#                         (cls_rollback_attn_img.max(dim=(1,2,3), keepdim=True) - cls_rollback_attn_img.min(dim=(1,2,3), keepdim=True))
#%%
show_imgrid(cls_rollback_attn_img_norm, nrow=5, figsize=(10, 10))
#%%
show_imgrid(cls_rollback_attn_img_norm, nrow=4, figsize=(10, 10))
#%%
F.interpolate(cls_rollback_attn_mats, (224, 224), mode="bilinear", align_corners=True).shape
#%%
torch.cuda.empty_cache()
#%%
img_rsz0 = F.interpolate(imgtsr_blk0, (224, 224), mode="bilinear", align_corners=True)
img_rsz1 = F.interpolate(imgtsr_blk1, (224, 224), mode="bilinear", align_corners=True)
with torch.no_grad():
    img_feat0 = feature_extractor(img_rsz0.cuda())["out"].cpu()
    img_feat1 = feature_extractor(img_rsz1.cuda())["out"].cpu()
#%%
with torch.no_grad():
    img_feat0 = vit_model.forward_features(img_rsz0.cuda()).cpu()
    img_feat1 = vit_model.forward_features(img_rsz1.cuda()).cpu()
#%% first token is the cls token
img_feat0_cls = img_feat0[:, :1, :]
img_feat0_patch = img_feat0[:, 1:, :]
img_feat1_cls = img_feat1[:, :1, :]
img_feat1_patch = img_feat1[:, 1:, :]
#%%
mean_feat0_cls = img_feat0_cls.mean([0, ])
mean_feat0_patch = img_feat0_patch.mean([0, ])
mean_feat1_cls = img_feat1_cls.mean([0, ])
mean_feat1_patch = img_feat1_patch.mean([0, ])
#%%
clspatch_dotprod00 = (mean_feat0_patch @ mean_feat0_cls.T).reshape(14, 14)
clspatch_dotprod01 = (mean_feat1_patch @ mean_feat0_cls.T).reshape(14, 14)
clspatch_dotprod10 = (mean_feat0_patch @ mean_feat1_cls.T).reshape(14, 14)
clspatch_dotprod11 = (mean_feat1_patch @ mean_feat1_cls.T).reshape(14, 14)
#%%
figh, axs = plt.subplots(2,2)
axs[0, 0].imshow(clspatch_dotprod00.detach().numpy())
axs[0, 1].imshow(clspatch_dotprod01.detach().numpy())
axs[1, 0].imshow(clspatch_dotprod10.detach().numpy())
axs[1, 1].imshow(clspatch_dotprod11.detach().numpy())
plt.show()

#%%.shae
from timm.models.vision_transformer import VisionTransformer

