import torch.nn.functional as F
from torch import nn

from .adapters.adapter import Adapter
from .attn import RecAttnClip
from .clip.clip import load
from .layer import MaskPostXrayProcess, PostClipProcess


class DS(nn.Module):
    def __init__(
        self, clip_name, adapter_vit_name, num_quires, fusion_map, mlp_dim, mlp_out_dim, head_num, mode="video"
    ):
        super().__init__()
        self.clip_model, self.processor = load(clip_name, download_root="weights/forensics_adapter")
        self.adapter = Adapter(
            vit_name=adapter_vit_name,
            num_quires=num_quires,
            fusion_map=fusion_map,
            mlp_dim=mlp_dim,
            mlp_out_dim=mlp_out_dim,
            head_num=head_num,
        )
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, num_quires)  # 全部参数被冻结
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=num_quires)
        self.clip_post_process = PostClipProcess(num_quires=num_quires, embed_dim=768)

        self.mode = mode
        self._freeze()

    def _freeze(self):
        for name, param in self.named_parameters():
            if "clip_model" in name:
                param.requires_grad = False

    def get_losses(self, data_dict, pred_dict):
        label = data_dict["label"]  # N
        xray = data_dict["xray"]
        pred = pred_dict["cls"]  # N2
        xray_pred = pred_dict["xray_pred"]
        loss_intra = pred_dict["loss_intra"]
        loss_clip = pred_dict["loss_clip"]
        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(pred.float(), label)
        if xray is not None:
            loss_mse = F.mse_loss(xray_pred.squeeze().float(), xray.squeeze().float())  # (N 1 224 224)->(N 224 224)

            loss = 10 * loss1 + 200 * loss_mse + 20 * loss_intra + 10 * loss_clip

            loss_dict = {"cls": loss1, "xray": loss_mse, "intra": loss_intra, "loss_clip": loss_clip, "overall": loss}
            return loss_dict
        else:
            loss_dict = {"cls": loss1, "overall": loss1}
            return loss_dict

    def forward(self, data_dict, inference=False):
        images = data_dict["image"]
        clip_images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        clip_features = self.clip_model.extract_features(clip_images, self.adapter.fusion_map.values())

        attn_biases, xray_preds, loss_adapter_intra = self.adapter(data_dict, clip_features, inference)
        clip_output, loss_clip = self.rec_attn_clip(
            data_dict, clip_features, attn_biases[-1], inference, normalize=True
        )

        # data_dict["if_boundary"] = data_dict["if_boundary"].to(self.device)
        # xray_preds = [self.masked_xray_post_process(xray_pred, data_dict["if_boundary"]) for xray_pred in xray_preds]

        clip_cls_output = self.clip_post_process(clip_output.float())  # N2

        # prob = torch.softmax(outputs["clip_cls_output"], dim=1)[:, 1]
        pred_dict = {
            "logits": clip_cls_output,
            # "cls": outputs["clip_cls_output"],
            # "prob": prob,
            # "xray_pred": xray_preds[-1], # N 1 224 224
            # "loss_intra": loss_adapter_intra,
            # "loss_clip": loss_clip,
        }

        return pred_dict
