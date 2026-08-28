import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_lora_image_encoder import LoRA_Sam
from model.modules.decoder_sam import SegFormerHead, SegFormerHead_adaption
from model import MemoryBank
from model.text_align import (
    ImageToTextCrossAttentionFusion,
    TextAlignAux,
    TextToImageGroundingFusion,
    TextPrototypeLogitHead,
    VisualTextLogitFusion,
)
from model.text_semantic import LightweightLanguageAdapter
from model.depth_adapter import LightDepthAdapter

class OmniSAM(nn.Module):
    def __init__(self, sam_model, num_classes, num_maskmem):
        super().__init__()
        self.sam_model = sam_model
        self.encoder = LoRA_Sam(sam_model, 4)
        self.decoder = SegFormerHead(num_classes=num_classes)
        # self.decoder = DMLPHead(num_classes=num_classes)
        self.num_maskmem = num_maskmem
        self.memory_bank = MemoryBank(sam_model, num_maskmem=num_maskmem)

    def forward(self, frame_idx, x, mem_type="source", historical_pseudo_labels=None):
        _, _, height, width = x.shape
        backbone_out = self.encoder(x)
        current_vision_feats = backbone_out['vision_features']
        current_vision_pos_embeds = backbone_out['vision_pos_enc'][-1]
        x = backbone_out['backbone_fpn']
        x[-1] = self.memory_bank(
                    frame_idx,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    num_obj_ptr_tokens=0,
                    mem_type=mem_type,
                    historical_pseudo_labels=historical_pseudo_labels,
                )
        x = self.decoder(x)
        pred = F.interpolate(x, size=(height,width), mode='bilinear', align_corners=False)
        # Encode new memory
        if self.num_maskmem > 0:
            output = torch.argmax(pred, 1)
            mask_for_mem = output.unsqueeze(1)
            with torch.no_grad():
                probs = torch.softmax(pred, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            self.memory_bank._encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type, uncertainty_map=entropy)

        return pred

class OmniSAM_adapt(nn.Module):
    def __init__(
        self,
        sam_model,
        num_classes: int = 13,
        num_maskmem: int = 9,
        text_prototypes: torch.Tensor | None = None,
        d_txt: int | None = None,
        lambda_txt: float = 0.0,
        txt_tau: float = 0.07,
        use_language_adapter: bool = False,
        adapter_num_heads: int = 4,
        adapter_scale: float = 0.1,
        use_lite_decoder: bool = False,
        decoder_channel_ratio: float = 1.0,
        decoder_conv_mode: str = "standard",
        decoder_upsample_mode: str = "bilinear",
        text_adapter_hidden_dim: int = 512,
        text_adapter_dropout: float = 0.0,
        text_adapter_normalize_output: bool = True,
        text_adapter_zero_init_last: bool = False,
        text_adapter_projector_type: str = "mlp",
        text_adapter_freeze: bool = False,
        text_logit_normalize_visual: bool = True,
        text_logit_normalize_text: bool = True,
        text_logit_learnable_scale: bool = True,
        text_logit_init_scale: float = 10.0,
        text_fusion_alpha_init: float = 0.0,
        text_fusion_alpha_learnable: bool = True,
        text_fusion_alpha_max: float | None = None,
        text_fusion_enabled: bool = False,
        text_fusion_type: str = "logit_add",
        text_cross_num_heads: int = 4,
        text_cross_dropout: float = 0.0,
        text_cross_beta_init: float = 0.05,
        text_cross_beta_learnable: bool = True,
        text_cross_beta_max: float | None = None,
        text_cross_use_ffn: bool = True,
        text_cross_ffn_ratio: float = 4.0,
        text_cross_norm_visual: bool = True,
        text_cross_norm_text: bool = True,
        text_cross_return_attn: bool = False,
        text_cross_attn_temperature: float = 1.0,
        text_ground_num_heads: int = 4,
        text_ground_num_layers: int = 2,
        text_ground_dropout: float = 0.0,
        text_ground_gamma_init: float = 0.05,
        text_ground_gamma_learnable: bool = True,
        text_ground_logit_scale_init: float = 10.0,
        text_ground_logit_scale_learnable: bool = True,
        text_ground_use_query_self_attn: bool = True,
        text_ground_use_ffn: bool = True,
        text_ground_ffn_ratio: float = 4.0,
        text_ground_num_queries_per_class: int = 1,
        text_ground_use_tie_breaker: bool = False,
        text_ground_query_delta_init: float = 0.0,
        text_ground_use_learnable_query_delta: bool = False,
        text_ground_kv_stride: int = 1,
        text_ground_return_attn: bool = True,
        text_ground_attn_temperature: float = 1.0,
        depth_adapter_cfg: dict | None = None,
    ):
        super().__init__()
        self.sam_model = sam_model
        self.encoder = LoRA_Sam(sam_model, 4)
        ratio = float(decoder_channel_ratio) if bool(use_lite_decoder) else 1.0
        conv_mode = str(decoder_conv_mode) if bool(use_lite_decoder) else "standard"
        upsample_mode = str(decoder_upsample_mode) if bool(use_lite_decoder) else "bilinear"
        self.decoder = SegFormerHead_adaption(
            num_classes=num_classes,
            channel_ratio=ratio,
            conv_mode=conv_mode,
            upsample_mode=upsample_mode,
        )
        # self.decoder = DMLPHead_adaption(num_classes=num_classes)
        self.num_maskmem = num_maskmem
        self.memory_bank = MemoryBank(sam_model, num_maskmem=num_maskmem)
        self.lambda_txt = float(lambda_txt)
        self.txt_tau = float(txt_tau)
        self.use_language_adapter = bool(use_language_adapter)
        self.depth_adapter: LightDepthAdapter | None = None

        self.txt_head: TextPrototypeLogitHead | None = None
        self.lang_adapter: LightweightLanguageAdapter | None = None
        self.text_fusion: VisualTextLogitFusion | None = None
        self.text_cross_fusion: ImageToTextCrossAttentionFusion | None = None
        self.text_grounding_fusion: TextToImageGroundingFusion | None = None
        self.text_fusion_enabled = bool(text_fusion_enabled)
        self.text_fusion_type = str(text_fusion_type or "logit_add").lower()
        d_decoder = int(getattr(self.decoder, "out_channels", 256))
        dcfg = depth_adapter_cfg if isinstance(depth_adapter_cfg, dict) else {}
        if bool(dcfg.get("enable", False)):
            self.depth_adapter = LightDepthAdapter(
                out_channels=d_decoder,
                mid_channels=int(dcfg.get("mid_channels", 32)),
                gate_init=float(dcfg.get("gate_init", 0.0)),
                norm=str(dcfg.get("norm", "gn")),
            )
        if text_prototypes is not None:
            if d_txt is None:
                d_txt = int(text_prototypes.shape[-1])
            self.txt_head = TextPrototypeLogitHead(
                d_model=d_decoder,
                d_text=int(d_txt),
                num_classes=num_classes,
                adapter_hidden_dim=int(text_adapter_hidden_dim),
                adapter_dropout=float(text_adapter_dropout),
                adapter_normalize_output=bool(text_adapter_normalize_output),
                adapter_zero_init_last=bool(text_adapter_zero_init_last),
                adapter_projector_type=str(text_adapter_projector_type),
                normalize_visual=bool(text_logit_normalize_visual),
                normalize_text=bool(text_logit_normalize_text),
                learnable_scale=bool(text_logit_learnable_scale),
                init_scale=float(text_logit_init_scale),
            )
            self.txt_head.set_text_prototypes(text_prototypes)
            self.text_fusion = VisualTextLogitFusion(
                alpha_init=float(text_fusion_alpha_init),
                alpha_learnable=bool(text_fusion_alpha_learnable),
                alpha_max=text_fusion_alpha_max,
            )
            self.text_cross_fusion = ImageToTextCrossAttentionFusion(
                dim=d_decoder,
                num_heads=int(text_cross_num_heads),
                dropout=float(text_cross_dropout),
                beta_init=float(text_cross_beta_init),
                beta_learnable=bool(text_cross_beta_learnable),
                beta_max=text_cross_beta_max,
                use_ffn=bool(text_cross_use_ffn),
                ffn_ratio=float(text_cross_ffn_ratio),
                norm_visual=bool(text_cross_norm_visual),
                norm_text=bool(text_cross_norm_text),
                return_attn=bool(text_cross_return_attn),
                attn_temperature=float(text_cross_attn_temperature),
            )
            self.text_grounding_fusion = TextToImageGroundingFusion(
                dim=d_decoder,
                num_classes=int(num_classes),
                num_heads=int(text_ground_num_heads),
                num_layers=int(text_ground_num_layers),
                dropout=float(text_ground_dropout),
                gamma_init=float(text_ground_gamma_init),
                gamma_learnable=bool(text_ground_gamma_learnable),
                use_query_self_attn=bool(text_ground_use_query_self_attn),
                use_ffn=bool(text_ground_use_ffn),
                ffn_ratio=float(text_ground_ffn_ratio),
                logit_scale_init=float(text_ground_logit_scale_init),
                logit_scale_learnable=bool(text_ground_logit_scale_learnable),
                num_queries_per_class=int(text_ground_num_queries_per_class),
                use_tie_breaker=bool(text_ground_use_tie_breaker),
                query_delta_init=float(text_ground_query_delta_init),
                use_learnable_query_delta=bool(text_ground_use_learnable_query_delta),
                kv_stride=int(text_ground_kv_stride),
                return_attn=bool(text_ground_return_attn),
                attn_temperature=float(text_ground_attn_temperature),
            )
            if bool(text_adapter_freeze):
                for p in self.txt_head.projector.parameters():
                    p.requires_grad = False
            if self.use_language_adapter:
                self.lang_adapter = LightweightLanguageAdapter(
                    d_in=d_decoder,
                    d_txt=int(d_txt),
                    num_heads=int(adapter_num_heads),
                    adapter_scale=float(adapter_scale),
                )

    def forward(self, frame_idx, x, mem_type="source", return_aux: bool = False, depth=None, historical_pseudo_labels=None):
        _, _, height, width = x.shape
        backbone_out = self.encoder(x)
        current_vision_feats = backbone_out['vision_features']
        current_vision_pos_embeds = backbone_out['vision_pos_enc'][-1]
        feat = backbone_out['backbone_fpn']
        feat[-1] = self.memory_bank(
                    frame_idx,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    mem_type=mem_type,
                    num_obj_ptr_tokens=0,
                    historical_pseudo_labels=historical_pseudo_labels,
                )
        x, feat_de = self.decoder(feat)
        feat_de_visual = feat_de
        if self.depth_adapter is not None:
            feat_de = self.depth_adapter(feat_de, depth=depth)
            feat_de_visual = feat_de
            x = self.decoder.linear_pred(self.decoder.dropout(feat_de))
        l_vis = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)

        if self.lang_adapter is not None and self.txt_head is not None:
            text_proto = self.txt_head.text_prototypes  # (K, D_txt)
            feat_de, _ = self.lang_adapter(feat_de, text_proto)

        l_txt = None
        fusion_alpha = 0.0
        text_scale = 0.0
        cross_aux = {}
        ground_aux = {}
        l_ground = None
        projected_text_proto = None
        need_txt_aux = bool(return_aux)
        if self.text_fusion_enabled:
            fusion_type = self.text_fusion_type
        elif self.txt_head is not None and self.lambda_txt != 0.0:
            fusion_type = "logit_add"
        else:
            fusion_type = "none"
        if self.txt_head is not None and fusion_type in {"image_to_text_cross_attn", "hybrid"}:
            projected_text_proto = self.txt_head.projector(self.txt_head.text_prototypes)
            feat_de_fused, cross_aux = self.text_cross_fusion(feat_de, projected_text_proto)
            feat_de = feat_de_fused
            x = self.decoder.linear_pred(self.decoder.dropout(feat_de))
        if self.txt_head is not None and fusion_type == "text_to_image_grounding":
            projected_text_proto = self.txt_head.projector(self.txt_head.text_prototypes)
            l_ground_low, ground_aux = self.text_grounding_fusion(feat_de_visual, projected_text_proto, visual_logits=x)
            l_ground = F.interpolate(l_ground_low, size=(height, width), mode='bilinear', align_corners=False)
            pred = l_vis + self.text_grounding_fusion.gamma.to(dtype=l_vis.dtype) * l_ground
        if self.txt_head is not None and (self.lambda_txt != 0.0 or need_txt_aux):
            # feat_de is the fused decoder feature map (B,256,H',W')
            txt_feat = feat_de if fusion_type in {"image_to_text_cross_attn", "hybrid"} else feat_de_visual
            l_txt_low = self.txt_head(txt_feat)
            l_txt = F.interpolate(l_txt_low, size=(height, width), mode='bilinear', align_corners=False)
            if fusion_type == "text_to_image_grounding":
                text_scale = float(self.txt_head.text_scale.detach().item())
            elif fusion_type == "image_to_text_cross_attn":
                pred = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
            elif fusion_type == "hybrid" and self.lambda_txt != 0.0:
                l_vis_fused = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
                if self.text_fusion is not None:
                    pred = self.text_fusion(l_vis_fused, self.lambda_txt * l_txt)
                    fusion_alpha = float(self.text_fusion.current_alpha().detach().item())
                else:
                    pred = l_vis_fused + (self.lambda_txt * l_txt)
            elif self.lambda_txt != 0.0:
                if self.text_fusion is not None and fusion_type != "none":
                    pred = self.text_fusion(l_vis, self.lambda_txt * l_txt)
                    fusion_alpha = float(self.text_fusion.current_alpha().detach().item())
                else:
                    pred = l_vis + (self.lambda_txt * l_txt)
            else:
                pred = l_vis
            text_scale = float(self.txt_head.text_scale.detach().item())
        else:
            pred = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False) if fusion_type in {"image_to_text_cross_attn", "hybrid"} else l_vis
        # Encode new memory
        if self.num_maskmem > 0:
            output = torch.argmax(pred, 1)
            mask_for_mem = output.unsqueeze(1)
            # NEW: Compute uncertainty (entropy) for uncertainty-guided token selection
            with torch.no_grad():
                probs = torch.softmax(pred, dim=1)  # [B, K, H, W]
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B, H, W]
            self.memory_bank._encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type, uncertainty_map=entropy)

        if return_aux and l_txt is not None:
            return pred, feat, feat_de, TextAlignAux(
                l_vis=l_vis,
                l_txt=l_txt,
                l_ground=l_ground,
                fusion_alpha=fusion_alpha,
                text_scale=text_scale,
                cross_beta=float(cross_aux.get("beta", 0.0)),
                cross_attn_mean=float(cross_aux.get("attn_mean", 0.0)),
                cross_attn_std=float(cross_aux.get("attn_std", 0.0)),
                cross_attn_entropy=float(cross_aux.get("attn_entropy", 0.0)),
                cross_attn_max_mean=float(cross_aux.get("text_attn_max", 0.0)),
                feature_delta_mean=float(cross_aux.get("feature_delta_mean", 0.0)),
                feature_delta_norm=float(cross_aux.get("feature_delta_norm", 0.0)),
                feature_delta_ratio=float(cross_aux.get("feature_delta_ratio", 0.0)),
                fused_feature_mean=float(cross_aux.get("fused_feature_mean", 0.0)),
                fused_feature_std=float(cross_aux.get("fused_feature_std", 0.0)),
                visual_feature_mean=float(cross_aux.get("visual_feature_mean", 0.0)),
                visual_feature_std=float(cross_aux.get("visual_feature_std", 0.0)),
                cross_attn_aux=dict(cross_aux),
                grounding_gamma=float(ground_aux.get("gamma", 0.0)),
                grounding_logit_scale=float(ground_aux.get("logit_scale", 0.0)),
                grounding_attn_mean=float(ground_aux.get("attn_mean", 0.0)),
                grounding_attn_std=float(ground_aux.get("attn_std", 0.0)),
                grounding_attn_entropy=float(ground_aux.get("attn_entropy", 0.0)),
                grounding_attn_max_mean=float(ground_aux.get("attn_max_mean", 0.0)),
                grounding_logits_mean=float(ground_aux.get("grounding_logits_mean", 0.0)),
                grounding_logits_std=float(ground_aux.get("grounding_logits_std", 0.0)),
                grounding_feature_delta_mean=float(ground_aux.get("query_delta_mean", 0.0)),
                grounding_feature_delta_norm=float(ground_aux.get("query_delta_norm", 0.0)),
                grounding_feature_delta_ratio=float(ground_aux.get("query_delta_ratio", 0.0)),
                grounding_topk_area_ratio=float(ground_aux.get("topk_area_ratio", 0.0)),
                grounding_aux=dict(ground_aux),
            )
        return pred, feat, feat_de