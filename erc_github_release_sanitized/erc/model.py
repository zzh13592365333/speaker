import torch
import torch.nn as nn

from .utils import DEFAULT_HIDDEN_DIM as HIDDEN_DIM


class TextFeatureEncoder(nn.Module):
    def __init__(self, input_dim=1024, output_dim=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class AudioFeatureEncoder(nn.Module):
    def __init__(self, input_dim=1024, output_dim=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x):
        return self.transformer(self.proj(x))


class VideoFeatureEncoder(nn.Module):
    def __init__(self, input_dim=4096, output_dim=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim * 4), nn.GELU(), nn.Linear(output_dim * 4, output_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim * 4), nn.GELU(), nn.Linear(output_dim * 4, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, mask=None):
        del mask
        x = self.proj(x)
        x = x + self.ffn1(x)
        x = x + self.ffn2(x)
        return self.norm(x)


class ContextAwareEncoder(nn.Module):
    def __init__(self, feature_dim=HIDDEN_DIM):
        super().__init__()
        self.attn_fusion = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=4, dim_feedforward=feature_dim * 2, batch_first=True, dropout=0.2
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, curr, prev_seq, next_seq, prev_mask, next_mask, prev_spk, next_spk):
        del prev_spk, next_spk
        bsz = curr.shape[0]
        win = prev_seq.shape[1]
        device = curr.device
        seq_all = torch.cat([prev_seq, curr.unsqueeze(1), next_seq], dim=1)
        curr_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
        key_mask = torch.cat([(prev_mask == 0).bool(), curr_mask, (next_mask == 0).bool()], dim=1)
        out_seq = self.attn_fusion(seq_all, src_key_padding_mask=key_mask)
        return self.norm(out_seq[:, win, :])


class CrossModalInteraction(nn.Module):
    """Video-as-query interaction for retrieving audio/text affective references."""

    def __init__(self, hidden_dim=HIDDEN_DIM, use_delta_diff=True):
        super().__init__()
        self.use_delta_diff = use_delta_diff
        self.v_a_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.v_t_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.delta_fuser = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, v_seq, a_seq=None, t_seq=None, enable_audio_video=True, enable_text_video=True):
        v_vec = v_seq.squeeze(1)
        v_query = v_vec.unsqueeze(1)
        v_guided = self.norm_v(v_vec)
        outputs = {"v_guided": v_guided}
        delta_parts = []

        if a_seq is not None:
            if enable_audio_video:
                a_ctx, _ = self.v_a_attn(query=v_query, key=a_seq, value=a_seq)
                a_base = a_seq.mean(dim=1, keepdim=True)
                a_guided = self.norm_a(a_ctx + a_base).squeeze(1)
                delta_parts.append(a_guided - v_guided if self.use_delta_diff else a_guided)
            else:
                a_guided = a_seq.mean(dim=1)
            outputs["a_guided"] = a_guided

        if t_seq is not None:
            if enable_text_video:
                t_ctx, _ = self.v_t_attn(query=v_query, key=t_seq, value=t_seq)
                t_base = t_seq.mean(dim=1, keepdim=True)
                t_guided = self.norm_t(t_ctx + t_base).squeeze(1)
                delta_parts.append(t_guided - v_guided if self.use_delta_diff else t_guided)
            else:
                t_guided = t_seq.mean(dim=1)
            outputs["t_guided"] = t_guided

        if len(delta_parts) == 0:
            delta = torch.zeros_like(v_guided)
        elif len(delta_parts) == 1:
            delta = delta_parts[0]
        else:
            delta = self.delta_fuser(torch.cat(delta_parts, dim=-1))
        outputs["delta"] = delta
        return outputs


class ConfidenceGate(nn.Module):
    def forward(self, v_logits, delta):
        v_conf = torch.max(torch.softmax(v_logits, dim=-1), dim=-1).values
        gate = (1.0 - v_conf).unsqueeze(-1)
        return gate * delta, gate


class MultimodalERCModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.modes = set(args.modalities.split(","))
        self.disable_context = getattr(args, "disable_context", False)
        self.disable_crossmodal = getattr(args, "disable_crossmodal", False)
        self.disable_audio_video_interaction = getattr(args, "disable_audio_video_interaction", False)
        self.disable_text_video_interaction = getattr(args, "disable_text_video_interaction", False)
        self.use_conf_gate = getattr(args, "use_conf_gate", True)

        if "text" in self.modes:
            self.text_encoder = TextFeatureEncoder(args.text_dim)
            self.ctx_fusion_text = ContextAwareEncoder()
            self.proj_t = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 128))
        if "audio" in self.modes:
            self.audio_encoder = AudioFeatureEncoder(args.audio_dim)
            self.ctx_fusion_audio = ContextAwareEncoder()
            self.proj_a = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 128))
        if "video" in self.modes:
            self.video_encoder = VideoFeatureEncoder(args.video_dim)
            self.ctx_fusion_video = ContextAwareEncoder()
            self.proj_v = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 128))
            self.video_aux_head = nn.Linear(HIDDEN_DIM, args.num_classes)
        else:
            self.video_aux_head = None

        if "video" in self.modes and len(self.modes) > 1:
            self.interaction = CrossModalInteraction(use_delta_diff=not getattr(args, "disable_delta_diff", False))
        else:
            self.interaction = None
        self.conf_gate = ConfidenceGate() if ("video" in self.modes and len(self.modes) > 1 and self.use_conf_gate) else None

        self.fusion_dim = HIDDEN_DIM * len(self.modes)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(512, args.num_classes)
        )
        self.proj_fused = nn.Sequential(nn.Linear(self.fusion_dim, self.fusion_dim), nn.ReLU(), nn.Linear(self.fusion_dim, 128))

    def _encode_text_ctx(self, x):
        bsz, win, dim = x.shape
        enc = self.text_encoder(x.reshape(-1, dim).unsqueeze(1)).squeeze(1)
        return enc.reshape(bsz, win, -1)

    def _text_branch(self, batch, cl_feats):
        t_seq = self.text_encoder(batch["text"])
        t_curr = t_seq.squeeze(1)
        if self.disable_context:
            t_vec = t_curr
        else:
            t_vec = self.ctx_fusion_text(t_curr, self._encode_text_ctx(batch["prev_text"]), self._encode_text_ctx(batch["next_text"]), batch["prev_mask"], batch["next_mask"], batch["prev_spk"], batch["next_spk"])
        if not self.args.disable_cl:
            cl_feats["t"] = self.proj_t(t_vec)
        return t_vec.unsqueeze(1), t_vec

    def _audio_branch(self, batch, cl_feats):
        a_seq = self.audio_encoder(batch["audio"])
        a_curr = a_seq.mean(dim=1)
        if self.disable_context:
            a_vec = a_curr
        else:
            a_vec = self.ctx_fusion_audio(a_curr, self.audio_encoder.proj(batch["prev_audio"]), self.audio_encoder.proj(batch["next_audio"]), batch["prev_mask"], batch["next_mask"], batch["prev_spk"], batch["next_spk"])
        a_seq_ctx = a_seq + (a_vec - a_curr).unsqueeze(1)
        if not self.args.disable_cl:
            cl_feats["a"] = self.proj_a(a_vec)
        return a_seq_ctx, a_vec

    def _video_branch(self, batch, cl_feats):
        v_seq = self.video_encoder(batch["video"], batch.get("video_mask"))
        v_curr = v_seq.squeeze(1)
        if self.disable_context:
            v_vec = v_curr
        else:
            v_vec = self.ctx_fusion_video(v_curr, self.video_encoder.proj(batch["prev_video"]), self.video_encoder.proj(batch["next_video"]), batch["prev_mask"], batch["next_mask"], batch["prev_spk"], batch["next_spk"])
        v_seq_ctx = v_seq + (v_vec - v_curr).unsqueeze(1)
        if not self.args.disable_cl:
            cl_feats["v"] = self.proj_v(v_vec)
        return v_seq_ctx, v_vec

    def forward(self, batch):
        cl_feats = {}
        t_seq_ctx = t_vec = a_seq_ctx = a_vec = v_seq_ctx = v_vec = None
        if "text" in self.modes:
            t_seq_ctx, t_vec = self._text_branch(batch, cl_feats)
        if "audio" in self.modes:
            a_seq_ctx, a_vec = self._audio_branch(batch, cl_feats)
        if "video" in self.modes:
            v_seq_ctx, v_vec = self._video_branch(batch, cl_feats)

        t_final, a_final, v_final = t_vec, a_vec, v_vec
        v_logits_aux = self.video_aux_head(v_vec) if self.video_aux_head is not None and v_vec is not None else None
        gate_value = None

        has_cross_source = (a_seq_ctx is not None and not self.disable_audio_video_interaction) or (t_seq_ctx is not None and not self.disable_text_video_interaction)
        if v_seq_ctx is not None and has_cross_source and not self.disable_crossmodal and self.interaction is not None:
            inter = self.interaction(
                v_seq=v_seq_ctx,
                a_seq=a_seq_ctx,
                t_seq=t_seq_ctx,
                enable_audio_video=a_seq_ctx is not None and not self.disable_audio_video_interaction,
                enable_text_video=t_seq_ctx is not None and not self.disable_text_video_interaction,
            )
            delta = inter["delta"]
            if self.conf_gate is not None and v_logits_aux is not None:
                gated_delta, gate_value = self.conf_gate(v_logits_aux, delta)
                v_final = inter["v_guided"] + gated_delta
            else:
                v_final = inter["v_guided"] + delta
            if "text" in self.modes:
                t_final = inter.get("t_guided", t_vec)
            if "audio" in self.modes:
                a_final = inter.get("a_guided", a_vec)

        active = [x for x in (t_final, a_final, v_final) if x is not None]
        combined = torch.cat(active, dim=-1)
        logits = self.classifier(combined)
        out = {"logits": logits, "features": combined}
        if v_logits_aux is not None:
            out["v_logits_aux"] = v_logits_aux
        if gate_value is not None:
            out["gate_value"] = gate_value
        if not self.args.disable_cl and cl_feats:
            cl_feats["fused"] = self.proj_fused(combined)
            out["cl_features"] = cl_feats
        return out


class UnimodalERCModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.modality = args.modality_type
        self.disable_context = getattr(args, "disable_context", False)
        if self.modality == "text":
            self.encoder = TextFeatureEncoder(args.text_dim)
        elif self.modality == "audio":
            self.encoder = AudioFeatureEncoder(args.audio_dim)
        elif self.modality == "video":
            self.encoder = VideoFeatureEncoder(args.video_dim)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
        self.context_encoder = ContextAwareEncoder()
        self.classifier = nn.Sequential(nn.Linear(HIDDEN_DIM, 256), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(256, args.num_classes))

    def _enc_text_ctx(self, x):
        bsz, win, dim = x.shape
        return self.encoder(x.reshape(-1, dim).unsqueeze(1)).squeeze(1).reshape(bsz, win, -1)

    def forward(self, batch):
        if self.modality == "text":
            feat = self.encoder(batch["text"])
            prev_feat, next_feat = self._enc_text_ctx(batch["prev_text"]), self._enc_text_ctx(batch["next_text"])
        elif self.modality == "audio":
            feat = self.encoder(batch["audio"])
            prev_feat, next_feat = self.encoder.proj(batch["prev_audio"]), self.encoder.proj(batch["next_audio"])
        else:
            feat = self.encoder(batch["video"], batch.get("video_mask"))
            prev_feat, next_feat = self.encoder.proj(batch["prev_video"]), self.encoder.proj(batch["next_video"])
        curr_vec = feat.mean(dim=1) if feat.dim() == 3 and feat.shape[1] > 1 else feat.squeeze(1)
        if self.disable_context:
            fused = curr_vec
        else:
            fused = self.context_encoder(curr_vec, prev_feat, next_feat, batch["prev_mask"], batch["next_mask"], batch["prev_spk"], batch["next_spk"])
        return {"logits": self.classifier(fused), "features": fused}
