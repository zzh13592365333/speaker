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
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
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
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=feature_dim * 2,
            batch_first=True,
            dropout=0.2,
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

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.v_a_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.v_t_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.delta_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, v_seq, a_seq, t_seq):
        v_vec = v_seq.squeeze(1)
        v_query = v_vec.unsqueeze(1)
        v_guided = self.norm_v(v_vec)

        a_ctx, _ = self.v_a_attn(query=v_query, key=a_seq, value=a_seq)
        a_base = a_seq.mean(dim=1, keepdim=True)
        a_guided = self.norm_a(a_ctx + a_base).squeeze(1)

        t_ctx, _ = self.v_t_attn(query=v_query, key=t_seq, value=t_seq)
        t_base = t_seq.mean(dim=1, keepdim=True)
        t_guided = self.norm_t(t_ctx + t_base).squeeze(1)

        delta = self.delta_fuser(torch.cat([t_guided - v_guided, a_guided - v_guided], dim=-1))
        return {
            "v_guided": v_guided,
            "a_guided": a_guided,
            "t_guided": t_guided,
            "delta": delta,
        }


class ConfidenceGate(nn.Module):
    def forward(self, v_logits, delta):
        v_conf = torch.max(torch.softmax(v_logits, dim=-1), dim=-1).values
        gate = (1.0 - v_conf).unsqueeze(-1)
        return gate * delta, gate


class MultimodalERCModel(nn.Module):
    """Full text-audio-video ERC model."""

    def __init__(self, args):
        super().__init__()
        self.text_encoder = TextFeatureEncoder(args.text_dim)
        self.audio_encoder = AudioFeatureEncoder(args.audio_dim)
        self.video_encoder = VideoFeatureEncoder(args.video_dim)

        self.ctx_fusion_text = ContextAwareEncoder()
        self.ctx_fusion_audio = ContextAwareEncoder()
        self.ctx_fusion_video = ContextAwareEncoder()

        self.video_aux_head = nn.Linear(HIDDEN_DIM, args.num_classes)
        self.interaction = CrossModalInteraction()
        self.conf_gate = ConfidenceGate()

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, args.num_classes),
        )

        self.proj_t = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 128))
        self.proj_a = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 128))
        self.proj_v = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 128))
        self.proj_fused = nn.Sequential(nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM * 3), nn.ReLU(), nn.Linear(HIDDEN_DIM * 3, 128))

    def _encode_text_ctx(self, x):
        bsz, win, dim = x.shape
        enc = self.text_encoder(x.reshape(-1, dim).unsqueeze(1)).squeeze(1)
        return enc.reshape(bsz, win, -1)

    def _text_branch(self, batch):
        t_seq = self.text_encoder(batch["text"])
        t_curr = t_seq.squeeze(1)
        t_vec = self.ctx_fusion_text(
            t_curr,
            self._encode_text_ctx(batch["prev_text"]),
            self._encode_text_ctx(batch["next_text"]),
            batch["prev_mask"],
            batch["next_mask"],
            batch["prev_spk"],
            batch["next_spk"],
        )
        return t_vec.unsqueeze(1), t_vec

    def _audio_branch(self, batch):
        a_seq = self.audio_encoder(batch["audio"])
        a_curr = a_seq.mean(dim=1)
        a_vec = self.ctx_fusion_audio(
            a_curr,
            self.audio_encoder.proj(batch["prev_audio"]),
            self.audio_encoder.proj(batch["next_audio"]),
            batch["prev_mask"],
            batch["next_mask"],
            batch["prev_spk"],
            batch["next_spk"],
        )
        a_seq_ctx = a_seq + (a_vec - a_curr).unsqueeze(1)
        return a_seq_ctx, a_vec

    def _video_branch(self, batch):
        v_seq = self.video_encoder(batch["video"], batch.get("video_mask"))
        v_curr = v_seq.squeeze(1)
        v_vec = self.ctx_fusion_video(
            v_curr,
            self.video_encoder.proj(batch["prev_video"]),
            self.video_encoder.proj(batch["next_video"]),
            batch["prev_mask"],
            batch["next_mask"],
            batch["prev_spk"],
            batch["next_spk"],
        )
        v_seq_ctx = v_seq + (v_vec - v_curr).unsqueeze(1)
        return v_seq_ctx, v_vec

    def forward(self, batch):
        t_seq_ctx, t_vec = self._text_branch(batch)
        a_seq_ctx, a_vec = self._audio_branch(batch)
        v_seq_ctx, v_vec = self._video_branch(batch)

        v_logits_aux = self.video_aux_head(v_vec)
        inter = self.interaction(v_seq=v_seq_ctx, a_seq=a_seq_ctx, t_seq=t_seq_ctx)
        gated_delta, gate_value = self.conf_gate(v_logits_aux, inter["delta"])

        v_final = inter["v_guided"] + gated_delta
        t_final = inter["t_guided"]
        a_final = inter["a_guided"]

        combined = torch.cat([v_final, t_final, a_final], dim=-1)
        logits = self.classifier(combined)

        cl_features = {
            "t": self.proj_t(t_vec),
            "a": self.proj_a(a_vec),
            "v": self.proj_v(v_vec),
            "fused": self.proj_fused(combined),
        }

        return {
            "logits": logits,
            "features": combined,
            "v_logits_aux": v_logits_aux,
            "gate_value": gate_value,
            "cl_features": cl_features,
        }
