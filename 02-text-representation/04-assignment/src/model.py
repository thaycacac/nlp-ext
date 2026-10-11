"""LSTM một chiều, một tầng - dùng lại NGUYÊN VẸN cho mọi điều kiện embedding."""
import numpy as np
import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """token ids -> embedding -> LSTM -> trạng thái ẩn cuối HỢP LỆ -> Linear -> 1 logit."""

    def __init__(self, embedding_matrix, hidden_dim=128, trainable=False):
        super().__init__()
        weights = torch.as_tensor(np.asarray(embedding_matrix), dtype=torch.float32)
        # from_pretrained(freeze=...) đặt requires_grad theo `trainable`.
        # padding_idx=0: hàng <pad> không nhận gradient (luôn là 0).
        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=not trainable, padding_idx=0)
        self.lstm = nn.LSTM(weights.shape[1], hidden_dim, num_layers=1,
                            batch_first=True)          # một chiều, một tầng
        self.fc = nn.Linear(hidden_dim, 1)             # một logit mỗi review

    def forward(self, token_ids, lengths):
        # Padding nằm ở cuối chuỗi và LSTM chạy theo chiều thuận, nên đầu ra tại
        # các bước hợp lệ KHÔNG bị ảnh hưởng bởi padding phía sau. Ta chỉ cần lấy
        # đầu ra ở vị trí lengths-1 (trạng thái ẩn cuối hợp lệ) là padding không
        # trở thành tín hiệu học. (Cách này tương đương pack_padded_sequence nhưng
        # chạy tốt trên cả CPU lẫn GPU Apple.)
        out, _ = self.lstm(self.embedding(token_ids))          # [B, T, H]
        last = out[torch.arange(out.size(0), device=out.device), lengths - 1]
        return self.fc(last).squeeze(-1)                       # [B]

    @torch.no_grad()
    def extend_embedding(self, extra_rows):
        """Thêm hàng embedding (dùng cho tra cứu fastText động ở thời điểm suy luận)."""
        extra = torch.as_tensor(np.asarray(extra_rows), dtype=torch.float32,
                                device=self.embedding.weight.device)
        merged = torch.cat([self.embedding.weight.data, extra], dim=0)
        self.embedding = nn.Embedding.from_pretrained(merged, freeze=True, padding_idx=0)


def count_parameters(model):
    """(số tham số huấn luyện được, tổng số tham số)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
