"""
================================================================================
v35.0 模型定义模块
================================================================================
职责: BiLSTM+Attention模型定义（训练和预测共享）
确保: train_v35.py和predict_v35.py使用完全相同的模型架构
================================================================================
"""

import torch
import torch.nn as nn


class BiLSTMAttentionV2(nn.Module):
    """
    v34增强正则化版本 BiLSTM + 多头注意力

    架构:
      1. 输入投影: Linear(input_size, hidden_size) -> LayerNorm -> ReLU -> Dropout
      2. BiLSTM: num_layers层, hidden_size, bidirectional
      3. LSTM层归一化: LayerNorm(hidden_size * 2)
      4. 多头自注意力: attention_heads heads, embed_dim=hidden_size*2
      5. 注意力层归一化 + 残差连接
      6. 前馈网络: Linear(h*2, h*2) -> LayerNorm -> ReLU -> Dropout -> Linear(h*2, h*2) -> Dropout
      7. 前馈层归一化 + 残差连接
      8. 输出层: Dropout -> Linear(h*2, h) -> LayerNorm -> ReLU -> Dropout -> Linear(h, 1)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.6, attention_heads=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # BiLSTM编码器
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # LSTM层归一化
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)

        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # 注意力层归一化
        self.attn_norm = nn.LayerNorm(hidden_size * 2)

        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Dropout(dropout)
        )

        # 前馈层归一化
        self.ffn_norm = nn.LayerNorm(hidden_size * 2)

        # 输出层
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Xavier初始化
        self._init_weights()

    def _init_weights(self):
        """LSTM权重Xavier初始化"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        前向传播
        x: (batch, seq_len, input_size)
        return: (batch,) 预测分数
        """
        # 输入投影
        x_proj = self.input_projection(x)

        # BiLSTM编码
        lstm_out, _ = self.lstm(x_proj)
        lstm_out = self.lstm_norm(lstm_out)

        # 多头自注意力 + 残差
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)

        # 前馈网络 + 残差
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(attn_out + ffn_out)

        # 取最后一个时间步，输出预测
        last_hidden = ffn_out[:, -1, :]
        return self.output(last_hidden).squeeze(-1)
