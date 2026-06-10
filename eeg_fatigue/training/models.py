import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNetFatigue(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim, num_blocks, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.res_blocks(self.input_proj(x)))


class MLPFatigue(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_svm_model():
    return SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
               probability=True, random_state=42)


def build_rf_model():
    return RandomForestClassifier(n_estimators=200, max_depth=None,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
