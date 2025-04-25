import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

def evaluate_on_test(model, test_loader, device="cuda"):
    model = model.to(device).eval()
    criterion = nn.BCEWithLogitsLoss()
    all_preds, all_labels = [], []
    test_loss = 0.0
    with torch.no_grad():
        for images, extra_features, labels in test_loader:
            images, extra_features, labels = images.to(device), extra_features.to(device), labels.to(device)
            outputs = model(images, extra_features)
            test_loss += criterion(outputs, labels).item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    print(f"\n Test Loss: {test_loss / len(test_loader.dataset):.4f}")
    print(f" Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f" Test F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")