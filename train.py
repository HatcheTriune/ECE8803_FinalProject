def train_one_fold(model, train_loader, val_loader, fold=0, device="cuda", num_epochs=10, lr=1e-4, save_dir="checkpoints"):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_fold_{fold+1}.pt")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, extra_features, labels in train_loader:
            images, extra_features, labels = images.to(device), extra_features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images, extra_features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, extra_features, labels in val_loader:
                images, extra_features, labels = images.to(device), extra_features.to(device), labels.to(device)
                val_loss += criterion(model(images, extra_features), labels).item() * images.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader.dataset):.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), save_path)
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
    return best_val_loss