import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.sct_rdt import Full_sCTRDT_Model
from src.data_engine.dataset import AstroDataset
from src.utils.focal_loss import FocalLoss
from src.utils.metrics import calculate_metrics

from tqdm import tqdm

def main():
    # FAILSAFE: wrap config load so a missing/malformed YAML produces a clear error
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError("config.yaml is empty or invalid YAML.")
    except FileNotFoundError:
        print("[train.py] ERROR: 'config.yaml' not found. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    except (yaml.YAMLError, ValueError) as e:
        print(f"[train.py] ERROR parsing config.yaml: {e}", file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")

    # NOTE: val_path is intentionally set to train_path in config.yaml because PLAsTiCC
    # ships a single training_set.csv — split programmatically here or use a separate CSV.
    full_dataset = AstroDataset(config['data']['train_path'], config['data']['max_seq_len'], occlusion_level="0%")
    
    if config['data']['train_path'] == config['data']['val_path']:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        print(f"Split dataset: {train_size} training samples, {val_size} validation samples.")
    else:
        train_dataset = full_dataset
        val_dataset   = AstroDataset(config['data']['val_path'], config['data']['max_seq_len'], occlusion_level="0%")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    model = Full_sCTRDT_Model(config['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 1. CLASS IMBALANCE: Calculate dynamic alpha weights
    class_weights = full_dataset.get_class_weights(config['model']['num_classes']).to(device)
    criterion = FocalLoss(gamma=config['training']['gamma_focal'], alpha=class_weights)
    
    # 2. LR PLATEAU: Reduce learning rate when validation F1 stagnates
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_f1 = 0.0

    for epoch in range(config['training']['epochs']):
        # TRAIN
        model.train()
        total_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()
            f, p, t, e, mask, labels = [b.to(device) for b in batch]

            # mask needs to be [B, 1, 1, S]
            mask = mask.unsqueeze(1).unsqueeze(1)

            logits = model(f, p, t, e, mask)
            loss = criterion(logits, labels)
            loss.backward()

            # FAILSAFE: gradient clipping prevents NaN/inf explosions from the
            # additive attention kernels (LTDK + PEG) during early training.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{total_loss / (train_pbar.n + 1):.4f}"})

        # VALIDATE
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
            for batch in val_pbar:
                f, p, t, e, mask, labels = [b.to(device) for b in batch]
                mask = mask.unsqueeze(1).unsqueeze(1)

                logits = model(f, p, t, e, mask)
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc, val_f1 = calculate_metrics(all_preds, all_labels)
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        # Step the scheduler based on Validation F1
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  -> Current LR: {current_lr:.6f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print("  -> Saved new best model")

    print("Training Complete.")

if __name__ == "__main__":
    main()

