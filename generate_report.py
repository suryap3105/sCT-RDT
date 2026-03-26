import sys
import yaml
import torch
import numpy as np
import torch.utils.data
from src.models.sct_rdt import Full_sCTRDT_Model
from src.data_engine.dataset import AstroDataset
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def main():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}...")

    # For evaluation, we want the validation set.
    full_dataset = AstroDataset(config['data']['train_path'], config['data']['max_seq_len'], occlusion_level="0%")
    if config['data']['train_path'] == config['data']['val_path']:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        val_dataset = AstroDataset(config['data']['val_path'], config['data']['max_seq_len'], occlusion_level="0%")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    model = Full_sCTRDT_Model(config['model']).to(device)
    
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True), strict=False)
        print("Loaded best_model.pth successfully.")
    except Exception as e:
        print(f"Could not load best_model.pth: {e}")
        return

    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            f, p, t, e, mask, labels = [b.to(device) for b in batch]
            mask = mask.unsqueeze(1).unsqueeze(1)
            logits = model(f, p, t, e, mask)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
