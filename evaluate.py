import sys
import yaml
import torch
from torch.utils.data import DataLoader
from src.models.sct_rdt import Full_sCTRDT_Model
from src.data_engine.dataset import AstroDataset
from src.utils.metrics import calculate_metrics
from tqdm import tqdm

def run_ablation_benchmark():
    # FAILSAFE: wrap config load so a missing/malformed YAML produces a clear error
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError("config.yaml is empty or invalid YAML.")
    except FileNotFoundError:
        print("[evaluate.py] ERROR: 'config.yaml' not found. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    except (yaml.YAMLError, ValueError) as e:
        print(f"[evaluate.py] ERROR parsing config.yaml: {e}", file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Full_sCTRDT_Model(config['model']).to(device)

    try:
        # FAILSAFE: weights_only=True avoids arbitrary code execution from untrusted
        # checkpoint files and silences the PyTorch >= 2.0 FutureWarning.
        model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
        print("Loaded 'best_model.pth' successfully.")
    except FileNotFoundError:
        print("[evaluate.py] Warning: 'best_model.pth' not found. Evaluating with untrained (random) weights.")
    except RuntimeError as e:
        print(f"[evaluate.py] Warning: Could not load 'best_model.pth' ({e}). Evaluating with untrained weights.")

    model.eval()

    levels = ["0%", "50%_random", "75%_block"]

    print("\n--- sCT-RDT OCCLUSION BENCHMARK ---")
    
    # Evaluate on the holdout validation split from the training dataset so we have real targets to score against.
    base_dataset = AstroDataset(config['data']['train_path'], config['data']['max_seq_len'], occlusion_level="0%")
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        base_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    for level in levels:
        # Dynamically change occlusion level on the base dataset object
        base_dataset.occlusion_level = level
        test_loader  = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Evaluating [{level}]")
            for batch in test_pbar:
                f, p, t, e, mask, labels = [b.to(device) for b in batch]
                mask = mask.unsqueeze(1).unsqueeze(1)

                logits = model(f, p, t, e, mask)
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc, f1_score = calculate_metrics(all_preds, all_labels)
        print(f"Masking Level: {level:<12} | Accuracy: {acc:.4f} | Macro F1-Score: {f1_score:.4f}")

if __name__ == "__main__":
    run_ablation_benchmark()

