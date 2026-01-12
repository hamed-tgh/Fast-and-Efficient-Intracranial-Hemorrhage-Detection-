import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
from Kaggle_process_dataset import main
from loader_data import train_model
from torch.utils.tensorboard import SummaryWriter
import torchvision
from Visualize import visualize
from sklearn.metrics import roc_auc_score
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from plot import plot
from ploting_tensorboard import main


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_repvit_m1_1(num_classes=6, pretrained=False):
    model = timm.create_model( "repvit_m1_1", pretrained=pretrained, num_classes=num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, writer, epoch):
    model.train()
    running_loss = 0
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # --- writer: log batch loss ---
        global_step = epoch * len(loader) + step
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)

        pbar.set_postfix({"loss": loss.item()})

    # --- نمونه تصاویر برای TensorBoard ---
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("Sample_Images", img_grid, epoch)

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, writer, epoch):
    model.eval()
    running_loss = 0

    for images, labels in tqdm(loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    val_loss = running_loss / len(loader)

    # Log val loss
    writer.add_scalar("Loss/val_epoch", val_loss, epoch)

    return val_loss


def train_repvit(train_loader, val_loader, num_classes=6, epochs=50):

    # -------- tensorboard writer --------
    writer = SummaryWriter(log_dir="runs/RepViT_experiment")

    model = create_repvit_m1_1(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=7e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n========== EPOCH {epoch}/{epochs} ==========")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, writer, epoch)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)

        val_loss = validate(model, val_loader, criterion, writer, epoch)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "repvit_m1_1_rsna_best.pth")
            print("Model Saved!")
        
        torch.save(model.state_dict(), "repvit_m1_1_rsna_ep_" + str(epoch) + ".pth")

    writer.close()
    return model


@torch.no_grad()
def test_accuracy(model, loader, threshold=0.5):
    model.eval()
    
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Testing Accuracy"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)              # logits
        probs = torch.sigmoid(outputs)       # probabilities
        preds = (probs > threshold).float()  # binary predictions

        correct += (preds == labels).sum().item()
        total += labels.numel()

    accuracy = correct / total
    return accuracy


@torch.no_grad()
def evaluate_model(model, loader, threshold=0.5):
    model.eval()

    all_labels = []
    all_probs = []

    correct = 0
    total = 0
    correct_samples = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        # label-wise accuracy
        correct += (preds == labels).sum().item()
        total += labels.numel()

        # sample-wise accuracy
        matches = (preds == labels).all(dim=1)
        correct_samples += matches.sum().item()
        total_samples += labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = (y_prob >= threshold).astype(int)

    # ---- Precision / Recall / F1 ----
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_c, r_c, f1_c, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # ---- AUROC (macro) ----
    try:
        auroc = roc_auc_score(y_true, y_prob, average="macro")
    except:
        auroc = np.nan

    return {
        "label_acc": correct / total,
        "sample_acc": correct_samples / total_samples,

        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f1_micro),

        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),

        "precision_per_class": p_c.tolist(),
        "recall_per_class": r_c.tolist(),
        "f1_per_class": f1_c.tolist(),
        "support_per_class": sup.tolist(),

        "auroc": auroc
    }

def evaluate_all_checkpoints(model_fn, loader, weights_dir="."):
    results = []

    weight_files = sorted(glob.glob(f"{weights_dir}/repvit_m1_1_rsna_*.pth"))

    for w in weight_files:
        print(f"\nLoading {w}")
        model = model_fn(num_classes=6, pretrained=False)
        model.load_state_dict(torch.load(w, map_location=device))
        model.to(device)

        metrics = evaluate_model(model, loader)

        results.append({
            "checkpoint": w.split("/")[-1],
            **metrics
        })

    df = pd.DataFrame(results)
    return df




if __name__ == "__main__":
    Train = True
    Visual = False
    validate_whole_model = True
    validate_one_epoche = True

    writer = SummaryWriter(log_dir="runs/exp1")

    model, train_loader, val_loader = train_model(
        num_epochs=3,
        use_sample=False,
        sample_size=500,
        batch_size=64,
        lr=1e-3
    )
    
    if Visual: 
        visualize(train_loader)

    if Train :
        train_repvit(train_loader, val_loader, num_classes=6, epochs=20)

    if validate_whole_model:
        df = evaluate_all_checkpoints(
            create_repvit_m1_1,
            val_loader,
            weights_dir="."
        )

        # Sort AUROC
        df_sorted = df.sort_values("auroc", ascending=False)

        print(df_sorted)

        # Save CSV 
        df_sorted.to_csv("repvit_rsna_checkpoints_eval.csv", index=False)
        plot()
        main()





