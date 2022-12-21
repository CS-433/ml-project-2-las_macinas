"""
Functions to train our model
"""
import torch
import torch.nn as nn
import numpy as np

from compute_metrics import compute_metrics

def run_training(model, num_epochs, train_loader, val_loader, device="cuda"):
    # ===== Model, Optimizer and Criterion =====
    optimizer_kwargs = dict(lr=1e-3, weight_decay=1e-3,)
    model = model.to(device=device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    # ===== Train Model =====
    model.train()
    loss_train_history = []
    f1_train_history = []
    best_f1 = 0

    for epoch in range(1, num_epochs + 1):
        print('-------------> epoch: {}'.format(epoch))
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            pred = torch.round(output)

            # Compute confusion vector between 2 tensors
            confusion_vector = pred / target
            # Compute validation f1 score
            f1_train = compute_metrics(confusion_vector, print_values=False)
            f1_train_history.append(f1_train)
            
            loss.backward()
            optimizer.step()         
            loss_float = loss.item()
            loss_train_history.append(loss_float)

        train_losses = np.mean(loss_train_history)
        epoch_f1 = np.mean(f1_train_history)
        print('Loss score for training:', train_losses)
        print('F1 score for training:', epoch_f1)

        # ===== Validate Model =====
        model.eval()
        loss_val_history = []
        f1_val_history = []
        for data_val, target_val in val_loader:
            data_val, target_val = data_val.to(device), target_val.to(device)
            data_val = data_val.permute(0, 3, 1, 2)

            output_val = model(data_val)

            loss_val = criterion(output_val, target_val).item()
            loss_val_history.append(loss_val)

            pred_val = torch.round(output_val)

            # Compute confusion vector between 2 tensors
            confusion_vector_val = pred_val / target_val
            # Compute validation f1 score
            f1_val = compute_metrics(confusion_vector_val, print_values=False)
            f1_val_history.append(f1_val)
        
        val_loss = np.mean(loss_val_history)
        epoch_f1_val = np.mean(f1_val_history)
        print('\n Loss score for validation:', val_loss)
        print('F1 score for validation:', epoch_f1_val)

        # Save model if best
        if epoch_f1_val > best_f1:
            best_f1 = epoch_f1_val
            best_model = model
            torch.save(model.state_dict(), 'models/best_model_f1.pth')
            print('-------------> NEW BEST MODEL SAVED')

    return best_model

