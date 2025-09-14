#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import random
import gc; gc.collect()

# 1. Set Python random seed
random.seed(1)
# 2. Set NumPy random seed
np.random.seed(1)
# 3. Set PyTorch seed
torch.manual_seed(1)

#%%

train = pd.read_csv("fmtraining_short.csv")
valid = pd.read_csv("fmvalid_short.csv")
test = pd.read_csv("fmtest_short.csv")

#%% hyperparameter

window_size=40 #num of previous event considered
epochs=10 #num of epochs for model training
num_workers = 0 #num of cpu
training_batch_size = 256
valid_batch_size = 1024
test_batch_size = 1024 

#%%

# List of all possible actions
actions = [
    'Pass', 'Carry', 'Miscontrol', 'Ball Recovery', 'Dribble', 
    'Shot', 'Clearance', 'Dispossessed', 'Interception', 'Shield', 'PC1', 'PC0'
]

# Create mapping dictionaries for actions
char2idx = {action: idx for idx, action in enumerate(actions)}
idx2char = {idx: action for action, idx in char2idx.items()}

num_actions = len(actions)

# Encode actions
train['act'] = train['act'].replace(char2idx).astype(np.int64)
valid['act'] = valid['act'].replace(char2idx).astype(np.int64)
test['act']  = test['act'].replace(char2idx).astype(np.int64)

# Encode Team ID (TID) if not numeric
# If TID is already numeric, skip this encoding
tid_values = sorted(train['TID'].unique())
tid2idx = {tid: idx for idx, tid in enumerate(tid_values)}
idx2tid = {idx: tid for tid, idx in tid2idx.items()}

train['TID'] = train['TID'].replace(tid2idx).astype(np.int64)
valid['TID'] = valid['TID'].replace(tid2idx).astype(np.int64)
test['TID']  = test['TID'].replace(tid2idx).astype(np.int64)

num_teams = len(tid_values)

#%% Zero-index zone labels (convert from 1-9 to 0-8)

train['zone'] -= 1
valid['zone'] -= 1
test['zone']  -= 1

num_zones = 9  # useful for defining embedding layer later

#%% Specify input and target variables

all_input_vars = ['deltaT', 'zone', 'act', 'TID']
all_target_vars = ['deltaT', 'zone', 'act', 'TID']

def get_input_vars(mode='train'):
    return all_input_vars

#%% Specify loss function weighting

def get_class_weights(train):
    # Zone (already 0–8 after preprocessing)
    num_zones = train['zone'].nunique()
    zone_classes = np.arange(num_zones)
    weight_zone = torch.tensor(compute_class_weight(
        class_weight="balanced",
        classes=zone_classes,
        y=train['zone'].values
    ), dtype=torch.float32)

    # Action (already integer IDs via char2idx)
    num_actions = train['act'].nunique()
    action_classes = np.arange(num_actions)
    weight_action = torch.tensor(compute_class_weight(
        class_weight="balanced",
        classes=action_classes,
        y=train['act'].values
    ), dtype=torch.float32)
    
    # TID (already integer IDs consistently across datasets)
    tid_classes = np.arange(num_teams)  # use global num_teams
    weight_TID_values = compute_class_weight(
        class_weight="balanced",
        classes=tid_classes,
        y=train['TID'].values
    )

    weight_TID_values = weight_TID_values * 4  

    weight_TID = torch.tensor(weight_TID_values, dtype=torch.float32)

    # DeltaT
    weight_deltaT = torch.tensor([1.], dtype=torch.float32)
    
    return weight_zone, weight_action, weight_TID, weight_deltaT

weight_zone_class, weight_action_class, weight_TID_class, weight_deltaT = get_class_weights(train)

#%% Flag valid slices for sequence modeling

def valid_slice_flag(df, window_size):
    df = df.copy()
    df = df.assign(valid_slice_flag = True)
    
    # Flag slices that cross match boundaries
    for i in range(len(df)-1):
        if df.MID[i] != df.MID[i+1]:
            df.loc[max(i+1-window_size, 0):i+1, "valid_slice_flag"] = False
    
    # Flag last window_size rows
    df.loc[max(len(df)-window_size, 0):len(df), "valid_slice_flag"] = False
    return df

# Apply to datasets
train = valid_slice_flag(train, window_size)
valid = valid_slice_flag(valid, window_size)
test  = valid_slice_flag(test, window_size)

#%% Encode dataframe to PyTorch tensor

def encode(df):
    tensors = []

    # Categorical features
    tensors.append(torch.tensor(df['act'].values, dtype=torch.long).view(-1, 1))
    tensors.append(torch.tensor(df['zone'].values, dtype=torch.long).view(-1, 1))

    # Continuous feature
    tensors.append(torch.tensor(df['deltaT'].values, dtype=torch.float32).view(-1, 1))

    # TID (always included)
    tensors.append(torch.tensor(df['TID'].values, dtype=torch.long).view(-1, 1))

    # Concatenate along feature dimension
    encode_df = torch.cat(tensors, dim=1)
    return encode_df

encode_train = encode(train)
encode_valid = encode(valid)
encode_test  = encode(test)

#%% model
def positional_encoding(seq_len, d_model, device="cpu"):
    # [seq_len, 1]
    positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)

    # [1, d_model/2]
    div_terms = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
        -(np.log(10000.0) / d_model)
    )

    # [seq_len, d_model]
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div_terms)  # even indices
    pe[:, 1::2] = torch.cos(positions * div_terms)  # odd indices

    return pe

#%%

class NMSTPP(nn.Module):
    def __init__(self,
                 action_emb_in, action_emb_out,
                 zone_emb_in, zone_emb_out,
                 TID_emb_in, TID_emb_out,
                 other_lin_in, other_lin_out,
                 input_features_len, hidden_dim, 
                 num_zones, num_actions, num_teams,
                 multihead_attention=1,
                 scale_grad_by_freq=True,
                 device='cpu',
                 dropout_p=0.3):
        super(NMSTPP, self).__init__()
        self.device = device

        # Embeddings
        self.emb_act  = nn.Embedding(action_emb_in, action_emb_out, scale_grad_by_freq=scale_grad_by_freq)
        self.emb_zone = nn.Embedding(zone_emb_in, zone_emb_out, scale_grad_by_freq=scale_grad_by_freq)
        self.emb_TID  = nn.Embedding(TID_emb_in, TID_emb_out, scale_grad_by_freq=scale_grad_by_freq)

        # Continuous features
        self.lin0 = nn.Linear(other_lin_in, other_lin_out)

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features_len,
            nhead=multihead_attention,
            batch_first=True,
            dim_feedforward=hidden_dim
        ).to(device)

        self.lin_relu = nn.Linear(input_features_len, input_features_len)

        # Δt head
        self.NN_deltaT  = nn.Sequential(
            nn.Linear(input_features_len, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.lin_deltaT = nn.Linear(256, 1)

        # Zone head
        self.NN_zone    = nn.Sequential(
            nn.Linear(256 + input_features_len, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.lin_zone   = nn.Linear(128, num_zones)

        # Action head
        self.NN_action  = nn.Sequential(
            nn.Linear(128 + 1 + num_zones + input_features_len, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.lin_action = nn.Linear(128, num_actions)

        # TID head
        self.NN_TID     = nn.Sequential(
            nn.Linear(128 + 128 + 1 + input_features_len, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.lin_TID    = nn.Linear(128, num_teams)

    def forward(self, X):
        # Separate inputs
        feed_action = X[:, :, 0].long()
        feed_zone   = X[:, :, 1].long()
        feed_deltaT = X[:, :, 2].unsqueeze(-1)
        feed_TID    = X[:, :, 3].long()

        # Embeddings + continuous
        X_act  = self.emb_act(feed_action)
        X_zone = self.emb_zone(feed_zone)
        X_TID  = self.emb_TID(feed_TID)
        X_cont = self.lin0(feed_deltaT.float())
        X_cat  = torch.cat((X_act, X_zone, X_TID, X_cont), dim=2).float()  # [B, L, D]

        # Add positional encoding
        seq_len = X_cat.size(1)
        d_model = X_cat.size(2)
        pos_enc = positional_encoding(seq_len, d_model, device=X.device)
        X_cat = X_cat + pos_enc.unsqueeze(0)  # broadcast across batch

        # Transformer
        X_cat_seqnet = self.encoder_layer(X_cat)
        x_relu = self.lin_relu(X_cat_seqnet[:, -1, :])

        # Δt head
        deltaT_hidden = self.NN_deltaT(x_relu)
        deltaT_out = F.softplus(self.lin_deltaT(deltaT_hidden))

        # Zone head
        zone_input = torch.cat((deltaT_hidden, x_relu), dim=1)
        zone_hidden = self.NN_zone(zone_input)
        zone_out = self.lin_zone(zone_hidden)

        # Action head
        action_input = torch.cat((zone_hidden, deltaT_out, zone_out, x_relu), dim=1)
        action_hidden = self.NN_action(action_input)
        action_out = self.lin_action(action_hidden)

        # TID head
        TID_input = torch.cat((action_hidden, zone_hidden, deltaT_out, x_relu), dim=1)
        TID_hidden = self.NN_TID(TID_input)
        TID_out = self.lin_TID(TID_hidden)

        # Combine outputs
        out = torch.cat((deltaT_out, zone_out, action_out, TID_out), dim=1)
        return out

#%% cost function

def cost_function(y, y_head, deltaT_weight, zone_weight, action_weight, TID_weight, device='cpu'):
    # targets 
    y_deltaT = y[:, 0].float()   # regression
    y_zone   = y[:, 1].long()    # classification
    y_action = y[:, 2].long()    # classification
    y_TID    = y[:, 3].long()    # classification

    # define number of classes dynamically
    num_zones = 9
    num_actions = 12
    num_teams = 14

    # predictions (split dynamically by number of classes)
    idx_start = 0
    y_head_deltaT = y_head[:, idx_start] # scalar regression
    idx_start += 1

    y_head_zone = y_head[:, idx_start:idx_start + num_zones]
    idx_start += num_zones

    y_head_action = y_head[:, idx_start:idx_start + num_actions]
    idx_start += num_actions

    y_head_TID = y_head[:, idx_start:idx_start + num_teams]

    # compute losses
    loss_deltaT = torch.sqrt(torch.mean((y_deltaT - y_head_deltaT) ** 2))

    # Use global weight tensors
    CEL_zone   = nn.CrossEntropyLoss(weight=weight_zone_class.float().to(device), reduction="mean")
    loss_zone  = CEL_zone(y_head_zone, y_zone)

    CEL_action = nn.CrossEntropyLoss(weight=weight_action_class.float().to(device), reduction="mean")
    loss_action = CEL_action(y_head_action, y_action)

    CEL_TID    = nn.CrossEntropyLoss(weight=weight_TID_class.float().to(device), reduction="mean")
    loss_TID   = CEL_TID(y_head_TID, y_TID)

    # weighted sum
    total_loss = (
        deltaT_weight * loss_deltaT +
        zone_weight   * loss_zone +
        action_weight * loss_action +
        3*TID_weight  * loss_TID
    )

    return total_loss, loss_deltaT, loss_zone, loss_action, loss_TID

#%%
    
class SequenceDataset(Dataset):
    def __init__(self, df, window_size):
        self.window = window_size
        #self.mode = mode

        # Encode features into tensor
        self.Xflat = encode(df)  # [N, F] tensor

        # Targets: deltaT (float), zone/action/TID (long)
        y_values = df[['deltaT', 'zone', 'act', 'TID']].copy()
        self.Yall = torch.zeros((len(df), 4), dtype=torch.float32)
        self.Yall[:, 0] = torch.tensor(y_values['deltaT'].values, dtype=torch.float32)
        self.Yall[:, 1] = torch.tensor(y_values['zone'].values, dtype=torch.float32)
        self.Yall[:, 2] = torch.tensor(y_values['act'].values, dtype=torch.float32)
        self.Yall[:, 3] = torch.tensor(y_values['TID'].values, dtype=torch.float32)

        # Determine valid indices (targets) based on valid_slice_flag and window
        flags = df['valid_slice_flag'].to_numpy()
        self.idxs = [i for i in range(window_size, len(df)) if flags[i]]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, j):
        i = self.idxs[j]
        # Input sequence: previous `window_size` rows
        Xwin = self.Xflat[i-self.window:i, :]  # [window_size, feature_dim]
        Y    = self.Yall[i, :]                  # [4] tensor (deltaT, zone, act, TID)
        return Xwin, Y, i

# Prepare datasets
train_dataset = SequenceDataset(train, window_size)
valid_dataset = SequenceDataset(valid, window_size)
test_dataset  = SequenceDataset(test,  window_size)

#%% DataLoader creation

# Prepare datasets and dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=training_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=valid_batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
)

#%% training,valid,test one epoch

def model_epoch(dataloader, model, optimiser, scheduler=None, epochtype="train", grad_clip=None):
    model.train() if epochtype == "train" else model.eval()

    # rolling mean accumulators
    loss_rolling = 0.0
    loss_zone_rolling = 0.0
    loss_action_rolling = 0.0
    loss_TID_rolling = 0.0
    loss_deltaT_rolling = 0.0

    for batch_idx, (X, Y, _) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        with torch.set_grad_enabled(epochtype == "train"):
            pred = model(X)

            total_loss, deltaT_loss, zone_loss, action_loss, TID_loss = cost_function(
                Y, pred,
                deltaT_weight=1,
                zone_weight=1,
                action_weight=1,
                TID_weight=3
            )

            # update rolling means
            loss_rolling += (total_loss - loss_rolling) / (batch_idx + 1)
            loss_zone_rolling += (zone_loss - loss_zone_rolling) / (batch_idx + 1)
            loss_action_rolling += (action_loss - loss_action_rolling) / (batch_idx + 1)
            loss_TID_rolling += (TID_loss - loss_TID_rolling) / (batch_idx + 1)
            loss_deltaT_rolling += (deltaT_loss - loss_deltaT_rolling) / (batch_idx + 1)

            if epochtype == "train":
                optimiser.zero_grad()
                total_loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimiser.step()

    # scheduler step per epoch
    if scheduler is not None and epochtype == "train":
        scheduler.step()

    # detach and move to CPU for logging
    loss_rolling = loss_rolling.detach().cpu().item()
    loss_zone_rolling = loss_zone_rolling.detach().cpu().item()
    loss_action_rolling = loss_action_rolling.detach().cpu().item()
    loss_TID_rolling = loss_TID_rolling.detach().cpu().item()
    loss_deltaT_rolling = loss_deltaT_rolling.detach().cpu().item()

    # print summary
    print("Epoch ended")
    print(f"Total Loss:       {loss_rolling:.6f}, ln(1+loss): {np.log1p(loss_rolling):.6f}")
    print(f"Zone CE Loss:     {loss_zone_rolling:.6f}, ln(1+loss): {np.log1p(loss_zone_rolling):.6f}")
    print(f"Action CE Loss:   {loss_action_rolling:.6f}, ln(1+loss): {np.log1p(loss_action_rolling):.6f}")
    print(f"TID CE Loss:      {loss_TID_rolling:.6f}, ln(1+loss): {np.log1p(loss_TID_rolling):.6f}")
    print(f"Δt RMSE Loss:     {loss_deltaT_rolling:.6f}, ln(1+loss): {np.log1p(loss_deltaT_rolling):.6f}")

    return (
        loss_rolling,
        loss_zone_rolling,
        loss_action_rolling,
        loss_TID_rolling,
        loss_deltaT_rolling
    )

#%% Possibly combining the training and validation into one part

def model_train(
    model,
    train_loader,
    valid_loader,
    epochs,
    save_path="best_model.pth",
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    patience=5
):
    torch.cuda.empty_cache()
    gc.collect()

    optimiser = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        eps=1e-16,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.1, patience=3
    )

    history = pd.DataFrame(columns=[
        "epoch",
        "trn_total", "trn_CEL_zone", "trn_CEL_action", "trn_CEL_TID", "trn_RMSE_deltaT",
        "val_total", "val_CEL_zone", "val_CEL_action", "val_CEL_TID", "val_RMSE_deltaT"
    ])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    time_start = datetime.now()

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\nEpoch {epoch}\n-------------------------------")

        # --- Training ---
        train_loss = model_epoch(
            train_loader, model, optimiser, scheduler=None,
            epochtype="train", grad_clip=grad_clip
        )

        # --- Validation ---
        with torch.no_grad():
            val_loss = model_epoch(
                valid_loader, model, optimiser=None, scheduler=None,
                epochtype="val"
            )

        # --- Record history ---
        row = pd.DataFrame([[epoch, *train_loss, *val_loss]], columns=history.columns)
        history = pd.concat([history, row], ignore_index=True)

        # Step scheduler based on validation loss
        scheduler.step(val_loss[0])

        # --- Check for improvement ---
        if val_loss[0] < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.6f} -> {val_loss[0]:.6f}). Saving model.")
            best_val_loss = val_loss[0]
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # --- Early stopping ---
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # --- Load best model and save ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path} with validation loss {best_val_loss:.6f}.")

    trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_time = (datetime.now() - time_start).total_seconds()

    return history, train_time, trainable_params_num, model

#%% Test function

def model_test(model, test_loader):
    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
    with torch.no_grad():
        test_loss = model_epoch(
            dataloader=test_loader,
            model=model,
            optimiser=None,
            scheduler=None,
            epochtype="test"
        )

    test_loss_dict = {
        "total_loss": test_loss[0],
        "CEL_zone": test_loss[1],
        "CEL_action": test_loss[2],
        "CEL_TID": test_loss[3],
        "RMSE_deltaT": test_loss[4]
    }

    print(f"Test Loss: {test_loss_dict['total_loss']:.6f}")
    return test_loss_dict

#%% function for predict

def predict(model, dataloader, device='cuda', verbose=True):
    model.eval()
    all_deltaT, all_zone, all_action, all_TID = [], [], [], []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (X, Y, idx) in enumerate(dataloader):
            X = X.to(device)
            pred = model(X)

            # Split predictions by head
            all_deltaT.append(pred[:, 0].cpu())
            all_zone.append(pred[:, 1:10].cpu())
            all_action.append(pred[:, 10:22].cpu())
            all_TID.append(pred[:, 22:].cpu())

            all_indices.append(idx)

            if verbose and batch_idx % 500 == 0:
                print(f"Batch {batch_idx} / {len(dataloader)} processed.")

    # Concatenate batch-wise results
    all_indices = torch.cat(all_indices, dim=0).numpy()
    all_deltaT  = torch.cat(all_deltaT, dim=0)
    all_zone    = torch.cat(all_zone, dim=0)
    all_action  = torch.cat(all_action, dim=0)
    all_TID     = torch.cat(all_TID, dim=0)

    return all_deltaT, all_zone, all_action, all_TID, all_indices

#%%

# Extract unique classes from the datasets
action_classes = train['act'].unique().tolist()
zone_classes   = train['zone'].unique().tolist()
team_classes   = train['TID'].unique().tolist()

# --- Embedding sizes / model input dimensions ---
action_emb_in, action_emb_out = len(action_classes), len(action_classes)
zone_emb_in, zone_emb_out     = len(zone_classes), len(zone_classes)
TID_emb_in, TID_emb_out       = len(team_classes), len(team_classes)

# Continuous features (e.g., deltaT)
other_lin_in, other_lin_out = 1, 1  

# Total input features for Transformer
input_features_len = other_lin_out + action_emb_out + zone_emb_out + TID_emb_out

# Number of classes per prediction head
num_zones   = len(zone_classes)
num_actions = len(action_classes)
num_teams   = len(team_classes)

# Device selection
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#%% ---------- TRAIN MULTIPLE MODELS ----------

# --- Model 1 ---
hidden_dim = 1024
multihead_attention = 1

model1 = NMSTPP(
    action_emb_in, action_emb_out,
    zone_emb_in, zone_emb_out,
    TID_emb_in, TID_emb_out,
    other_lin_in, other_lin_out,
    input_features_len, hidden_dim,
    num_zones, num_actions, num_teams,
    multihead_attention=multihead_attention,
    device=device
).to(device)

history1, train_time1, trainable_params_num1, model1 = model_train(
    model1, train_loader, valid_loader, epochs, save_path="model1.pth"
)
history1.to_csv("model1_training_history_TID_included.csv", index=False)

#%%

# --- Model 2 ---
hidden_dim = 1024
multihead_attention = 2

model2 = NMSTPP(
    action_emb_in, action_emb_out,
    zone_emb_in, zone_emb_out,
    TID_emb_in, TID_emb_out,
    other_lin_in, other_lin_out,
    input_features_len, hidden_dim,
    num_zones, num_actions, num_teams,
    multihead_attention=multihead_attention,
    device=device
).to(device)

history2, train_time2, trainable_params_num2, model2 = model_train(
    model2, train_loader, valid_loader, epochs, save_path="model2.pth"
)
history2.to_csv("model2_training_history_TID_included.csv", index=False)

#%%

# --- Model 3 ---
hidden_dim = 512
multihead_attention = 1

model3 = NMSTPP(
    action_emb_in, action_emb_out,
    zone_emb_in, zone_emb_out,
    TID_emb_in, TID_emb_out,
    other_lin_in, other_lin_out,
    input_features_len, hidden_dim,
    num_zones, num_actions, num_teams,
    multihead_attention=multihead_attention,
    device=device
).to(device)

history3, train_time3, trainable_params_num3, model3 = model_train(
    model3, train_loader, valid_loader, epochs, save_path="model3.pth"
)
history3.to_csv("model3_training_history_TID_included.csv", index=False)

#%%

# --- Model 4 ---
hidden_dim = 512
multihead_attention = 2

model4 = NMSTPP(
    action_emb_in, action_emb_out,
    zone_emb_in, zone_emb_out,
    TID_emb_in, TID_emb_out,
    other_lin_in, other_lin_out,
    input_features_len, hidden_dim,
    num_zones, num_actions, num_teams,
    multihead_attention=multihead_attention,
    device=device
).to(device)

history4, train_time4, trainable_params_num4, model4 = model_train(
    model4, train_loader, valid_loader, epochs, save_path="model4.pth"
)
history4.to_csv("model4_training_history_TID_included.csv", index=False)

#%% ---------- Select Best Model ----------

model_histories = {
    "model1": (history1, model1),
    "model2": (history2, model2),
    "model3": (history3, model3),
    "model4": (history4, model4)
}

best_model_name = None
best_val_loss = float('inf')

for name, (hist, mdl) in model_histories.items():
    min_val = hist['val_total'].min()
    if min_val < best_val_loss:
        best_val_loss = min_val
        best_model_name = name

best_model = model_histories[best_model_name][1]
print(f"Best model based on validation loss: {best_model_name} ({best_val_loss:.6f})")

#%% ---------- Predict on Test Set ----------

deltaT_pred, zone_pred, action_pred, TID_pred, all_indices = predict(
    best_model, test_loader, device=device, verbose=True
)

# Combine predictions into DataFrame
test_pred_sub = pd.DataFrame(
    torch.cat([deltaT_pred.unsqueeze(1), zone_pred, action_pred, TID_pred], dim=1).numpy()
)
test_pred_sub['original_index'] = all_indices
test_pred_sub.to_csv("pred_best_model_TID_included.csv", index=False)

