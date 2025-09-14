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

#%% Define categorical encoders/decoders for actions

# List of all possible actions
actions = [
    'Pass', 'Carry', 'Miscontrol', 'Ball Recovery', 'Dribble', 
    'Shot', 'Clearance', 'Dispossessed', 'Interception', 'Shield', 'PC1', 'PC0'
]

# Create mapping dictionaries
char2idx = {action: idx for idx, action in enumerate(actions)}
idx2char = {idx: action for action, idx in char2idx.items()}

# Number of unique actions
num_actions = len(actions)

train['act'] = train['act'].replace(char2idx).astype(np.int64)
valid['act'] = valid['act'].replace(char2idx).astype(np.int64)
test['act']  = test['act'].replace(char2idx).astype(np.int64)

#%% Encode TEAM IDs (for training only)

# Factorize TID in train set
train_team_codes, team_uniques = pd.factorize(train['TID'])
train['TID'] = train_team_codes
num_teams = len(team_uniques)  # number of unique teams in training data

# Store mapping from team → code (for training)
team_to_code = {t: i for i, t in enumerate(team_uniques)}

def map_tid(tid):
    return team_to_code.get(tid, num_teams)

# Apply mapping to validation and test sets
valid['TID'] = valid['TID'].map(map_tid)

def map_team(df):
    return df['TID'].map(team_to_code).fillna(0).astype(int)

valid['TID_encoded'] = map_team(valid)  
test['TID_encoded']  = map_team(test)


#%% Zero-index zone labels (convert from 1-9 to 0-8)

train['zone'] -= 1
valid['zone'] -= 1
test['zone']  -= 1

num_zones = 9  # useful for defining embedding layer later

#%% Specify input and target variables

all_input_vars = ['deltaT', 'zone', 'act', 'TID']
all_target_vars = ['deltaT', 'zone', 'act', 'TID']

def get_input_vars(mode='train'):
    if mode == 'train':
        return all_input_vars
    else:
        return [col for col in all_input_vars if col != 'TID']

#%% Specify loss function weighting

def get_class_weights(train):
    num_zones = train['zone'].nunique()
    zone_classes = np.arange(num_zones)
    weight_zone = torch.tensor(compute_class_weight(
        class_weight="balanced",
        classes=zone_classes,
        y=train['zone'].values
    ), dtype=torch.float32)

    num_actions = train['act'].nunique()
    action_classes = np.arange(num_actions)
    weight_action = torch.tensor(compute_class_weight(
        class_weight="balanced",
        classes=action_classes,
        y=train['act'].values
    ), dtype=torch.float32)
    
    
    train_num_teams = train['TID'].nunique()  # = 14
    weight_TID_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(train_num_teams),
        y=train['TID'].values
    )

    weight_TID_values = weight_TID_values * 4   

    full_weights = np.ones(num_teams)       
    full_weights[:train_num_teams] = weight_TID_values

    weight_TID = torch.tensor(full_weights, dtype=torch.float32)
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

def encode(df, mode='train'):
    tensors = []

    # Categorical features
    tensors.append(torch.tensor(df['act'].values, dtype=torch.long).view(-1, 1))
    tensors.append(torch.tensor(df['zone'].values, dtype=torch.long).view(-1, 1))

    # Continuous feature
    tensors.append(torch.tensor(df['deltaT'].values, dtype=torch.float32).view(-1, 1))

    # TID (only for training)
    if mode in ['train', 'val']:
        tensors.append(torch.tensor(df['TID'].values, dtype=torch.long).view(-1, 1))

    # Concatenate along feature dimension
    encode_df = torch.cat(tensors, dim=1)
    return encode_df

encode_train = encode(train, mode='train')
encode_valid = encode(valid, mode='val')
encode_test  = encode(test, mode='test')

#%% model

def positional_encoding(seq_len, d_model, device="cpu"):
    # Positions: [seq_len, 1]
    positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)

    # Divisor terms: [1, d_model/2]
    div_terms = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
        -(np.log(10000.0) / d_model)
    )

    # Encoding matrix
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div_terms)  # even dims
    pe[:, 1::2] = torch.cos(positions * div_terms)  # odd dims

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
                 device='cuda',
                 dropout_p=0.3):
        super(NMSTPP, self).__init__()
        self.device = device

        # Embeddings
        self.emb_act  = nn.Embedding(action_emb_in, action_emb_out, scale_grad_by_freq=scale_grad_by_freq)
        self.emb_zone = nn.Embedding(zone_emb_in, zone_emb_out, scale_grad_by_freq=scale_grad_by_freq)
        self.emb_TID  = nn.Embedding(TID_emb_in, TID_emb_out, scale_grad_by_freq=scale_grad_by_freq)

        # Continuous features
        self.lin0 = nn.Linear(other_lin_in, other_lin_out)

        # Transformer
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
        # Inputs
        feed_action = X[:, :, 0].long()
        feed_zone   = X[:, :, 1].long()
        feed_deltaT = X[:, :, 2].unsqueeze(-1)

        if X.shape[2] == 4:
            feed_TID = X[:, :, 3].long()
            X_TID = self.emb_TID(feed_TID)
        else:
            X_TID = torch.zeros(X.size(0), X.size(1), self.emb_TID.embedding_dim, device=X.device)

        # Embeddings + continuous
        X_act  = self.emb_act(feed_action)
        X_zone = self.emb_zone(feed_zone)
        X_cont = self.lin0(feed_deltaT.float())
        X_cat  = torch.cat((X_act, X_zone, X_TID, X_cont), dim=2).float()

        # Positional encoding 
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

def cost_function(y, y_head, deltaT_weight, zone_weight, action_weight, TID_weight):
    # targets 
    y_deltaT = y[:, 0].float()   # regression
    y_zone   = y[:, 1].long()    # classification (9 classes)
    y_action = y[:, 2].long()    # classification (12 classes)
    y_TID    = y[:, 3].long()    # classification (14 classes)

    # predictions (split by head)
    y_head_deltaT = y_head[:, 0].float()          # [batch]
    y_head_zone   = y_head[:, 1:10]              # [batch, 9]
    y_head_action = y_head[:, 10:22]             # [batch, 12]
    y_head_TID    = y_head[:, 22:36]             # [batch, 14]
        
    # losses
    # Δt → RMSE
    loss_deltaT = torch.sqrt(torch.mean((y_deltaT - y_head_deltaT) ** 2))

    # Zone → CrossEntropy
    CEL_zone = nn.CrossEntropyLoss(weight=weight_zone_class.float().to(device), reduction="mean")
    loss_zone = CEL_zone(y_head_zone, y_zone)

    # Action → CrossEntropy
    CEL_action = nn.CrossEntropyLoss(weight=weight_action_class.float().to(device), reduction="mean")
    loss_action = CEL_action(y_head_action, y_action)

    # TID → CrossEntropy
    CEL_TID = nn.CrossEntropyLoss(weight=weight_TID_class.float().to(device), reduction="mean")
    loss_TID = CEL_TID(y_head_TID, y_TID)

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
    def __init__(self, df, window_size, mode='train'):
        self.window = window_size
        self.mode = mode

        # features per row 
        Xflat = encode(df, mode=mode)                 
        Yall  = torch.tensor(
            df[['deltaT', 'zone', 'act', 'TID']].values,
            dtype=torch.float32
        )                                             

        # pick valid end indices i (the target is row i; inputs are rows [i-window:i])
        flags = df['valid_slice_flag'].to_numpy()
        self.idxs = [i for i in range(len(df)) if i >= window_size and flags[i]]

        self.Xflat = Xflat
        self.Yall  = Yall

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, j):
        i = self.idxs[j]
        Xwin = self.Xflat[i-self.window:i, :]
        Y    = self.Yall[i, :]
        return Xwin, Y, i

# Prepare arrays
train_dataset = SequenceDataset(train, window_size, mode='train')
valid_dataset = SequenceDataset(valid, window_size, mode='val')
test_dataset  = SequenceDataset(test,  window_size, mode='test')

#%% DataLoader creation

    # Prepare datasets and dataloaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=training_batch_size,
                              num_workers=num_workers, drop_last=True)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=valid_batch_size,
                              num_workers=num_workers, drop_last=False)
test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size,
                              num_workers=num_workers, drop_last=False)

#%% training,valid,test one epoch

def model_epoch(dataloader, model, optimiser, scheduler=None, epochtype="train", grad_clip=None):
    if epochtype == "train":
        model.train()
    else:
        model.eval()
    
    loss_rollingmean = 0.0
    lossCEL_zone_rollingmean = 0.0
    lossCEL_action_rollingmean = 0.0
    lossCEL_TID_rollingmean = 0.0
    lossRMSE_rollingmean = 0.0

    for batch_idx, (X, Y, _) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        with torch.set_grad_enabled(epochtype == "train"):
            pred = model(X)

            Loss, RMSE_deltaT, CEL_zone, CEL_action, CEL_TID = cost_function(
                Y, pred,
                deltaT_weight=1,
                zone_weight=1,
                action_weight=1,
                TID_weight=3
            )

            # rolling mean update
            loss_rollingmean = loss_rollingmean + (Loss - loss_rollingmean) / (batch_idx + 1)
            lossCEL_zone_rollingmean = lossCEL_zone_rollingmean + (CEL_zone - lossCEL_zone_rollingmean) / (batch_idx + 1)
            lossCEL_action_rollingmean = lossCEL_action_rollingmean + (CEL_action - lossCEL_action_rollingmean) / (batch_idx + 1)
            lossCEL_TID_rollingmean = lossCEL_TID_rollingmean + (CEL_TID - lossCEL_TID_rollingmean) / (batch_idx + 1)
            lossRMSE_rollingmean = lossRMSE_rollingmean + (RMSE_deltaT - lossRMSE_rollingmean) / (batch_idx + 1)

            if epochtype == "train":
                optimiser.zero_grad()
                Loss.backward()
                
                # gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimiser.step()

    # scheduler step per epoch
    if scheduler is not None and epochtype == "train":
        scheduler.step()

    # detach for reporting
    loss_rollingmean = loss_rollingmean.detach().cpu().item()
    lossCEL_zone_rollingmean = lossCEL_zone_rollingmean.detach().cpu().item()
    lossCEL_action_rollingmean = lossCEL_action_rollingmean.detach().cpu().item()
    lossCEL_TID_rollingmean = lossCEL_TID_rollingmean.detach().cpu().item()
    lossRMSE_rollingmean = lossRMSE_rollingmean.detach().cpu().item()

    # epoch summary
    print("Epoch ended")
    print(f"Epoch loss: mean: {loss_rollingmean:.6f}, ln(1+loss): {np.log1p(loss_rollingmean):.6f}")
    print(f"Epoch CEloss_zone:   mean: {lossCEL_zone_rollingmean:.6f}, ln(1+loss): {np.log1p(lossCEL_zone_rollingmean):.6f}")
    print(f"Epoch CEloss_action: mean: {lossCEL_action_rollingmean:.6f}, ln(1+loss): {np.log1p(lossCEL_action_rollingmean):.6f}")
    print(f"Epoch CEloss_TID:    mean: {lossCEL_TID_rollingmean:.6f}, ln(1+loss): {np.log1p(lossCEL_TID_rollingmean):.6f}")
    print(f"Epoch MSEloss:       mean: {lossRMSE_rollingmean:.6f}, ln(1+loss): {np.log1p(lossRMSE_rollingmean):.6f}")

    return (
        loss_rollingmean,
        lossCEL_zone_rollingmean,
        lossCEL_action_rollingmean,
        lossCEL_TID_rollingmean,
        lossRMSE_rollingmean
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
    patience=3
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
    time_start = datetime.now()

    best_model_state = None  

    for t in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\nEpoch {t}\n-------------------------------")

        # Training
        train_loss = model_epoch(train_loader, model, optimiser, None, "train", grad_clip=grad_clip)

        # Validation
        with torch.no_grad():
            val_loss = model_epoch(valid_loader, model, None, None, "val")

        # Record history
        row = pd.DataFrame([[t, *train_loss, *val_loss]], columns=history.columns)
        history = pd.concat([history, row], ignore_index=True)

        # Step LR scheduler
        scheduler.step(val_loss[0])

        # Check for improvement
        if val_loss[0] < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.6f} -> {val_loss[0]:.6f}). Saving model.")
            best_val_loss = val_loss[0]
            best_model_state = model.state_dict()  # store best weights
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # Load best model before returning
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
    import gc; gc.collect()

    with torch.no_grad():
        test_loss = model_epoch(test_loader, model, optimiser=None, scheduler=None, epochtype="test")

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
        for batch_idx, (X, Y, idx) in enumerate(dataloader):  # unpack idx
            X = X.to(device)
            pred = model(X)

            all_deltaT.append(pred[:, 0].cpu())
            all_zone.append(pred[:, 1:10].cpu())
            all_action.append(pred[:, 10:22].cpu())
            all_TID.append(pred[:, 22:].cpu())

            all_indices.append(idx)  # collect original indices

            if verbose and batch_idx % 500 == 0:
                print(f"Batch {batch_idx} / {len(dataloader)}")

    # concatenate batch-wise predictions and indices
    all_indices = torch.cat(all_indices, dim=0).numpy()
    all_deltaT  = torch.cat(all_deltaT, dim=0)
    all_zone    = torch.cat(all_zone, dim=0)
    all_action  = torch.cat(all_action, dim=0)
    all_TID     = torch.cat(all_TID, dim=0)

    return all_deltaT, all_zone, all_action, all_TID, all_indices

#%%

# Extract unique classes from the CSVs 
action_classes = train['act'].unique().tolist()
zone_classes   = train['zone'].unique().tolist()
team_classes = train['TID'].unique().tolist()

# Embedding / model parameters
action_emb_in, action_emb_out = len(action_classes), len(action_classes)
zone_emb_in, zone_emb_out     = len(zone_classes), len(zone_classes)
TID_emb_in, TID_emb_out       = len(team_classes), len(team_classes)

other_lin_in, other_lin_out = 1, 1  # e.g., deltaT feature
input_features_len = other_lin_out + action_emb_out + zone_emb_out + TID_emb_out

num_zones   = len(zone_classes)
num_actions = len(action_classes)
num_teams   = len(team_classes)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#%%
# ---------- TRAIN MODEL ----------

hidden_dim = 1024
multihead_attention = 1

# Example for Model 1
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

#%%

history_csv_path_1 = "model1_training_history.csv"
history1.to_csv(history_csv_path_1, index=False)

#%%
hidden_dim = 1024
multihead_attention = 2

# Model 2
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

#%%

history_csv_path_2 = "model2_training_history.csv"
history2.to_csv(history_csv_path_2, index=False)

#%%

hidden_dim = 512
multihead_attention = 1

# Model 3
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

#%%

history_csv_path_3 = "model3_training_history.csv"
history3.to_csv(history_csv_path_3, index=False)

#%%

hidden_dim = 512
multihead_attention = 2

# Model 4
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

#%%

history_csv_path_4 = "model4_training_history.csv"
history4.to_csv(history_csv_path_4, index=False)

#%%

# Collect histories and corresponding models
model_histories = {
    "model1": (history1, model1),
    "model2": (history2, model2),
    "model3": (history3, model3),
    "model4": (history4, model4)
}

# Find the model with the lowest validation total loss
best_model_name = None
best_val_loss = float('inf')
for name, (hist, mdl) in model_histories.items():
    min_val = hist['val_total'].min()  # lowest validation total loss
    if min_val < best_val_loss:
        best_val_loss = min_val
        best_model_name = name

best_model = model_histories[best_model_name][1]
print(f"Best model based on validation loss: {best_model_name} ({best_val_loss:.6f})")

# Run predictions using the automatically selected best model
deltaT_pred, zone_pred, action_pred, TID_pred, all_indices = predict(
    best_model, test_loader, device=device, verbose=True
)

# Convert predictions to DataFrame and save
test_pred_sub = pd.DataFrame(
    torch.cat([deltaT_pred.unsqueeze(1), zone_pred, action_pred, TID_pred], dim=1).numpy()
)
test_pred_sub['original_index'] = all_indices
test_pred_sub.to_csv("pred_best_model.csv", index=False)
