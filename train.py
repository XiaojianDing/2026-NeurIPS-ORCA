import os
import torch
import numpy as np
import random
import argparse
from network import Network
from metric import valid
from loss import ContrastiveLoss
from dataloader import load_data
import torch.nn.functional as F
import torch.nn as nn

Dataname = 'Synthetic3d'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--temperature", default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

lam = 1.0


if args.dataset == "Synthetic3d":
    args.con_epochs = 100;
    seed = 1;
    alpha = 1.0; beta = 10.0; gamma = 1.0

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)

def compute_view_value(rs, H, view, num_heads=4):
    device = H.device
    N, d = H.shape
    d_v = rs[0].shape[1]
    mha = nn.MultiheadAttention(embed_dim=d_v, num_heads=num_heads, batch_first=True).to(device)
    Q = H.unsqueeze(1)
    Q = nn.Linear(d, d_v).to(device)(Q)

    view_scores = []
    for v in range(len(rs)):
        K = rs[v].unsqueeze(1)
        attn_output, attn_weights = mha(Q, K, K)
        score = attn_weights.mean().item()
        view_scores.append(score)

    w = torch.tensor(view_scores, device=device)
    w = F.softmax(w, dim=0)
    return w

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    model.train()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        xs = [x.to(device) for x in xs]
        optimizer.zero_grad()
        xrs, *_ = model(xs)
        loss_list = [criterion(xs[v], xrs[v]) for v in range(view)]
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f'Epoch {epoch}, Pretrain Loss: {tot_loss / len(data_loader):.6f}')


def contrastive_train(epoch, lam=1.0, alpha=1.0, beta=1.0, gamma=1.0):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    model.train()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        xs = [x.to(device) for x in xs]
        optimizer.zero_grad()
        xrs, zs, rs, H, new_rs, intermediate_rs = model(xs)
        loss_list = []

        with torch.no_grad():
            w_new = compute_view_value(new_rs, H, len(new_rs))
        for v in range(len(new_rs)):
            loss_list.append(lam * alpha * contrastiveloss(H, new_rs[v], w_new[v].item()))

        with torch.no_grad():
            w_rs = compute_view_value(rs, H, view)
        for v in range(view):
            loss_list.append(lam * beta * contrastiveloss(H, rs[v], w_rs[v].item()))
            loss_list.append(mse(xs[v], xrs[v]))

        if len(intermediate_rs) > 0:
            with torch.no_grad():
                w_inter = compute_view_value(intermediate_rs, H, len(intermediate_rs))
            for v in range(len(intermediate_rs)):
                loss_list.append(lam * gamma * contrastiveloss(H, intermediate_rs[v], w_inter[v].item()))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f'Epoch {epoch}, ConTrain Loss: {tot_loss / len(data_loader):.6f}')
    return tot_loss / len(data_loader)

if not os.path.exists('./models'):
    os.makedirs('./models')

setup_seed(seed)

model = Network(view, dims, args.feature_dim, args.high_feature_dim, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

best_acc, best_nmi, best_ari, best_pur = 0, 0, 0, 0
epoch = 1

model_save_path = f'./models/{args.dataset}.pth'

while epoch <= args.pre_epochs:
    pretrain(epoch)
    epoch += 1

while epoch <= args.pre_epochs + args.con_epochs:
    loss = contrastive_train(epoch, lam, alpha, beta, gamma)

    acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)

    if acc > best_acc:
        best_acc, best_nmi, best_ari, best_pur = acc, nmi, ari, pur
        torch.save(model.state_dict(), model_save_path)
    epoch += 1

print("\n" + "="*50)

print(f"ACC = {best_acc:.4f}")
print(f"NMI = {best_nmi:.4f}")
print(f"ARI = {best_ari:.4f}")
print(f"PUR = {best_pur:.4f}")