import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloader import normal_img_Dataset
from utils import save_state_dict, load_state_dict
from models.model import AnoNAViLa_Model

def cos_sim(proj1, proj2=None):
    if proj2 is None:
        N = len(proj1)
        cosine_sim_matrix = F.cosine_similarity(proj1.unsqueeze(1), proj1.unsqueeze(0), dim=-1)
        triu_indices = torch.triu_indices(N, N, offset=1)
        pairwise_cosine_sims = cosine_sim_matrix[triu_indices[0], triu_indices[1]]
        return torch.mean(torch.exp(pairwise_cosine_sims))
    else:
        cosine_sim_matrix = F.cosine_similarity(proj1.unsqueeze(1), proj2.unsqueeze(0), dim=-1)
        return torch.mean(torch.exp(cosine_sim_matrix))

def loss_function(h_n, h_a):
    h_n_intra = cos_sim(h_n)
    h_a_intra = cos_sim(h_a)
    h_n_h_a_inter = cos_sim(h_n, h_a)

    loss = - torch.log((h_n_intra + h_a_intra) / (h_n_intra + h_a_intra + h_n_h_a_inter)).unsqueeze(dim=0)
    return loss

def train(args):
    prefix = args["prefix"]
    torch.manual_seed(args["random_seed"])
    device = args["device"]
    logger = args["logger"]
    accum_batches = args["accumulate_batches"]

    os.makedirs(f'./pth/{prefix}', exist_ok=True)

    # training normal image dataloader
    train_dataset = normal_img_Dataset(top_path=args["train_imgs_path"], transform=args["VLM_img_preprocess"])
    train_dataloader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True)

    # model, optimizer
    model = AnoNAViLa_Model(
        VLM=args["VLM"],
        embedding_dim=args["embs_dim"],
        normal_text_embs=args["normal_text_embs"],
        abnormal_text_embs=args["abnormal_text_embs"],
        device=device
    ).to(device)
    if args["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.MLP.parameters(), lr=args["learning_rate"])
    else:
        optimizer = None
        logger.error(f'{args["optimizer"]} is not a supported optimizer.')

    # train
    model.train()
    for epoch in range(1, args["epochs"] + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Loading train images batch"), start=1):
            imgs, _ = batch
            imgs = imgs.to(device)

            h_n, h_a = model.forward(imgs)

            loss = loss_function(h_n, h_a)
            loss.backward()
            accum_loss += loss.item()

            if batch_idx % accum_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / accum_batches
                logger.info(f"[Epoch {epoch} batch_idx {batch_idx}] Average Loss over last {accum_batches} batches: {avg_loss:.4f}")
                accum_loss = 0.0
                save_state_dict(f'./pth/{prefix}/{epoch}_{batch_idx}.pth', epoch, model, optimizer)

        if batch_idx % accum_batches != 0:
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = accum_loss / (batch_idx % accum_batches)
            logger.info(f"[Epoch {epoch} Final batch_idx {batch_idx}] Average Loss over last {batch_idx % accum_batches} batches: {avg_loss:.4f}")
            save_state_dict(f'./pth/{prefix}/{epoch}_{batch_idx}.pth', epoch, model, optimizer)

    args["trained_model"] = model

def load_trained_model(args):
    device = args["device"]

    model = AnoNAViLa_Model(
        VLM=args["VLM"],
        embedding_dim=args["embs_dim"],
        normal_text_embs=args["normal_text_embs"],
        abnormal_text_embs=args["abnormal_text_embs"],
        device=device
    ).to(device)
    load_state_dict(args["trained_model_pth"], model)
    args["logger"].info(f"Model successfully loaded from {args['trained_model_pth']}")

    args["trained_model"] = model

def prepare_centroids(args):
    # validation normal image dataloader
    val_dataset = normal_img_Dataset(top_path=args["val_imgs_path"], transform=args["VLM_img_preprocess"])
    val_dataloader = DataLoader(val_dataset, batch_size=args["train_batch_size"], shuffle=False)

    model = args["trained_model"]
    model.compute_centroids(val_dataloader)

    os.makedirs(f'./pth/{args["prefix"]}', exist_ok=True)
    save_state_dict(f'./pth/{args["prefix"]}/model_with_centroids.pth', None, model)
    args["logger"].info(f'Centroids are computed and saved model to ./pth/{args["prefix"]}/model_with_centroids.pth')
    args["trained_model"] = model