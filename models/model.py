import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, embedding_dim=512):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.linear3 = nn.Linear(embedding_dim // 2, embedding_dim // 4)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

def text_augmented_img_embs(img_embs, normal_text_embs, abnormal_text_embs, VLM, MLP):
    img_nword_sims = img_embs @ normal_text_embs.T * VLM.logit_scale.exp()
    n_word_weights = img_nword_sims.softmax(dim=-1)
    img_aword_sims = img_embs @ abnormal_text_embs.T * VLM.logit_scale.exp()
    a_word_weights = img_aword_sims.softmax(dim=-1)

    h_n = []
    h_a = []
    for i in range(img_embs.shape[0]):
        img_emb = img_embs[i].unsqueeze(0)

        replicated_img_emb_n = img_emb.repeat(normal_text_embs.shape[0], 1)
        replicated_img_emb_a = img_emb.repeat(abnormal_text_embs.shape[0], 1)

        scaled_text_n = normal_text_embs * torch.exp(n_word_weights[i].unsqueeze(1))
        scaled_text_a = abnormal_text_embs * torch.exp(a_word_weights[i].unsqueeze(1))

        concatenated_n = torch.cat((replicated_img_emb_n, scaled_text_n), dim=1)
        concatenated_a = torch.cat((replicated_img_emb_a, scaled_text_a), dim=1)

        averaged_n = torch.mean(MLP(concatenated_n), dim=0, keepdim=True)
        averaged_a = torch.mean(MLP(concatenated_a), dim=0, keepdim=True)

        h_n.append(averaged_n)
        h_a.append(averaged_a)

    h_n = torch.cat(h_n, dim=0)
    h_a = torch.cat(h_a, dim=0)

    return h_n, h_a

class AnoNAViLa_Model(nn.Module):
    def __init__(self, VLM, embedding_dim, normal_text_embs, abnormal_text_embs, device="cuda"):
        super().__init__()
        self.device = device

        # freeze the pre-trained visual-language model
        self.VLM = VLM
        self.VLM.eval()
        for p in self.VLM.parameters():
            p.requires_grad = False

        # trainable projection MLP
        self.MLP = MLP(embedding_dim=embedding_dim)

        # store fixed text embeddings as non-trainable buffers
        self.register_buffer("normal_text_embs", normal_text_embs.to(device))
        self.register_buffer("abnormal_text_embs", abnormal_text_embs.to(device))

        # centroid placeholders (will be computed later)
        self.register_buffer("normal_img_normal_text_centroid", torch.zeros(1, embedding_dim // 4))
        self.register_buffer("normal_img_abnormal_text_centroid", torch.zeros(1, embedding_dim // 4))

    def train(self, mode: bool = True):
        self.training = mode
        self.MLP.train(mode)
        self.VLM.eval()  # keep VLM always frozen
        return self

    def eval(self):
        return self.train(False)

    def forward(self, imgs):
        """
        Generate two embeddings: one for normal and one for abnormal text-augmented image embeddings.
        """
        with torch.inference_mode():
            img_embs_grad_false = self.VLM.encode_image(imgs, proj_contrast=True, normalize=True)

        # make image embeddings differentiable for MLP
        img_embs = img_embs_grad_false.clone().detach().requires_grad_(True)

        # use the internally stored text embeddings
        h_n, h_a = text_augmented_img_embs(
            img_embs,
            self.normal_text_embs,
            self.abnormal_text_embs,
            self.VLM,
            self.MLP
        )
        return h_n, h_a

    def compute_centroids(self, dataloader):
        """
        Compute two centroids: one for normal and one for abnormal text-augmented normal image embeddings.
        It is recommended to use normal images from the validation set, although the training set can also be used.
        """
        self.eval()
        val_normal_img_normal_text, val_normal_img_abnormal_text = [], []
        with torch.inference_mode():
            for imgs, _ in tqdm(dataloader, desc="Computing centroids"):
                imgs = imgs.to(self.device)

                img_embs = self.VLM.encode_image(imgs, proj_contrast=True, normalize=True)
                h_n, h_a = text_augmented_img_embs(
                    img_embs,
                    self.normal_text_embs,
                    self.abnormal_text_embs,
                    self.VLM,
                    self.MLP
                )

                val_normal_img_normal_text.append(h_n)
                val_normal_img_abnormal_text.append(h_a)

        nn_centroid = torch.cat(val_normal_img_normal_text).mean(dim=0, keepdim=True)
        na_centroid = torch.cat(val_normal_img_abnormal_text).mean(dim=0, keepdim=True)

        self.normal_img_normal_text_centroid = nn_centroid.to(self.device)
        self.normal_img_abnormal_text_centroid = na_centroid.to(self.device)

    def compute_distances(self, imgs):
        """
        Inference: compute two distances between the sample’s two embeddings and their corresponding centroids.
        """
        self.eval()
        with torch.inference_mode():
            img_embs = self.VLM.encode_image(imgs, proj_contrast=True, normalize=True)
            h_n, h_a = text_augmented_img_embs(
                img_embs,
                self.normal_text_embs,
                self.abnormal_text_embs,
                self.VLM,
                self.MLP
            )

            d_n = 1 - F.cosine_similarity(h_n, self.normal_img_normal_text_centroid, dim=-1)
            d_a = 1 - F.cosine_similarity(h_a, self.normal_img_abnormal_text_centroid, dim=-1)

        return d_n, d_a