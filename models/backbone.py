import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

from data.dataloader import word_Dataset
from utils import save_pkl

def get_text_features(device, model, tokenizer, word_dataloader):
    model.eval()
    all_text_features = []
    with torch.inference_mode():
        for batch in tqdm(word_dataloader, leave=False, desc='Getting text features', total=len(word_dataloader)):
            tokenized_prompts = tokenize(texts=batch, tokenizer=tokenizer).to(device)
            text_embedings = model.encode_text(tokenized_prompts, normalize=True)
            all_text_features.append(text_embedings)
        all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features

def set_CONCH(args):
    # set VLM and image preprocess
    CONCH_model, CONCH_preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=args["hf_auth_token"])
    CONCH_model.to(args["device"])

    args["VLM"] = CONCH_model
    args["VLM_img_preprocess"] = CONCH_preprocess

    # extract text features
    normal_words_path = args["normal_words_path"]
    abnormal_words_path = args["abnormal_words_path"]
    prefix = args["prefix"]

    normal_base = os.path.basename(normal_words_path)
    normal_name, _ = os.path.splitext(normal_base)
    n_words_f_path = os.path.join(
        os.path.dirname(normal_words_path),
        f"{prefix}_{normal_name}_f.pkl"
    )
    abnormal_base = os.path.basename(abnormal_words_path)
    abnormal_name, _ = os.path.splitext(abnormal_base)
    a_words_f_path = os.path.join(
        os.path.dirname(abnormal_words_path),
        f"{prefix}_{abnormal_name}_f.pkl"
    )

    normal_word_dataset = word_Dataset(normal_words_path, header='Normal')
    abnormal_word_dataset = word_Dataset(abnormal_words_path, header='Abnormal')
    normal_word_loader = DataLoader(normal_word_dataset, batch_size=10, shuffle=False)
    abnormal_word_loader = DataLoader(abnormal_word_dataset, batch_size=10, shuffle=False)

    CONCH_tokenizer = get_tokenizer()
    n_words_f = get_text_features(args["device"], CONCH_model, CONCH_tokenizer, normal_word_loader)
    a_words_f = get_text_features(args["device"], CONCH_model, CONCH_tokenizer, abnormal_word_loader)

    save_pkl(n_words_f, n_words_f_path)
    save_pkl(a_words_f, a_words_f_path)

    args["normal_text_embs"] = n_words_f.clone().detach().requires_grad_(True)
    args["abnormal_text_embs"] = a_words_f.clone().detach().requires_grad_(True)


