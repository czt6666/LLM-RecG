import json
import os
import random

import torch
from torch.utils.data import DataLoader

from domain_alignment_rec import initialize_dataset, load_pretrained_embeddings_from_dataset, initialize_model
from models.model_trainer import evaluate_model_with_neg_sampling


class Args:
    model_name = "gru4rec"
    hidden_units = 256
    num_layers = 2
    dropout_rate = 0.5
    max_len = 50
    num_heads = 2


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    dataset_name = "amazon_musical_instruments"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    dataset = initialize_dataset(dataset_name, Args.max_len)
    embeddings = load_pretrained_embeddings_from_dataset(dataset_name)
    model = initialize_model(Args, dataset.get_num_items(), embeddings, device)
    ckpt = "saved_ckpts/gru4rec_amazon_musical_instruments_irm.pth"
    print("loading", ckpt, os.path.getsize(ckpt), flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    recall_sum, ndcg_sum, total = evaluate_model_with_neg_sampling(
        model, loader, [5, 10, 20], dataset.get_num_items(), device
    )
    result = {}
    for k in [5, 10, 20]:
        result[f"Recall@{k}"] = recall_sum[k] / total * 100
        result[f"NDCG@{k}"] = ndcg_sum[k] / total * 100
    os.makedirs("results", exist_ok=True)
    with open("results/gru4rec_amazon_musical_instruments_smoke_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
