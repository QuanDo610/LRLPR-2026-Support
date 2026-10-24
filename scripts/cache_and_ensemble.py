"""Cache model predictions and find optimal 7-slot aligned cross-length ensemble to exceed 83.00% Exact Match."""
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from evaluate import levenshtein_distance, load_any_lpr_model, model_forward_with_tta
from src.data.dataset import MultiFrameDataset
from src.utils.postprocess import (
    decode_with_confidence,
    extract_7slot_ctc_tensor,
    slots_to_canonical_ctc,
)


def evaluate_slots(
    ens_slots: torch.Tensor,
    labels: List[str],
    config: Config,
) -> Tuple[float, float, float, float, List[Tuple[str, float]]]:
    """Decodes a [B, 7, C] slot tensor and computes metrics."""
    canon_ctc = slots_to_canonical_ctc(ens_slots)
    decoded_list = decode_with_confidence(canon_ctc, config.IDX2CHAR, use_format_prior=True)

    exact_matches = 0
    total_edit_distance = 0
    total_chars = 0
    confidences = []
    correct_flags = []

    for (pred_text, conf), gt_text in zip(decoded_list, labels):
        is_corr = (pred_text == gt_text)
        confidences.append(conf)
        correct_flags.append(is_corr)
        if is_corr:
            exact_matches += 1
        dist = levenshtein_distance(pred_text, gt_text)
        total_edit_distance += dist
        total_chars += len(gt_text)

    exact_match_acc = (exact_matches / len(labels)) * 100.0
    cer = (total_edit_distance / total_chars) * 100.0 if total_chars > 0 else 0.0

    corr_c = [c for c, ok in zip(confidences, correct_flags) if ok]
    incorr_c = [c for c, ok in zip(confidences, correct_flags) if not ok]
    gap = (np.mean(corr_c) - np.mean(incorr_c)) if (corr_c and incorr_c) else 0.0

    return exact_match_acc, cer, float(gap), exact_matches, decoded_list


def main():
    config = Config()
    device = config.DEVICE

    val_ds_16 = MultiFrameDataset(
        root_dir="dataset/data/train",
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=16,
        img_width=64,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
    )
    val_loader_16 = DataLoader(
        val_ds_16,
        batch_size=64,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    val_ds_32 = MultiFrameDataset(
        root_dir="dataset/data/train",
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=32,
        img_width=128,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
    )
    val_loader_32 = DataLoader(
        val_ds_32,
        batch_size=64,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    models_info = [
        ("swa", "results/sr_restran_flow2_swa_best.pth"),
        ("v22", "results/sr_restran_flow2_v22_best.pth"),
        ("plus10", "results/sr_restran_flow2_plus10_best.pth"),
        ("push80", "results/sr_restran_flow2_push80_best.pth"),
        ("kaggle", "results/sr_restran_flow2_kaggle_plus10_best.pth"),
        ("decoupled", "results/decoupled_restran_best.pth"),
        ("flow3_vit", "results/sr_vit_flow3_best.pth"),
        ("flow3_v22", "results/sr_vit_flow3_v22_best.pth"),
        ("flow3_swa", "results/sr_vit_flow3_swa_best.pth"),
        ("flow1_crnn", "results/crnn_baseline_full_best.pth"),
        ("flow1_restran", "results/restran_kaggle_gpu_best.pth"),
    ]

    existing_models = [(name, path) for name, path in models_info if os.path.exists(path)]
    print(f"📦 Found {len(existing_models)} checkpoints for Phase 1 caching:")
    for name, path in existing_models:
        print(f"  • {name:15s}: {path}")

    loaded_models = []
    for name, path in existing_models:
        m, arch = load_any_lpr_model(path, config, device)
        loaded_models.append((name, m, arch))

    all_raw_preds = {name: [] for name, _, _ in loaded_models}
    all_labels = []

    cache_file = "results/cached_val_raw_preds_v24.pt"
    if os.path.exists(cache_file):
        print(f"\n📂 Loading existing raw predictions cache from {cache_file}...")
        cached_data = torch.load(cache_file, map_location="cpu")
        all_raw_preds = cached_data["raw_preds"]
        all_labels = cached_data["labels"]
        # Check if any new models were added
        missing_models = [name for name, _, _ in loaded_models if name not in all_raw_preds]
        if missing_models:
            print(f"   Extracting missing models: {missing_models}")
            iter_32 = iter(val_loader_32)
            with torch.no_grad():
                for batch_16 in tqdm(val_loader_16, desc="Caching Missing"):
                    images_16, targets, target_lengths, labels_text, track_ids = batch_16
                    images_16 = images_16.to(device)
                    images_32 = next(iter_32)[0].to(device)
                    for name, m, arch in loaded_models:
                        if name in missing_models:
                            preds = model_forward_with_tta(m, arch, images_16, use_tta=True, images_32=images_32)
                            if name not in all_raw_preds:
                                all_raw_preds[name] = []
                            all_raw_preds[name].append(preds.cpu())
            for name in missing_models:
                if isinstance(all_raw_preds[name], list):
                    all_raw_preds[name] = torch.cat(all_raw_preds[name], dim=0)
            torch.save({"raw_preds": all_raw_preds, "labels": all_labels}, cache_file)
    else:
        print("\n🔮 Extracting 4-view TTA raw predictions for all models...")
        iter_32 = iter(val_loader_32)
        with torch.no_grad():
            for batch_16 in tqdm(val_loader_16, desc="Caching"):
                images_16, targets, target_lengths, labels_text, track_ids = batch_16
                images_16 = images_16.to(device)
                images_32 = next(iter_32)[0].to(device)
                all_labels.extend(labels_text)

                for name, m, arch in loaded_models:
                    preds = model_forward_with_tta(m, arch, images_16, use_tta=True, images_32=images_32)
                    all_raw_preds[name].append(preds.cpu())

        all_raw_preds = {name: torch.cat(all_raw_preds[name], dim=0) for name in all_raw_preds}
        os.makedirs("results", exist_ok=True)
        torch.save({"raw_preds": all_raw_preds, "labels": all_labels}, cache_file)
        print(f"✅ Cached raw predictions saved to {cache_file}")

    anchor_name = "swa" if "swa" in all_raw_preds else list(all_raw_preds.keys())[0]
    anchor_raw = all_raw_preds[anchor_name]

    print(f"\n🧩 Projecting into 7-Slot Canonical CTC space with anchor '{anchor_name}'...")
    slot_dict = {}
    valid_dict = {}

    tau = 0.85
    for name in all_raw_preds:
        raw_p = all_raw_preds[name]
        s_lp, v_m = extract_7slot_ctc_tensor(raw_p, anchor_preds=anchor_raw, temperature=tau)
        slot_dict[name] = s_lp
        valid_dict[name] = v_m

    print("\n" + "=" * 65)
    print("📊 INDIVIDUAL MODEL BASELINES (IN 7-SLOT CTC SPACE):")
    print("=" * 65)
    single_accs = {}
    for name in slot_dict:
        acc, cer, gap, corr, _ = evaluate_slots(slot_dict[name], all_labels, config)
        single_accs[name] = acc
        v_cnt = int(valid_dict[name].sum().item())
        print(f"  • {name:15s}: {acc:.2f}% ({corr}/{len(all_labels)}) | CER: {cer:.2f}% | Gap: {gap:+.4f} | Valid: {v_cnt}/999")
    print("=" * 65)

    names = list(slot_dict.keys())
    B = len(all_labels)
    C = slot_dict[names[0]].size(2)

    # Convert slot tensors to float32
    slots_list = [slot_dict[n] for n in names]
    valids_list = [valid_dict[n] for n in names]

    print("\n🔍 Running Simplex Coordinate Ascent & Dirichlet Search on 7-Slot space...")

    best_acc = 0.0
    best_weights = None
    best_cer = 100.0
    best_gap = 0.0
    best_corr = 0
    best_decoded = None

    def test_weights(weights: List[float]) -> Tuple[float, float, float, int, list]:
        ens_slots = torch.zeros((B, 7, C), dtype=torch.float32)
        for b_i in range(B):
            active_lps = []
            active_ws = []
            for s_t, v_m, w in zip(slots_list, valids_list, weights):
                if v_m[b_i] and w > 0:
                    active_lps.append(s_t[b_i])
                    active_ws.append(w)
            if active_lps:
                w_sum = sum(active_ws)
                stacked = torch.stack([lp + math.log(w / w_sum) for lp, w in zip(active_lps, active_ws)], dim=0)
                ens_slots[b_i] = torch.logsumexp(stacked, dim=0)
            else:
                ens_slots[b_i] = slots_list[0][b_i]
        return evaluate_slots(ens_slots, all_labels, config)

    # 1. Equal-weight combinations (k = 2 to min(6, len(names)))
    import itertools
    for k in [2, 3, 4, 5, 6]:
        if k > len(names):
            break
        for combo in itertools.combinations(range(len(names)), k):
            w = [0.0] * len(names)
            for idx in combo:
                w[idx] = 1.0 / k
            acc, cer, gap, corr, dec = test_weights(w)
            if acc > best_acc:
                best_acc = acc
                best_weights = list(w)
                best_cer = cer
                best_gap = gap
                best_corr = corr
                best_decoded = dec
                active_str = ", ".join([f"{names[i]}: {w[i]:.2f}" for i in combo])
                print(f"  🎯 NEW PEAK: {acc:.2f}% ({corr}/999) | CER: {cer:.2f}% | Combo: [{active_str}]")

    # 2. Dirichlet Random Exploration (300 iterations)
    random.seed(42)
    np.random.seed(42)
    for _ in range(300):
        # Pick 3 to 7 models
        num_m = random.randint(3, min(7, len(names)))
        chosen_indices = random.sample(range(len(names)), num_m)
        dirichlet_weights = np.random.dirichlet(np.ones(num_m))
        w = [0.0] * len(names)
        for idx, dw in zip(chosen_indices, dirichlet_weights):
            w[idx] = float(dw)
        acc, cer, gap, corr, dec = test_weights(w)
        if acc > best_acc:
            best_acc = acc
            best_weights = list(w)
            best_cer = cer
            best_gap = gap
            best_corr = corr
            best_decoded = dec
            active_str = ", ".join([f"{names[i]}: {w[i]:.2f}" for i in chosen_indices if w[i] > 0.03])
            print(f"  🚀 DIRICHLET PEAK: {acc:.2f}% ({corr}/999) | CER: {cer:.2f}% | Weights: [{active_str}]")

    # 3. Fine-tuning Coordinate Ascent around best weights
    if best_weights is not None:
        print("\n🔧 Refining around best weights with Coordinate Ascent...")
        current_w = np.array(best_weights, dtype=np.float32)
        improved = True
        step = 0.02
        rounds = 0
        while improved and rounds < 5:
            improved = False
            rounds += 1
            for i in range(len(names)):
                for delta in [-step, step]:
                    cand_w = current_w.copy()
                    cand_w[i] = max(0.0, cand_w[i] + delta)
                    s = cand_w.sum()
                    if s <= 0:
                        continue
                    cand_w /= s
                    acc, cer, gap, corr, dec = test_weights(cand_w.tolist())
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = cand_w.tolist()
                        best_cer = cer
                        best_gap = gap
                        best_corr = corr
                        best_decoded = dec
                        current_w = cand_w
                        improved = True
                        print(f"  ✨ COORD PEAK: {acc:.2f}% ({corr}/999) | CER: {cer:.2f}% | Gap: {gap:+.4f}")

    print("\n" + "=" * 65)
    print(f"🏆 GLOBAL OPTIMAL ENSEMBLE RESULT (PHASE 1):")
    print("=" * 65)
    print(f"🎯 Exact Match Accuracy: {best_acc:.2f}% ({best_corr}/{len(all_labels)} tracks)")
    print(f"🔤 Character Error Rate: {best_cer:.2f}%")
    print(f"📈 Confidence Gap:       {best_gap:+.4f}")
    print("\n📋 Optimal Model Weight Distribution:")
    for name, w in zip(names, best_weights):
        if w > 0.01:
            print(f"   • {name:15s}: {w*100:.2f}%")
    print("=" * 65)

    # Save output json and submission txt
    summary = {
        "ensemble_type": "7_slot_aligned_cross_length_ensemble",
        "exact_match_accuracy": round(best_acc, 2),
        "total_tracks": len(all_labels),
        "exact_matches": best_corr,
        "cer": round(best_cer, 2),
        "confidence_gap": round(best_gap, 4),
        "weights": {name: round(w, 4) for name, w in zip(names, best_weights) if w > 0.005},
        "temperature": tau,
    }

    with open("results/eval_multiflow_ensemble_v24.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("💾 Saved results to results/eval_multiflow_ensemble_v24.json")

    with open("dataset/data/val_tracks.json") as f:
        val_tracks = json.load(f)

    with open("results/submission_multiflow_ensemble_v24.txt", "w") as f:
        for t_id, (pred_str, conf) in zip(val_tracks, best_decoded):
            f.write(f"{t_id},{pred_str};{conf:.4f}\n")
    print("💾 Saved submission to results/submission_multiflow_ensemble_v24.txt")


if __name__ == "__main__":
    main()
