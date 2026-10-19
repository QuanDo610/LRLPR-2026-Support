"""Post-processing utilities for OCR decoding with FP32 Grammar-Constrained Logit Calibration."""
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch

# Visual homoglyph priors for Brazilian / Mercosur license plates
DIGIT_TO_LETTER_HOMOGLYPHS: Dict[str, List[str]] = {
    '0': ['D', 'O', 'Q'],
    '1': ['I', 'L', 'J'],
    '2': ['Z'],
    '4': ['A'],
    '5': ['S'],
    '6': ['G', 'C'],
    '7': ['T', 'Z'],
    '8': ['B'],
}

LETTER_TO_DIGIT_HOMOGLYPHS: Dict[str, str] = {
    'D': '0',
    'O': '0',
    'Q': '0',
    'C': '0',
    'I': '1',
    'L': '1',
    'J': '1',
    'Z': '2',
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
}


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    use_format_prior: bool = True,
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores and optional 7-char plate grammar calibration.

    Performs class selection in FP32 log-probability space before exponentiation to prevent
    FP16 underflow when calibrating confusable Letter/Digit positions.

    Args:
        preds: Log-softmax predictions of shape [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping (1-indexed, 0 is CTC blank).
        use_format_prior: If True, calibrates 7-character predictions against the deterministic
            Brazilian/Mercosur positional grammar (Pos 0,1,2 = Letters; Pos 3,5,6 = Digits; Pos 4 = Any).

    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    # Convert to FP32 log-probabilities first to prevent FP16 exp() underflow
    log_probs_np = preds.detach().float().cpu().numpy()
    greedy_indices_np = np.argmax(log_probs_np, axis=2)
    max_log_probs_np = np.max(log_probs_np, axis=2)

    char2idx = {c: idx for idx, c in idx2char.items()}
    letter_indices = [idx for idx, c in idx2char.items() if c.isalpha()]
    digit_indices = [idx for idx, c in idx2char.items() if c.isdigit()]

    batch_size = log_probs_np.shape[0]
    results: List[Tuple[str, float]] = []

    for batch_idx in range(batch_size):
        path = greedy_indices_np[batch_idx]
        sample_log_probs = log_probs_np[batch_idx]  # [T, C]
        sample_max_lp = max_log_probs_np[batch_idx]  # [T]

        # Collect non-blank CTC segments: (char_idx, start_t, end_t, max_log_prob)
        segments: List[Tuple[int, int, int, float]] = []
        time_idx = 0
        for char_idx, group in groupby(path):
            group_size = len(list(group))
            if char_idx != 0:
                seg_lp = float(np.max(sample_max_lp[time_idx:time_idx + group_size]))
                segments.append((int(char_idx), time_idx, time_idx + group_size, seg_lp))
            time_idx += group_size

        pred_chars: List[str] = []
        confidences: List[float] = []

        apply_grammar = use_format_prior and (len(segments) == 7) and bool(letter_indices) and bool(digit_indices)

        for pos_idx, (char_idx, t_start, t_end, seg_lp) in enumerate(segments):
            raw_char = idx2char.get(char_idx, '')
            chosen_char = raw_char
            chosen_lp = seg_lp

            if apply_grammar:
                # Positions 0, 1, 2 must be Letters (A-Z)
                if pos_idx in (0, 1, 2) and not raw_char.isalpha():
                    seg_matrix = sample_log_probs[t_start:t_end, :][:, letter_indices]
                    best_local_idx = int(np.argmax( np.max(seg_matrix, axis=0) ))
                    best_letter_idx = letter_indices[best_local_idx]
                    best_letter_lp = float(np.max(sample_log_probs[t_start:t_end, best_letter_idx]))

                    # Check visual homoglyph candidates in log-prob space
                    homoglyphs = DIGIT_TO_LETTER_HOMOGLYPHS.get(raw_char, [])
                    for h_char in homoglyphs:
                        h_idx = char2idx.get(h_char)
                        if h_idx is not None:
                            h_lp = float(np.max(sample_log_probs[t_start:t_end, h_idx]))
                            # Boost homoglyph slightly (+1.5 nats) if logit was saturated on the digit
                            if h_lp + 1.5 > best_letter_lp:
                                best_letter_idx = h_idx
                                best_letter_lp = h_lp

                    chosen_char = idx2char.get(best_letter_idx, raw_char)
                    chosen_lp = best_letter_lp

                # Positions 3, 5, 6 must be Digits (0-9)
                elif pos_idx in (3, 5, 6) and not raw_char.isdigit():
                    seg_matrix = sample_log_probs[t_start:t_end, :][:, digit_indices]
                    best_local_idx = int(np.argmax( np.max(seg_matrix, axis=0) ))
                    best_digit_idx = digit_indices[best_local_idx]
                    best_digit_lp = float(np.max(sample_log_probs[t_start:t_end, best_digit_idx]))

                    h_digit = LETTER_TO_DIGIT_HOMOGLYPHS.get(raw_char)
                    if h_digit is not None:
                        h_idx = char2idx.get(h_digit)
                        if h_idx is not None:
                            h_lp = float(np.max(sample_log_probs[t_start:t_end, h_idx]))
                            if h_lp + 1.5 > best_digit_lp:
                                best_digit_idx = h_idx
                                best_digit_lp = h_lp

                    chosen_char = idx2char.get(best_digit_idx, raw_char)
                    chosen_lp = best_digit_lp

            pred_chars.append(chosen_char)
            confidences.append(float(np.clip(np.exp(chosen_lp), 0.0, 1.0)))

        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))

    return results


class CTCDecoder:
    """Convenience CTC decoder wrapper around decode_with_confidence."""
    def __init__(self, idx2char: Dict[int, str], blank_idx: int = 0):
        self.idx2char = idx2char
        self.blank_idx = blank_idx

    def decode_batch(self, preds: torch.Tensor, use_format_prior: bool = False) -> List[str]:
        results = decode_with_confidence(preds, self.idx2char, use_format_prior=use_format_prior)
        return [text for text, _ in results]
