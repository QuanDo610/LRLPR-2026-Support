"""Post-processing utilities for OCR decoding with FP32 Grammar-Constrained Logit Calibration."""
from itertools import groupby
from typing import Dict, List, Set, Tuple

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

VALID_MERCOSUR_POS4_LETTERS = frozenset("ABCDEFGHIJ")

# Empirical log P(letter | pos) for Positions 0 and 1 computed strictly on 9,001 Scenario-B TRAINING tracks (Slide 22)
POS0_LETTER_LOG_PRIOR: Dict[str, float] = {
    'A': -1.54, 'B': -1.47, 'C': -5.37, 'D': -5.12, 'E': -3.96, 'F': -3.52,
    'G': -4.09, 'H': -5.55, 'I': -4.05, 'J': -5.14, 'K': -5.61, 'L': -5.35,
    'M': -4.03, 'N': -5.42, 'O': -4.85, 'P': -3.66, 'Q': -2.69, 'R': -2.27,
    'S': -2.28, 'T': -2.23, 'U': -5.50, 'V': -9.11, 'W': -9.11, 'X': -9.11,
    'Y': -9.11, 'Z': -9.11,
}

POS1_LETTER_LOG_PRIOR: Dict[str, float] = {
    'A': -2.46, 'B': -2.30, 'C': -2.70, 'D': -2.52, 'E': -2.30, 'F': -3.24,
    'G': -4.70, 'H': -2.78, 'I': -3.87, 'J': -3.98, 'K': -4.20, 'L': -4.03,
    'M': -4.02, 'N': -4.00, 'O': -4.01, 'P': -4.02, 'Q': -3.94, 'R': -4.05,
    'S': -4.09, 'T': -3.49, 'U': -3.43, 'V': -3.44, 'W': -3.50, 'X': -3.13,
    'Y': -3.08, 'Z': -3.18,
}

# Empirical conditional log P(L1 | L0) computed strictly on 9,001 Scenario-B TRAINING tracks (DENATRAN Paraná ranges)
POS1_GIVEN_POS0_LOG_PRIOR: Dict[str, Dict[str, float]] = {
    'A': {'A': -4.98, 'B': -4.13, 'C': -4.27, 'D': -5.33, 'E': -5.23, 'F': -4.23, 'G': -4.56, 'H': -4.10, 'I': -4.23, 'J': -4.27, 'K': -4.04, 'L': -3.88, 'M': -3.96, 'N': -3.85, 'T': -2.82, 'U': -2.78, 'V': -2.55, 'W': -2.47, 'X': -2.47, 'Y': -2.12, 'Z': -2.04},
    'B': {'A': -1.85, 'B': -1.70, 'C': -1.52, 'D': -1.49, 'E': -1.64, 'F': -8.34, 'G': -7.24, 'H': -8.34, 'I': -8.34, 'J': -7.24, 'K': -6.73, 'L': -6.73, 'M': -8.34, 'N': -7.24, 'O': -7.24, 'P': -8.34, 'Q': -8.34, 'R': -6.14, 'S': -6.39, 'T': -8.34, 'U': -8.34, 'V': -8.34, 'W': -6.73, 'X': -8.34, 'Y': -5.39, 'Z': -5.63},
    'G': {'A': -2.65, 'B': -2.84, 'C': -2.03, 'D': -2.65, 'E': -2.49, 'F': -2.18, 'G': -2.84, 'H': -2.49, 'I': -2.29, 'J': -2.03, 'L': -5.79, 'M': -5.79, 'N': -5.79, 'O': -4.69, 'P': -5.79, 'Q': -5.79, 'R': -5.79, 'S': -5.79, 'T': -5.79, 'U': -5.79, 'V': -5.79, 'W': -5.79, 'X': -5.79, 'Y': -5.79, 'Z': -4.18},
    'I': {'A': -4.73, 'B': -4.73, 'C': -4.73, 'D': -5.83, 'E': -5.83, 'F': -5.83, 'G': -3.88, 'H': -5.83, 'I': -4.73, 'J': -4.73, 'K': -4.73, 'L': -4.22, 'M': -3.88, 'N': -4.22, 'O': -4.22, 'P': -3.88, 'Q': -2.78, 'T': -2.46, 'U': -2.88, 'V': -2.88, 'W': -2.07, 'X': -2.22, 'Y': -2.07, 'Z': -2.12},
    'J': {'A': -1.06, 'B': -2.30, 'C': -2.47, 'D': -3.77, 'F': -3.77, 'G': -4.87, 'L': -4.87, 'M': -4.87, 'N': -4.87, 'O': -4.87, 'P': -3.77, 'Q': -4.87, 'R': -3.77, 'S': -3.77, 'T': -4.87, 'U': -3.77, 'V': -4.87, 'W': -4.87, 'X': -4.87, 'Y': -4.87, 'Z': -3.77},
    'M': {'A': -4.24, 'B': -3.90, 'D': -3.65, 'F': -2.80, 'G': -2.55, 'I': -2.29, 'J': -2.80, 'K': -1.77, 'L': -1.70, 'M': -2.55, 'N': -5.85, 'O': -5.85, 'P': -4.75, 'Q': -4.75, 'R': -4.75, 'S': -4.24, 'T': -5.85, 'U': -5.85, 'V': -5.85, 'W': -5.85, 'X': -5.85, 'Y': -4.75, 'Z': -5.85},
    'P': {'C': -5.09, 'D': -3.79, 'E': -4.24, 'F': -4.58, 'G': -5.09, 'H': -2.48, 'I': -5.09, 'J': -3.99, 'K': -3.99, 'L': -4.58, 'M': -5.09, 'N': -3.99, 'O': -5.09, 'P': -4.24, 'Q': -3.99, 'R': -4.24, 'S': -4.58, 'T': -3.99, 'V': -2.43, 'W': -2.69, 'X': -2.34, 'Y': -2.08, 'Z': -1.64},
    'Q': {'A': -4.43, 'B': -5.19, 'C': -4.09, 'D': -5.19, 'E': -4.43, 'F': -7.13, 'G': -5.19, 'H': -2.58, 'I': -2.37, 'J': -2.58, 'K': -4.74, 'L': -4.30, 'M': -3.77, 'N': -2.52, 'O': -2.26, 'P': -2.46, 'Q': -2.19, 'R': -4.94, 'S': -5.19, 'T': -3.64, 'U': -2.65, 'V': -4.94, 'X': -2.42, 'Y': -4.74, 'Z': -4.43},
    'R': {'B': -5.15, 'C': -5.35, 'D': -3.69, 'E': -4.05, 'F': -2.87, 'G': -5.93, 'H': -0.93, 'I': -4.98, 'J': -4.71, 'K': -4.50, 'M': -2.87, 'O': -6.45, 'P': -7.54, 'Q': -7.54, 'R': -6.45, 'S': -5.93, 'T': -2.61, 'U': -2.61, 'V': -2.73, 'W': -5.60, 'X': -3.61, 'Z': -5.35},
    'S': {'A': -5.59, 'B': -5.93, 'C': -5.34, 'D': -2.04, 'E': -0.99, 'F': -1.48, 'G': -5.93, 'J': -4.04, 'K': -7.53, 'L': -5.93, 'M': -6.44, 'N': -5.93, 'O': -7.53, 'P': -6.44, 'Q': -6.44, 'R': -5.93, 'S': -3.98, 'T': -4.10, 'U': -4.83, 'V': -4.59, 'W': -5.14, 'Z': -6.44},
    'T': {'A': -1.24, 'B': -0.84, 'D': -2.84, 'E': -2.70, 'F': -7.59, 'G': -6.49, 'H': -7.59, 'I': -5.39, 'J': -4.64, 'K': -4.45, 'L': -4.54, 'M': -5.39, 'N': -7.59, 'O': -7.59, 'P': -4.45, 'Q': -5.98, 'R': -6.49, 'S': -7.59, 'T': -6.49, 'U': -7.59, 'V': -7.59, 'W': -6.49, 'Y': -4.76, 'Z': -4.88},
}


def extract_7slot_ctc_tensor(
    preds: torch.Tensor,
    anchor_preds: torch.Tensor = None,
    temperature: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projects a [B, T, C] CTC log-probability tensor (for any sequence length T, e.g. T=16 or T=33)
    into a phase-invariant, non-blank normalized 7-slot tensor [B, 7, C] and a validity mask [B].
    """
    lp_np = (preds.detach().float().cpu().numpy() / max(1e-3, temperature)).astype(np.float32)
    g_np = np.argmax(lp_np, axis=2)
    B, T_n, C = lp_np.shape

    anchor_segs_list = None
    T_anc = T_n
    if anchor_preds is not None:
        anc_np = anchor_preds.detach().float().cpu().numpy()
        anc_g = np.argmax(anc_np, axis=2)
        T_anc = anc_np.shape[1]
        anchor_segs_list = []
        for b in range(B):
            s_b = []
            t_i = 0
            for c_idx, grp in groupby(anc_g[b]):
                gsz = len(list(grp))
                if c_idx != 0:
                    s_b.append((t_i, t_i + gsz))
                t_i += gsz
            anchor_segs_list.append(s_b)

    out = np.zeros((B, 7, C), dtype=np.float32)
    valid = np.zeros(B, dtype=bool)

    for b in range(B):
        segs = []
        t_i = 0
        for c_idx, grp in groupby(g_np[b]):
            gsz = len(list(grp))
            if c_idx != 0:
                segs.append((t_i, t_i + gsz))
            t_i += gsz

        if len(segs) == 7:
            valid[b] = True
            for k, (ts, te) in enumerate(segs):
                out[b, k, :] = np.max(lp_np[b, ts:te, :], axis=0)
        elif anchor_segs_list is not None and len(anchor_segs_list[b]) == 7:
            valid[b] = True
            for k, (ts, te) in enumerate(anchor_segs_list[b]):
                ts_n = max(0, int(round(ts * T_n / float(T_anc))))
                te_n = min(T_n, max(ts_n + 1, int(round(te * T_n / float(T_anc)))))
                out[b, k, :] = np.max(lp_np[b, ts_n:te_n, :], axis=0)

    out_t = torch.from_numpy(out)
    char_lp = torch.nn.functional.log_softmax(out_t[:, :, 1:], dim=-1)
    full_lp = torch.full((B, 7, C), -30.0, dtype=torch.float32)
    full_lp[:, :, 1:] = char_lp
    return full_lp, torch.from_numpy(valid)


def slots_to_canonical_ctc(slot_lp: torch.Tensor) -> torch.Tensor:
    """Interleaves a [B, 7, C] slot tensor with CTC blanks to form a canonical [B, 13, C] CTC tensor."""
    B, _, C = slot_lp.shape
    blank_step = torch.full((B, 1, C), -30.0, dtype=slot_lp.dtype, device=slot_lp.device)
    blank_step[:, 0, 0] = 0.0
    steps = []
    for k in range(7):
        steps.append(slot_lp[:, k : k + 1, :])
        if k < 6:
            steps.append(blank_step)
    return torch.cat(steps, dim=1)


def _beam_search_7char_fallback(
    sample_log_probs: np.ndarray,
    idx2char: Dict[int, str],
    letter_indices_set: set,
    digit_indices_set: set,
    beam_width: int = 5,
) -> Tuple[str, float]:
    """Prefix Beam Search fallback to recover valid 7-character plates when greedy length != 7."""
    T, _ = sample_log_probs.shape
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -np.inf)}

    for t in range(T):
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        top_c = np.argsort(sample_log_probs[t])[-5:][::-1]
        lp_blank = float(sample_log_probs[t, 0])

        for prefix, (p_b, p_nb) in beams.items():
            p_total = float(np.logaddexp(p_b, p_nb))
            nb_b, nb_nb = next_beams.get(prefix, (-np.inf, -np.inf))
            next_beams[prefix] = (float(np.logaddexp(nb_b, p_total + lp_blank)), nb_nb)

            for c in top_c:
                c_int = int(c)
                if c_int == 0:
                    continue
                lp_c = float(sample_log_probs[t, c_int])
                pos = len(prefix)
                grammar_bonus = 0.0
                if pos in (0, 1, 2) and c_int not in letter_indices_set:
                    grammar_bonus = -1.5
                elif pos in (3, 5, 6) and c_int not in digit_indices_set:
                    grammar_bonus = -1.5
                elif pos == 4 and c_int in letter_indices_set and idx2char.get(c_int) not in VALID_MERCOSUR_POS4_LETTERS:
                    grammar_bonus = -2.0

                if pos > 0 and prefix[-1] == c_int:
                    ob_b, ob_nb = next_beams.get(prefix, (-np.inf, -np.inf))
                    next_beams[prefix] = (ob_b, float(np.logaddexp(ob_nb, p_nb + lp_c)))
                    if pos < 7:
                        new_p = prefix + (c_int,)
                        nb2_b, nb2_nb = next_beams.get(new_p, (-np.inf, -np.inf))
                        next_beams[new_p] = (nb2_b, float(np.logaddexp(nb2_nb, p_b + lp_c + grammar_bonus)))
                elif pos < 7:
                    new_p = prefix + (c_int,)
                    nb2_b, nb2_nb = next_beams.get(new_p, (-np.inf, -np.inf))
                    next_beams[new_p] = (nb2_b, float(np.logaddexp(nb2_nb, p_total + lp_c + grammar_bonus)))

        sorted_beams = sorted(
            next_beams.items(),
            key=lambda kv: float(np.logaddexp(kv[1][0], kv[1][1])),
            reverse=True,
        )[:beam_width * 2]
        beams = dict(sorted_beams)

    best_7 = None
    best_7_score = -np.inf
    for prefix, (p_b, p_nb) in beams.items():
        score = float(np.logaddexp(p_b, p_nb))
        if len(prefix) == 7 and score > best_7_score:
            best_7 = prefix
            best_7_score = score

    if best_7 is None:
        return "", 0.0

    chars: List[str] = []
    for pos_idx, c_idx in enumerate(best_7):
        ch = idx2char.get(c_idx, "")
        if pos_idx in (0, 1, 2) and not ch.isalpha():
            h_list = DIGIT_TO_LETTER_HOMOGLYPHS.get(ch, [])
            if h_list:
                ch = h_list[0]
        elif pos_idx in (3, 5, 6) and not ch.isdigit():
            h_d = LETTER_TO_DIGIT_HOMOGLYPHS.get(ch)
            if h_d:
                ch = h_d
        elif pos_idx == 4 and ch.isalpha() and ch not in VALID_MERCOSUR_POS4_LETTERS:
            h_d = LETTER_TO_DIGIT_HOMOGLYPHS.get(ch)
            if h_d:
                ch = h_d
        chars.append(ch)

    return "".join(chars), float(np.clip(np.exp(best_7_score / max(1, T)), 0.0, 1.0))


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

        if use_format_prior and len(segments) != 7 and bool(letter_indices) and bool(digit_indices):
            fb_str, fb_conf = _beam_search_7char_fallback(
                sample_log_probs=sample_log_probs,
                idx2char=idx2char,
                letter_indices_set=set(letter_indices),
                digit_indices_set=set(digit_indices),
                beam_width=5,
            )
            if len(fb_str) == 7:
                results.append((fb_str, fb_conf))
                continue

        for pos_idx, (char_idx, t_start, t_end, seg_lp) in enumerate(segments):
            raw_char = idx2char.get(char_idx, '')
            chosen_char = raw_char
            chosen_lp = seg_lp

            if apply_grammar:
                # Positions 0, 1, 2 must be Letters (A-Z) with Slide-22 Bayesian & DENATRAN Bigram log-priors
                if pos_idx in (0, 1, 2):
                    prior_map = POS0_LETTER_LOG_PRIOR if pos_idx == 0 else (POS1_LETTER_LOG_PRIOR if pos_idx == 1 else {})
                    alpha_p = 0.25 if pos_idx == 0 else (0.12 if pos_idx == 1 else 0.0)
                    bg_map = POS1_GIVEN_POS0_LOG_PRIOR.get(pred_chars[0], {}) if (pos_idx == 1 and pred_chars) else {}
                    beta_bg = 0.15 if (pos_idx == 1 and bg_map) else 0.0
                    seg_max_lp = np.max(sample_log_probs[t_start:t_end, :], axis=0)
                    best_letter_idx = char_idx if raw_char.isalpha() else letter_indices[0]
                    best_score = -1e9
                    for l_idx in letter_indices:
                        l_char = idx2char[l_idx]
                        score = (
                            float(seg_max_lp[l_idx])
                            + alpha_p * prior_map.get(l_char, -3.25)
                            + beta_bg * bg_map.get(l_char, -3.26)
                        )
                        if score > best_score:
                            best_score = score
                            best_letter_idx = l_idx
                    best_letter_lp = float(seg_max_lp[best_letter_idx])

                    if not raw_char.isalpha():
                        homoglyphs = DIGIT_TO_LETTER_HOMOGLYPHS.get(raw_char, [])
                        for h_char in homoglyphs:
                            h_idx = char2idx.get(h_char)
                            if h_idx is not None:
                                h_lp = float(seg_max_lp[h_idx])
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

                # Position 4: Mercosur hard constraint (Digits 0-9 or Letters A-J only, K-Z strictly forbidden)
                elif pos_idx == 4 and raw_char.isalpha() and raw_char not in VALID_MERCOSUR_POS4_LETTERS:
                    seg_max_lp = np.max(sample_log_probs[t_start:t_end, :], axis=0)
                    valid_pos4_indices = digit_indices + [
                        idx for idx in letter_indices if idx2char[idx] in VALID_MERCOSUR_POS4_LETTERS
                    ]
                    best_p4_idx = max(valid_pos4_indices, key=lambda idx: float(seg_max_lp[idx]))
                    best_p4_lp = float(seg_max_lp[best_p4_idx])

                    h_digit = LETTER_TO_DIGIT_HOMOGLYPHS.get(raw_char)
                    if h_digit is not None:
                        h_idx = char2idx.get(h_digit)
                        if h_idx is not None:
                            h_lp = float(seg_max_lp[h_idx])
                            if h_lp + 1.5 > best_p4_lp:
                                best_p4_idx = h_idx
                                best_p4_lp = h_lp

                    chosen_char = idx2char.get(best_p4_idx, raw_char)
                    chosen_lp = best_p4_lp

            pred_chars.append(chosen_char)
            confidences.append(float(np.clip(np.exp(chosen_lp), 0.0, 1.0)))

        pred_str = "".join(pred_chars)
        if confidences:
            confidence = float(0.5 * np.mean(confidences) + 0.5 * np.min(confidences))
        else:
            confidence = 0.0
        results.append((pred_str, confidence))

    return results


class CTCDecoder:
    """Convenience CTC decoder wrapper around decode_with_confidence."""
    def __init__(self, idx2char: Dict[int, str], blank_idx: int = 0):
        self.idx2char = idx2char
        self.blank_idx = blank_idx

    def decode_batch(self, preds: torch.Tensor, use_format_prior: bool = True) -> List[str]:
        results = decode_with_confidence(preds, self.idx2char, use_format_prior=use_format_prior)
        return [text for text, _ in results]
