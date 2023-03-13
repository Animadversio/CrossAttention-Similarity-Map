import numpy as np


def summary_by_block(scores_vec, gens, maxgen=100, sem=True):
    """Summarize a score trajectory and and generation vector into the mean vector, sem, """
    genmin = min(gens)
    genmax = max(gens)
    if maxgen is not None:
        genmax = min(maxgen, genmax)

    score_m = []
    score_s = []
    blockarr = []
    for geni in range(genmin, genmax + 1):
        score_block = scores_vec[gens == geni]
        if len(score_block) == 1:
            continue
        score_m.append(np.mean(score_block))
        if sem:
            score_s.append(np.std(score_block) / np.sqrt(len(score_block)))
        else:
            score_s.append(np.std(score_block))
        blockarr.append(geni)
    score_m = np.array(score_m)
    score_s = np.array(score_s)
    blockarr = np.array(blockarr)
    return score_m, score_s, blockarr
