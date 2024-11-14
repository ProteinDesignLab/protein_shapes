# evaluates foldseek embeddings stored in specified json against pre-computed CATH ground truth embeddings by calling helper functions in foldseek_utils.py; stores token counts, token frequencies, chi square stat, forward kl divergence, and reverse kl divergence for both unigram and bigram

import json
import pandas as pd
from pathlib import Path
import os
import numpy as np
from scipy.stats import chi2_contingency


def bin_by_length(*args):
    strings = args[0]
    boundaries = args[1:] if len(args) > 1 else []

    bins = [[] for _ in range(len(boundaries))]

    def find_bin_index(s):
        length = len(s)
        for i, boundary in enumerate(boundaries):
            if length <= boundary:
                return i
        return len(boundaries)

    for string in strings:
        bin_index = find_bin_index(string)
        if bin_index == len(boundaries):
            bin_index = len(boundaries) - 1  # * group long sequences into the last bin
            # raise ValueError(f"String '{string}' is too long for the defined bins. Length: {len(string)}")
        bins[bin_index].append(string)
    return bins


def count_unigrams(*args):
    bins = args[0]
    boundaries = args[1:] if len(args) > 1 else []

    allowed_chars = set("ACDEFGHIKLMNPQRSTVWY")

    bin_char_counts = {}
    bin_char_freqs = {}

    for i, bin in enumerate(bins):
        char_counts = {char: 0 for char in allowed_chars}
        total_count = 0

        for string in bin:
            for char in string:
                if char in allowed_chars:
                    char_counts[char] += 1
                    total_count += 1

        if i == 0:
            label = f"1-{boundaries[i]}"
        else:
            label = f"{boundaries[i-1]+1}-{boundaries[i]}"

        bin_char_counts[f"{label}"] = char_counts

        if total_count > 0:
            bin_char_freqs[f"{label}"] = {
                char: (count / total_count * 100) for char, count in char_counts.items()
            }
        else:
            bin_char_freqs[f"{label}"] = {char: 0 for char in allowed_chars}

    return bin_char_counts, bin_char_freqs


def count_bigrams(*args):
    bins = args[0]
    boundaries = args[1:] if len(args) > 1 else []

    allowed_chars = "ACDEFGHIKLMNPQRSTVWY"

    bin_bigram_counts = {}
    bin_bigram_freqs = {}

    for i, bin in enumerate(bins):
        bigram_count = {f"{x}{y}": 0 for x in allowed_chars for y in allowed_chars}
        total_count = 0

        for string in bin:
            for j in range(len(string) - 1):
                bigram = string[j : j + 2]
                if bigram[0] in allowed_chars and bigram[1] in allowed_chars:
                    bigram_count[bigram] += 1
                    total_count += 1

        if i == 0:
            label = f"1-{boundaries[i]}"
        else:
            label = f"{boundaries[i-1]+1}-{boundaries[i]}"

        bin_bigram_counts[f"{label}"] = bigram_count

        if total_count > 0:
            bin_bigram_freqs[f"{label}"] = {
                bigram: (count / total_count * 100)
                for bigram, count in bigram_count.items()
            }
        else:
            bin_bigram_freqs[f"{label}"] = {bigram: 0 for bigram in bigram_count.keys()}

    return bin_bigram_counts, bin_bigram_freqs


def chi_square(dict1, dict2):
    chi_square_results = {}

    for key in dict1.keys():
        count1 = [x + 1e-10 for x in dict1[key].values()]
        count2 = [x + 1e-10 for x in dict2[key].values()]

        contingency_table = [count1, count2]

        chi2, p, dof, expected = chi2_contingency(contingency_table)

        chi_square_results[key] = (chi2, p, dof)

    return chi_square_results


def kl_divergence(dict1, dict2):
    kl_divergence_results = {}

    for key in dict1.keys():
        sorted_keys = sorted(dict1[key].keys())
        p = np.array([dict1[key][x] + 1e-10 for x in sorted_keys])
        q = np.array([dict2[key][x] + 1e-10 for x in sorted_keys])

        kl_div = np.sum(p * np.log(p / q))

        kl_divergence_results[key] = kl_div

    return kl_divergence_results


def get_distribution_stats(
    struc_seqs_file1: Path, struc_seqs_file2: Path, output_dir: Path
) -> None:
    with open(struc_seqs_file1, "r") as file:
        struc_seqs1 = json.load(file)

    with open(struc_seqs_file2, "r") as file:
        struc_seqs2 = json.load(file)

    boundaries = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    bins1 = bin_by_length(struc_seqs1, *boundaries)
    unigram_count1, unigram_freq1 = count_unigrams(bins1, *boundaries)
    bigram_count1, bigram_freq1 = count_bigrams(bins1, *boundaries)

    bins2 = bin_by_length(struc_seqs2, *boundaries)
    unigram_count2, unigram_freq2 = count_unigrams(bins2, *boundaries)
    bigram_count2, bigram_freq2 = count_bigrams(bins2, *boundaries)

    chi_square_unigram = chi_square(unigram_count1, unigram_count2)
    chi_square_bigram = chi_square(bigram_count1, bigram_count2)

    kl_divergence_unigram = kl_divergence(unigram_freq1, unigram_freq2)
    kl_divergence_bigram = kl_divergence(bigram_freq1, bigram_freq2)

    kl_divergence_unigram_reverse = kl_divergence(unigram_freq2, unigram_freq1)
    kl_divergence_bigram_reverse = kl_divergence(bigram_freq2, bigram_freq1)

    counts1 = {
        "unigram_count": unigram_count1,
        "unigram_freq": unigram_freq1,
        "bigram_count": bigram_count1,
        "bigram_freq": bigram_freq1,
    }
    df1 = pd.DataFrame(counts1).T
    df1.index.name = "length"  # Set index name
    struc_seqs_file1_stem = Path(struc_seqs_file1).stem
    df1.to_csv(
        os.path.join(output_dir, f"token_counts_{struc_seqs_file1_stem}.csv"),
        header=True,
    )
    print(
        f"Stored token counts in {os.path.join(output_dir, f'token_counts_{struc_seqs_file1_stem}.csv')}"
    )

    counts2 = {
        "unigram_count": unigram_count2,
        "unigram_freq": unigram_freq2,
        "bigram_count": bigram_count2,
        "bigram_freq": bigram_freq2,
    }
    df2 = pd.DataFrame(counts2).T
    df2.index.name = "length"  # Set index name
    struc_seqs_file2_stem = Path(struc_seqs_file2).stem
    df2.to_csv(
        os.path.join(output_dir, f"token_counts_{struc_seqs_file2_stem}.csv"),
        header=True,
    )
    print(
        f"Stored token counts in {os.path.join(output_dir, f'token_counts_{struc_seqs_file2_stem}.csv')}"
    )

    results = {
        "chi_square_unigram": chi_square_unigram,
        "chi_square_bigram": chi_square_bigram,
        "kl_divergence_unigram": kl_divergence_unigram,
        "kl_divergence_bigram": kl_divergence_bigram,
        "kl_divergence_unigram_reverse": kl_divergence_unigram_reverse,
        "kl_divergence_bigram_reverse": kl_divergence_bigram_reverse,
    }
    results_df = pd.DataFrame(results).T
    results_df.index.name = "length"  # Set index name

    if output_dir == "embeddings/foldseek":
        current_dir = Path(__file__).parent
        output_dir = current_dir.parent
    results_df.to_csv(
        os.path.join(
            output_dir,
            f"distribution_stats_{struc_seqs_file2_stem}_against_{struc_seqs_file1_stem}.csv",
        ),
        header=True,
        float_format='%.4f',
    )
    print(
        f"Stored distribution stats in {os.path.join(output_dir, f'distribution_stats_{struc_seqs_file2_stem}_against_{struc_seqs_file1_stem}.csv')}"
    )
