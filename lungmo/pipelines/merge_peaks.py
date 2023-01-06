# vim: fdm=indent
'''
author:     Fabio Zanini
date:       06/01/23
content:    Merge ATAC peaks between two samples.
'''
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import anndata


if __name__ == '__main__':

    overlap_threshold = 0.4

    data_fdn = pathlib.Path('../../data/pilot_Dec2022/counts')
    adatas = [
        anndata.read_h5ad(data_fdn / 'nor-1.h5ad'),
        anndata.read_h5ad(data_fdn / 'nor-2.h5ad'),
    ]

    peak_dfs = []
    for i, adata in enumerate(adatas, 1):
        df = adata.var_names[adata.var['feature_types'] == 'Peaks'].str.split('[:-]', expand=True).to_frame()
        df.columns = ['chromosome', 'start', 'end']
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
        df['sample'] = i
        peak_dfs.append(df)
    peak_df = pd.concat(peak_dfs).reset_index(drop=True)

    # NOTE: the peaks are already ordered
    peaks_merged = []
    for chrom, peaks_chrom in peak_df.groupby('chromosome'):
        print(chrom)
        peaks_chrom_sample = [x for _, x in peaks_chrom.groupby('sample')]
        # If a sample is missing a chromosome altogether, skip, the overlap would be "zero"
        # NOTE: This is only valid because the two samples are biological replicates. Otherwise,
        # we should just take it
        if len(peaks_chrom_sample) < 2:
            continue

        ns = [len(x) for x in peaks_chrom_sample]
        print(ns)

        # First, check if there is any overlap at all
        tmp = np.zeros(tuple(ns), int)

        # End of the p1 cannot be before start of p2
        tmp.T[:] = peaks_chrom_sample[0]['end'].values
        tmp -= peaks_chrom_sample[1]['start'].values
        overlap = tmp >= 0
        # Start of the p1 cannot be after end of p2
        tmp.T[:] = peaks_chrom_sample[0]['start'].values
        tmp -= peaks_chrom_sample[1]['end'].values
        overlap &= tmp <= 0

        idxs = overlap.nonzero()

        # Then, refine by computing the amount of overlap
        # Because overlapping is a sparse operation, this should be fast
        for i1, i2 in zip(*idxs):
            p1 = peaks_chrom_sample[0].iloc[i1]
            p2 = peaks_chrom_sample[1].iloc[i2]
            s1, e1 = p1['start'], p1['end']
            s2, e2 = p2['start'], p2['end']
            smin = min([s1, s2])
            smax = max([s1, s2])
            emin = min([e1, e2])
            emax = max([e1, e2])
            ov = (emin + 1 - smax) / (emax + 1 - smin)

            if ov >= overlap_threshold:
                peaks_merged.append(
                    # Chromosome, start p1, end p1, start p2, end p2, start merge, end merge
                    [chrom, s1, e1, s2, e2, smin, emax, ov,
                     f'{chrom}:{s1}-{e1}', f'{chrom}:{s2}-{e2}', f'{chrom}:{smin}-{emax}'],
                )
            overlap[i1, i2] = ov

    peaks_merged = pd.DataFrame(
        peaks_merged,
        columns=['chromosome', 'start1', 'end1', 'start2', 'end2', 'start',
                 'end', 'overlap_fraction', 'name1', 'name2', 'name'],
    )

    
    adata_to_merge = []
    for i, adata in enumerate(adatas, 1):
        # I checked, the genes are the same (come from the GTF)
        genes = list(adata.var_names[adata.var['feature_types'] == 'Gene Expression'])
        peaks_shared = list(peaks_merged[f'name{i}'].values)
        features = genes + peaks_shared
        adata_to_merge.append(adata[:, features])

    adata_merged = anndata.concat(
            adata_to_merge,
            index_unique=':',
            keys=['nor-1', 'nor-2'],
    )
    adata_merged.write(data_fdn / 'merged_nor-1_nor-2.h5ad')
