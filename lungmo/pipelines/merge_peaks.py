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


def annotate_genes(genes):
    import gzip
    gtf_file = '../../data/gene_annotations/mm10.ncbiRefSeq.gtf.gz'
    with gzip.open(gtf_file, 'rt') as gtf:
        transcript_annos = []
        for line in gtf:
            if '\ttranscript\t' not in line:
                continue
            fields = line.split('\t')
            if fields[2] != 'transcript':
                continue
            attrs = fields[-1].split(';')
            gene_name = None
            transcript_id = None
            for attr in attrs:
                if 'gene_name' in attr:
                    gene_name = attr.split(' ')[-1][1:-1]
                elif 'transcript_id' in attr:
                    transcript_id = attr.split(' ')[-1][1:-1]
            if (gene_name is None) or (transcript_id is None):
                continue
            transcript_annos.append({
                'transcript_id': transcript_id,
                'gene_name': gene_name,
                'chromosome_name': fields[0],
                'start_position': int(fields[3]),
                'end_position': int(fields[4]),
                'strand': 1 if fields[6] == '+' else -1,
                'transcription_start_site': int(fields[3]) if fields[6] == '+' else int(fields[4]),
                })
    transcript_annos = pd.DataFrame(transcript_annos)

    gene_annos = transcript_annos[['gene_name', 'chromosome_name', 'strand']].groupby('gene_name').first()
    gene_annos['start_position'] = transcript_annos[['gene_name', 'start_position']].groupby('gene_name').min()['start_position']
    gene_annos['end_position'] = transcript_annos[['gene_name', 'end_position']].groupby('gene_name').min()['end_position']

    # Align index
    gene_annos = gene_annos.reindex(genes)
    gene_annos['chromosome_name'] = gene_annos['chromosome_name'].fillna('')
    gene_annos['strand'] = gene_annos['strand'].fillna(0).astype('i2')
    gene_annos['start_position'] = gene_annos['start_position'].fillna(-1).astype('i8')
    gene_annos['end_position'] = gene_annos['end_position'].fillna(-1).astype('i8')

    return gene_annos


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

    
    peaks_shared = list(peaks_merged['name'].values)
    adata_to_merge = []
    for i, adata in enumerate(adatas, 1):
        # I checked, the genes are the same (come from the GTF)
        genes = list(adata.var_names[adata.var['feature_types'] == 'Gene Expression'])
        peaks_i = list(peaks_merged[f'name{i}'].values)
        features_i = genes + peaks_i
        features_new = genes + peaks_shared
        adata = adata[:, features_i]
        # Rename peaks
        adata.var.index = features_new
        # Add to merge
        adata_to_merge.append(adata)
    # Perform the merge
    adata_merged = anndata.concat(
            adata_to_merge,
            index_unique=':',
            keys=['nor-1', 'nor-2'],
    )
    adata_merged.var['feature_id'] = adata_merged.var_names
    adata_merged.var.loc[genes, 'feature_id'] = adata.var.loc[genes, 'feature_id']
    adata_merged.var['feature_types'] = 'Peaks'
    adata_merged.var.loc[genes, 'feature_types'] = 'Gene Expression'

    # Annotate gene coordinates since we are here
    gene_annos = annotate_genes(genes)
    peak_annos = peaks_merged['name'].str.split(':', expand=True).rename(columns={0: 'chromosome'})
    tmp = peak_annos[1].str.split('-', expand=True).astype(int)
    del peak_annos[1]
    peak_annos['start'] = tmp[0]
    peak_annos['end'] = tmp[1]
    del tmp
    adata_merged.var['strand'] = 0
    adata_merged.var.loc[genes, 'strand'] = gene_annos['strand']
    adata_merged.var['chromosome'] = ''
    adata_merged.var.loc[genes, 'chromosome'] = gene_annos['chromosome_name']
    adata_merged.var.loc[peaks_shared, 'chromosome'] = peak_annos['chromosome'].values
    adata_merged.var['start'] = 0
    adata_merged.var.loc[genes, 'start'] = gene_annos['start_position']
    adata_merged.var.loc[peaks_shared, 'start'] = peak_annos['start'].values
    adata_merged.var['end'] = 0
    adata_merged.var.loc[genes, 'end'] = gene_annos['end_position']
    adata_merged.var.loc[peaks_shared, 'end'] = peak_annos['end'].values

    adata_merged.write(data_fdn / 'merged_nor-1_nor-2.h5ad')
