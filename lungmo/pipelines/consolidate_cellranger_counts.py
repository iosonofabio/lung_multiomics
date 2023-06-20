'''Consolidate output of cellranger-arc into h5ad files'''
import pathlib
import numpy as np
import anndata
import scanpy as sc


data_fdn = '../../data/pilot_Dec2022'


# NOTE: ATM the two samples have distinct peak called so we store them in separate files. We should find a solution for that.
samples = ['nor-1', 'nor-2']
samples = ['nor-2']
for sample in samples:
    print(sample)
    print('Read from MTX files')
    adatai = sc.read_10x_mtx(
        f'{data_fdn}/cellranger_output/{sample}/outs/filtered_feature_bc_matrix',
        gex_only=False,
    )
    print('Set metadata')
    adatai.var.rename(columns={'gene_ids': 'feature_id'}, inplace=True)
    adatai.obs['sample'] = sample
    print('Write to h5ad file')
    adatai.write(
        f'{data_fdn}/counts/{sample}.h5ad',
    )
