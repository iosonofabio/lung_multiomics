# vim: fdm=indent
'''
author:     Fabio Zanini
date:       09/01/23
content:    Explore first two samples, P7 normals.
'''
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import anndata
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    print('Read h5ad file')
    data_fdn = pathlib.Path('../../data/pilot_Dec2022/counts')
    adata = anndata.read_h5ad(data_fdn / 'merged_nor-1_nor-2.h5ad')

    adata.obs['coverage_ge'] = adata[:, adata.var['feature_types'] == 'Gene Expression'].X.sum(axis=1)
    adata.obs['coverage_pe'] = adata[:, adata.var['feature_types'] == 'Peaks'].X.sum(axis=1)

    adatag = adata[adata.obs['coverage_ge'] >= 500]

    if False:
        print('Check heatmap of marker genes')
        marker_exp = pd.DataFrame([], index=adata.obs_names)
        genes = ['Cdh5', 'Pecam1',
                 'Ptprc',
                 'Epcam', 'Krt19',
                 'Col1a1', 'Col1a2', 'Col6a2',
                 'Snap25',
                 'Gja5', 'Bmx', 'Vwf', 'Car4', 'Peg3', 'Car8',
                 'Fn1', 'Ctsh', 'Kcne3', 'Cdh13', 'Thy1', 'Ccl21a',
                 'Sirpa', 'Fibin',
                 'Cd3d', 'Cd3e', 'Cd19', 'Ms4a1', 'Cd68', 'Plac8', 'Dab2', 'C1qa',
                 'Mcpt4', 'Mcpt8', 'Itgax', 'Retnlg', 'Gzma', 'Areg',
                 'Acta2', 'Pdgfrb', 'Hhip', 'Pdgfra', 'Adh1',
                 ]
        adatam = adatag[:, genes]

        X = adatam.X.todense()
        X = X / (X.sum(axis=1) + 0.1)

        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, leaves_list

        pdis = pdist(X)
        Z = linkage(pdis, method='average')
        idx_cells = leaves_list(Z)

        pdis = pdist(X.T, metric='correlation')
        Z = linkage(pdis, method='average')
        idx_genes = leaves_list(Z)
        gene_names = [genes[i] for i in idx_genes]

        Xcl = X[idx_cells].T[idx_genes].T

        fig, ax = plt.subplots()
        sns.heatmap(np.log(Xcl + 0.5), yticklabels=False, ax=ax, xticklabels=True)
        ax.set_xticklabels(gene_names, rotation=90)
        fig.tight_layout()

    if True:
        print('Some QC')
        adata_ge = adata[:, adata.var['feature_types'] == 'Gene Expression']
        adata_pe = adata[:, adata.var['feature_types'] == 'Peaks']
        sc.pp.normalize_total(adata_ge, target_sum=1e4)
        sc.pp.normalize_total(adata_pe, target_sum=1e4)
        adata_nor = anndata.concat([adata_ge, adata_pe], axis=1, merge='first')

        fig, ax = plt.subplots()
        colors = {'nor-1': 'tomato', 'nor-2': 'steelblue'}
        for sample, group in adata_nor.obs.groupby('sample'):
            ax.scatter(
                0.1 + group['coverage_ge'],
                0.1 + group['coverage_pe'],
                color=colors[sample],
                alpha=0.2,
                label=sample,
            )
        ax.set_xlabel('Coverage GEX [UMI]')
        ax.set_ylabel('Coverage ATAC [UMI]')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

    if True:
        print('Try standard clustering, with features from both')
        adatag = adata_nor[adata_nor.obs['coverage_ge'] >= 500]
        adata_ge = adatag[:, adatag.var['feature_types'] == 'Gene Expression']
        
        # Log
        sc.pp.log1p(adata_ge)

        # HVG
        sc.pp.highly_variable_genes(adata_ge, min_mean=0.0125, max_mean=3, min_disp=0.5)

        adata_ge.raw = adata_ge
        adata_ge = adata_ge[:, adata_ge.var.highly_variable]

        # scale
        sc.pp.scale(adata_ge, max_value=10)

        # PCA
        sc.tl.pca(adata_ge, svd_solver='arpack')

        # neighborhood graph
        sc.pp.neighbors(adata_ge, n_neighbors=10, n_pcs=40)

        # Embed
        sc.tl.umap(adata_ge)

        # Cluster
        sc.tl.leiden(adata_ge)

        # Compute avg expression of markers
        genes = ['Cdh5', 'Pecam1',
                 'Ptprc',
                 'Epcam', 'Krt19',
                 'Col1a1', 'Col1a2', 'Col6a2',
                 'Snap25',
                 'Gja5', 'Bmx', 'Vwf', 'Car4', 'Peg3', 'Car8',
                 'Fn1', 'Ctsh', 'Kcne3', 'Cdh13', 'Thy1', 'Ccl21a',
                 'Sirpa', 'Fibin',
                 'Cd3d', 'Cd3e', 'Cd19', 'Ms4a1', 'Cd68', 'Plac8', 'Dab2', 'C1qa',
                 'Mcpt4', 'Mcpt8', 'Itgax', 'Retnlg', 'Gzma', 'Areg',
                 'Acta2', 'Pdgfrb', 'Hhip', 'Pdgfra', 'Adh1',
                 ]
        adata_tmp = adata_ge.raw.to_adata()[:, genes]
        clusters = adata_ge.obs['leiden'].cat.categories
        ge_avg = np.zeros((len(clusters), len(genes)))
        for i, clu in enumerate(clusters):
            adatai = adata_tmp[adata_tmp.obs['leiden'] == clu]
            avgi = adatai.X.mean(axis=0)
            ge_avg[i] = avgi
        ge_avg = pd.DataFrame(ge_avg, index=clusters, columns=genes)

        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, leaves_list

        pdis = pdist(ge_avg.values)
        Zrow = linkage(pdis, method='average', optimal_ordering=True)
        idx_cells = leaves_list(Zrow) 

        pdis = pdist(ge_avg.values.T, metric='correlation')
        Zcol = linkage(pdis, method='average', optimal_ordering=True)
        idx_genes = leaves_list(Zcol)
        gene_names = [genes[i] for i in idx_genes]

        sns.clustermap(
            ge_avg, row_linkage=Zrow, col_linkage=Zcol,
            xticklabels=True, yticklabels=True,
        )

        assignment_d = {
            'Gja5': 'Arterial EC',
            'Ccl21a': 'Lymphatic EC',
            'Car8': 'Venous EC',
            'Hpgd': 'CAP 1',
            'Car4': 'CAP2',
            'C1qa': 'Interstitial mac',
            'Itgax': 'Alveolar mac',
            'Itgae': 'cDC1',
            'Cd209a': 'cDC2',
            'Mreg': 'cDC3',
            'Mcpt8': 'Mast',
            'Mcpt4': 'Basophil',
            'Retnlg': 'Neutrophil',
            'Ms4a1': 'B cell',
            'Gzma': 'NK cell',
            'Cd3e': 'T cell',
            'Areg': 'ILC2',
            'Col13a1': 'Alveolar FB',
            'Col14a1': 'Adventitial FB',
            'Hhip': 'ASM',
            'Higd1b': 'Pericyte',
            'Prrx1': 'VSM',
            'Epcam': 'Epithelial',
            'Actc1': 'ASM',
        }
        genes = list(assignment_d.keys())
        adata_tmp = adata_ge.raw.to_adata()[:, genes]
        clusters = adata_ge.obs['leiden'].cat.categories
        ge_avg = np.zeros((len(clusters), len(genes)))
        for i, clu in enumerate(clusters):
            adatai = adata_tmp[adata_tmp.obs['leiden'] == clu]
            avgi = adatai.X.mean(axis=0)
            ge_avg[i] = avgi
        ge_avg = pd.DataFrame(ge_avg, index=clusters, columns=genes)

        # Automatic detection based on marker genes
        cell_type_d = {}
        for gene in genes:
            cell_type_d[ge_avg[gene].idxmax()] = assignment_d[gene]
        for clu in clusters:
            if clu not in cell_type_d:
                cell_type_d[clu] = clu
        # Manual corrections
        manual_corr = {
            '0': 'Alveolar FB',
            '1': 'Low-quality',
            '2': 'CAP 1',
            '3': 'Epithelial',
            '7': 'Alveolar FB',
            '5': 'Adventitial FB',
            '20': 'VSM',
            '8': 'Epithelial 2',
            '9': 'Epithelial 3',
            '11': 'Mese of some kind?',
            '12': 'ASM',
            '15': 'Epithelial 3',
        }
        cell_type_d.update(manual_corr)
        adata_ge.obs['Cell Type'] = pd.Categorical(
            adata_ge.obs['leiden'].map(cell_type_d),
        )
