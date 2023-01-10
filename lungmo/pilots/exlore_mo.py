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


def plot_clustermap_avg(ge_avg, log=True):
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list

    if log:
        ge_avg = np.log(ge_avg + 0.1)

    pdis = pdist(ge_avg.values)
    Zrow = linkage(pdis, method='average', optimal_ordering=True)

    pdis = pdist(ge_avg.values.T, metric='correlation')
    Zcol = linkage(pdis, method='average', optimal_ordering=True)

    g = sns.clustermap(
        ge_avg, row_linkage=Zrow, col_linkage=Zcol,
        xticklabels=True, yticklabels=True,
    )

    return g


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

    if False:
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

    if False:
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
        if adata_ge.raw is not None:
            adata_tmp = adata_ge.raw.to_adata()[:, genes]
        else:
            adata_tmp = adata_ge[:, genes]
        clusters = adata_ge.obs['leiden'].cat.categories
        ge_avg = np.zeros((len(clusters), len(genes)))
        for i, clu in enumerate(clusters):
            adatai = adata_tmp[adata_tmp.obs['leiden'] == clu]
            avgi = adatai.X.mean(axis=0)
            ge_avg[i] = avgi
        ge_avg = pd.DataFrame(ge_avg, index=clusters, columns=genes)

        plot_clustermap_avg(ge_avg, log=False)


    if True:
        print('Assign cell types to clusters')
        adata_ge = adata[:, adata.var['feature_types'] == 'Gene Expression']
        adata_pe = adata[:, adata.var['feature_types'] == 'Peaks']
        sc.pp.normalize_total(adata_ge, target_sum=1e4)
        sc.pp.normalize_total(adata_pe, target_sum=1e4)
        adata_nor = anndata.concat([adata_ge, adata_pe], axis=1, merge='first')

        assignment_d = {
            'Gja5': 'Arterial EC',
            'Ccl21a': 'Lymphatic EC',
            'Slc6a2': 'Venous EC',
            'Hpgd': 'CAP1',
            'Car4': 'CAP2',
            'C1qa': 'Interstitial mac',
            'Itgax': 'Alveolar mac',
            'Plac8': 'Monocyte',
            'Itgae': 'cDC1',
            'Cd209a': 'cDC2',
            #'Mreg': 'cDC3',
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
            'Actc1': 'ASM',
            'Slc34a2': 'AT2',
            'Col4a4': 'AT1',
            'Cdkn1c': 'Ciliated',
            'Reg3g': 'Club',
            'Snap25': 'Neuron',
        }
        other_genes = [
                'Ptprc', 'Cdh5', 'Epcam', 'Col6a2',
                'Cd68', 'Pecam1', 'Krt19', 'Col1a1', 'Lyz1',
                'Lyz2', 'Pdgfra', 'Pdgfrb', 'Tgfbi',
                ]
        genes = list(assignment_d.keys()) + other_genes
        if adata_ge.raw is not None:
            adata_tmp = adata_ge.raw.to_adata()[:, genes]
        else:
            adata_tmp = adata_ge[:, genes]
        clusters = adata_ge.obs['leiden'].cat.categories
        ge_avg = np.zeros((len(clusters), len(genes)))
        for i, clu in enumerate(clusters):
            adatai = adata_tmp[adata_tmp.obs['leiden'] == clu]
            avgi = adatai.X.mean(axis=0)
            ge_avg[i] = avgi
        ge_avg = pd.DataFrame(ge_avg, index=clusters, columns=genes)

        # Automatic detection based on marker genes
        cell_type_d = {}
        for gene in assignment_d:
            cell_type_d[ge_avg[gene].idxmax()] = assignment_d[gene]
        for clu in clusters:
            if clu not in cell_type_d:
                cell_type_d[clu] = clu

        ge_avg_tmp = ge_avg.copy()
        ge_avg_tmp.index = ge_avg_tmp.index.map(cell_type_d)
        #plot_clustermap_avg(ge_avg_tmp, log=True)

        # Manual corrections
        manual_corr = {
            '0': 'Alveolar FB',
            '1': 'Low-quality',
            '2': 'CAP1',
            '3': 'Epithelial',
            '4': 'Myofibroblast',
            '5': 'Adventitial FB',
            '7': 'Alveolar FB',
            '8': 'Club',
            '11': 'Doublet',
            '17': 'Unknown',
            '20': 'VSM',
            '23': 'Doublet',
            '25': 'Venous EC',
            '29': 'Doublet',
            '30': 'Doublet',
            '32': 'Doublet',

        }
        cell_type_d.update(manual_corr)

        ge_avg_tmp = ge_avg.copy()
        ge_avg_tmp.index = ge_avg_tmp.index.map(cell_type_d)
        plot_clustermap_avg(ge_avg_tmp, log=True)

        adata_ge.obs['Cell Type'] = pd.Categorical(
            adata_ge.obs['leiden'].map(cell_type_d),
        )
        adata_ge.obs['High-quality'] = (~adata_ge.obs['Cell Type'].isin(
                ['Doublet', 'Low-quality', 'Unknown'],
                )).astype('i2')
        adata_nor.obs['Cell Type'] = adata_ge.obs['Cell Type']
        adata_nor.obs['High-quality'] = adata_ge.obs['High-quality']

    if True:
        print('DE and DA within lineages')
        lineages = {
            'Endothelial': [
                'Arterial EC',
                'Venous EC',
                'Lymphatic EC',
                'CAP1',
                'CAP2',
            ],
            'Immune': [
                'T cell',
                'B cell',
                'NK cell',
                'Alveolar mac',
                'Interstitial mac',
                'Monocyte',
            ],
            # TODO
        }

        lineage = 'Endothelial'
        adata_nor = adata_nor[adata_nor.obs['High-quality'] == 1]
        adata_nor_gr = adata_nor[adata_nor.obs['Cell Type'].isin(lineages[lineage])]
        adata_ge = adata_nor_gr[:, adata_nor_gr.var['feature_types'] == 'Gene Expression']
        adata_pe = adata_nor_gr[:, adata_nor_gr.var['feature_types'] == 'Peaks']
        sc.tl.rank_genes_groups(adata_pe, 'Cell Type', method='wilcoxon')
        sc.tl.rank_genes_groups(adata_ge, 'Cell Type', method='wilcoxon')

        from collections import defaultdict
        da_lineage_dict = defaultdict(list)
        for ftype, adatai in [('Peaks', adata_pe), ('Gene Expression', adata_ge)]:
            tmp = adatai.uns['rank_genes_groups']
            cell_types = tmp['names'].dtype.names
            for i, ct in enumerate(cell_types):
                idx = [x[i] for x in tmp['names']]
                log_fc = [x[i] for x in tmp['logfoldchanges']]
                score = [x[i] for x in tmp['scores']]
                pval = [x[i] for x in tmp['pvals']]
                df = pd.DataFrame({
                    'feature': idx,
                    'logfc': log_fc,
                    'score': score,
                    'pval': pval,
                }).set_index('feature')
                df.sort_values('score', ascending=False, inplace=True)
                df['rank'] = np.arange(len(df)) + 1

                for col in ['chromosome', 'start', 'end']:
                    df[col] = adatai.var.loc[df.index, col]
                df['feature_types'] = ftype
                da_lineage_dict[ct].append(df)
        for ct, dfs in da_lineage_dict.items():
            da_lineage_dict[ct] = pd.concat(dfs, axis=0)

    def plot_de_on_chroms(var, df):
        chromosomes = [f'chr{i+1}' for i in range(19)] + ['chrX', 'chrY']
        nchr = len(chromosomes)
        chr_max = (var[['chromosome', 'start', 'end']]
                      .groupby('chromosome')
                      .max()
                      .max(axis=1)
                      .loc[chromosomes]) + 1

        fig, ax = plt.subplots(figsize=(10, 10))

        # Chromosome tracks
        for i, chrom in enumerate(chromosomes):
            ax.plot([0, 100], [nchr - 1 - i] * 2, lw=1, color='k', alpha=0.3)

        cmap = {'Gene Expression': 'tomato', 'Peaks': 'steelblue'}
        offsets = {'Gene Expression': +0.1, 'Peaks': -0.1}
        for feature, row in df.iterrows():
            chrom = row['chromosome']
            if chrom not in chromosomes:
                continue
            ftype = row['feature_types']
            irow = nchr - 1 - chromosomes.index(chrom)
            x0 = 100.0 * row['start'] / chr_max[chrom]
            x1 = 100.0 * row['end'] / chr_max[chrom]
            # Min width 0.5% of chromosome, otherwise you don't even see them
            if (x1 - x0) < 0.5:
                x0 -= 0.25
                x1 += 0.25
            ax.plot(
                [x0, x1], [irow + offsets[ftype]] * 2,
                color=cmap[ftype], lw=3, alpha=0.8,
            )
            xm = 0.5 * (x0 + x1)
            if ftype == 'Gene Expression':
                ax.text(xm, irow + 0.15, feature, ha='center', va='bottom')
                strand = var.at[feature, 'strand']
                # Mark the beginning
                if strand == -1:
                    ax.plot(
                        [x0 + 0.8 * (x1 - x0), x1],
                        [irow + offsets[ftype]] * 2,
                        color=cmap[ftype], lw=4, alpha=0.9,
                        )
                else:
                    ax.plot(
                        [x0, x0 + 0.2 * (x1 - x0)],
                        [irow + offsets[ftype]] * 2,
                        color=cmap[ftype], lw=4, alpha=0.9,
                        )

        labels = ['Gene', 'Open chromatin']
        handles = [
            plt.Rectangle((0, 0), 0, 0, color=cmap['Gene Expression']),
            plt.Rectangle((0, 0), 0, 0, color=cmap['Peaks']),
        ]
        ax.legend(handles, labels,
                  loc='center',
                  bbox_to_anchor=(0.85, 1.015), bbox_transform=ax.transAxes,
                  ncol=2,
                  frameon=False,
                  )

        ax.set_yticks(np.arange(0, len(chromosomes)))
        ax.set_yticklabels(chromosomes[::-1])
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel('Position within chromosome[%]')
        ax.set_ylim(-0.5, len(chromosomes) - 0.5)
        ax.set_xlim(-1, 101)
        fig.tight_layout()
        
        return (fig, ax)

    for ct in da_lineage_dict:
        fig, ax = plot_de_on_chroms(
            adata.var,
            da_lineage_dict[ct].nsmallest(100, 'rank'),
        ) 
        ax.set_title(ct)
        fig.savefig(
            f'../../figures/pilots/de_da_across_chromosomes_endos/{ct}.png',
        )
