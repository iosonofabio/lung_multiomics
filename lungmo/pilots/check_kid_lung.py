# vim: fdm=indent
'''
author:     Fabio Zanini
date:       12/03/20
content:    Check lung data from newborns or kids
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet import Dataset, CountsTable, FeatureSheet, CountsTableSparse, SampleSheet


if __name__ == '__main__':

    pa = argparse.ArgumentParser()
    pa.add_argument('--age', choices=['d1', 'm21'], required=True)
    args = pa.parse_args()

    ds = Dataset(
        counts_table='human_{:}'.format(args.age),
        )
    if ds.counts._normalized:
        # Default normalization is counts per 10,000 reads
        ds.counts *= 100
    else:
        ds.counts.normalize(inplace=True)

    if True:
        print('Feature selection')
        features = ds.feature_selection.overdispersed(inplace=False)
        dsf = ds.query_features_by_name(features)

        print('PCA')
        dsc = dsf.dimensionality.pca(n_dims=30, robust=False, return_dataset='samples')

        print('tSNE')
        vs = dsc.dimensionality.tsne(perplexity=40)
    else:
        vs = pd.read_csv('../../data/lungmap/human_{:}/tsne.tsv'.format(args.age), sep='\t', index_col=0)


    edges = dsc.graph.knn(return_kind='edges')
    labels = ds.cluster.leiden('samples', edges, resolution_parameter=0.01)
    ds.samplesheet['cluster'] = [str(x) for x in labels]

    if True:
        fig, axs = plt.subplots(4, 8, figsize=(13, 7), sharex=True, sharey=True)
        axs = axs.ravel()
        marker_genes = [
                'EPCAM',
                ('PECAM1', 'CD31'),
                ('PTPRC', 'CD45'),
                'COL6A2',
                'CD3E',
                'GZMA',
                ('MS4A1', 'CD20'),
                'CPA3',
                #'MCPT4',
                'PLAC8',
                'CD68',
                'BATF3',
                #'CD209C',
                #'STFA2',
                'SFTPA1',
                'SFTPC',
                'SFTPB',
                'TFF3',
                'CCL21',
                'MSRB2',
                'ACTA2',
                'TAGLN',
                'SPARC',
                'ADIRF',
                'CD74',
                'HLA-DRA',
                'COL6A3',
                'LUM',
                'MKI67',
                'ACE2',
                ]
        markers = marker_genes + [
                'cluster',
            ]
        mgs = [x if isinstance(x, str) else x[0] for x in marker_genes]
        for ipl, (gene, ax) in enumerate(zip(markers, axs)):
            print('Plotting gene {:} of {:}'.format(ipl+1, len(markers)))
            if isinstance(gene, str):
                gene, title = gene, gene
            else:
                gene, title = gene
            if gene in ('cluster', 'cellSubtype'):
                tmp1 = ds.samplesheet[gene].unique()
                ncol = len(tmp1)
                ind = np.arange(ncol)
                np.random.shuffle(ind)
                cmap = sns.color_palette('Set1', n_colors=ncol)
                cmap = [cmap[i] for i in ind]
                cmap = dict(zip(tmp1, cmap))
            elif gene == 'cellType':
                cmap = sns.color_palette('Set1', n_colors=6)
            else:
                cmap = 'viridis'
            ds.plot.scatter_reduced_samples(
                    vs,
                    ax=ax,
                    s=10,
                    alpha=0.24 + 0.1 * (gene not in ['annotated', 'cluster', 'cellType', 'Mousename', 'Treatment', 'Timepoint']),
                    color_by=gene,
                    color_log=(gene in mgs + ['number_of_genes_1plusreads']),
                    cmap=cmap,
                    )
            ax.grid(False)
            ax.set_title(title)

            if gene == 'cluster':
                for com in ds.samplesheet['cluster'].unique():
                    vsc = vs.loc[ds.samplesheet[gene] == com]
                    xc, yc = vsc.values.mean(axis=0)
                    ax.scatter([xc], [yc], s=10, facecolor='none', edgecolor='red', marker='^')
                    ax.text(xc, yc, str(com), fontsize=8, ha='center', va='bottom')

            if gene in ('Treatment', 'Timepoint'):
                import matplotlib.lines as mlines
                d = ax._singlet_cmap
                handles = []
                labels = []
                for key, color in d.items():
                    h = mlines.Line2D(
                        [], [], color=color, marker='o', lw=0,
                        markersize=5,
                        )
                    handles.append(h)
                    if gene == 'Treatment':
                        key = key[0]
                        loc = 'upper left'
                    else:
                        loc = 'lower left'
                    labels.append(key.upper())
                if gene == 'Timepoint':
                    labels_old = list(labels)
                    labels = ['E18.5', 'P1', 'P7', 'P21']
                    handles = [handles[labels_old.index(li)] for li in labels]
                    ncol = 2
                else:
                    ncol = 1
                ax.legend(handles, labels, loc=loc, fontsize=6, ncol=ncol)

        fig.tight_layout()

    if True:
        for cl in ds.samplesheet['cluster'].unique():
            ds.samplesheet['is_cluster_{:}'.format(cl)] = ds.samplesheet['cluster'] == cl

        comps = {}
        for cl in np.sort(ds.samplesheet['cluster'].unique()):
            print('Find markers for cluster {:}'.format(cl))
            dsp = ds.split('is_cluster_{:}'.format(cl))
            if dsp[True].n_samples < 5:
                continue
            elif dsp[True].n_samples > 300:
                dsp[True].subsample(300, inplace=True)

            dsp[False].subsample(300, inplace=True)
            comp = dsp[True].compare(dsp[False])
            comp['cluster'] = cl
            comps[cl] = comp

            print(comp.loc[comp['log2_fold_change'] > 0].nlargest(10, 'statistic'))


    dsa = ds.average('samples', 'cluster')
    print(dsa.counts.loc['ACE'])
    print(dsa.counts.loc['ACE2'])

    plt.ion()
    plt.show()

