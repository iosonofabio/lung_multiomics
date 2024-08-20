# vim: fdm=indent
'''
author:     Fabio Zanini
date:       20/08/24
content:    Check the VDJ assemblies of Toshie's T cells.
'''
import os
import numpy as np
import pandas as pd
import anndata
import anndataks
import matplotlib
matplotlib.use('Gtk4agg')
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    seq_fdn = '../data/Tcells_Toshie/vdjpuzzle/final_receptor_results/'
    fnd = {x: f"{seq_fdn}TR{x}.tsv" for x in ['A', 'B', 'D']}

    dfd = {x: pd.read_csv(fnd[x], sep='\t') for x in fnd}

    dfd['B'].groupby(['v_call', 'j_call']).size().sort_values()

    dfd['A']['CellName'] = dfd['A']['CellID'].str.slice(4)
    dfd['B']['CellName'] = dfd['B']['CellID'].str.slice(4)
    dfd['D']['CellName'] = dfd['D']['CellID'].str.slice(4)

    presence = {}
    for chain, df in dfd.items():
        for _, row in df.iterrows():
            cname = row['CellName']
            if cname not in presence:
                presence[cname] = {x: False for x in dfd}
            presence[cname][chain] = True
    presence = pd.DataFrame(presence).T

    print('Check how many have both alpha and beta chain')
    presence[['A', 'B']].all(axis=1).sum()

    adata = anndata.read_h5ad('../data/Tcells_Toshie/acz_tcell.gz.h5ad')

    presence_a = presence.loc[presence.index.isin(adata.obs_names)]
    
    adata.obs.loc[presence_a.index].groupby(['Timepoint', 'Treatment']).size()
    adata.obs.groupby(['Timepoint', 'Treatment']).size()

    print('Identify large quasi-clones')
    def subindex(df, boolidx):
        cellnames = df.loc[boolidx, 'CellName'].values
        cellnames = cellnames[pd.Index(cellnames).isin(adata.obs_names)]
        adata_sub = adata[cellnames]
        return adata_sub

    largest_Aclone = dfd['A'].groupby(['v_call', 'j_call']).size().idxmax()
    adata_largestA = subindex(dfd['A'], (dfd['A']['v_call'] == largest_Aclone[0]) & (dfd['A']['j_call'] == largest_Aclone[1]))
    adata_counterA = subindex(dfd['A'], (dfd['A']['v_call'] != largest_Aclone[0]) & (dfd['A']['j_call'] != largest_Aclone[1]))
    degA = anndataks.compare(adata_largestA, adata_counterA, log1p=True)
    table_largestA = adata_largestA.obs.groupby(['Timepoint', 'Treatment']).size().unstack(fill_value=0)
    table_counterA = adata_counterA.obs.groupby(['Timepoint', 'Treatment']).size().unstack(fill_value=0)

    # Ppia is lower in the large alpha clone than elsewhere, which could mean the large clone is more activated - Ppia inhibits CD4+ T cell signal transduction here https://pubmed.ncbi.nlm.nih.gov/15308100/
    # Ssr2 is higher in large clone alpha - perhaps more ER stress as result of the activation

    largest_Bclone = dfd['B'].groupby(['v_call', 'j_call']).size().idxmax()
    adata_largestB = subindex(dfd['B'], (dfd['B']['v_call'] == largest_Bclone[0]) & (dfd['B']['j_call'] == largest_Bclone[1]))
    adata_counterB = subindex(dfd['B'], (dfd['B']['v_call'] != largest_Bclone[0]) & (dfd['B']['j_call'] != largest_Bclone[1]))
    degB = anndataks.compare(adata_largestB, adata_counterB, log1p=True)
    table_largestB = adata_largestB.obs.groupby(['Timepoint', 'Treatment']).size().unstack(fill_value=0)
    table_counterB = adata_counterB.obs.groupby(['Timepoint', 'Treatment']).size().unstack(fill_value=0)

    # Foxp1 is lower in the large beta clone - see https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2022.971045/full
    # lower Foxp1 could mean exit from naive/quiescence
    #
    # Sec61g is also lower in large beta clone - meaning less ER translocation and folding I guess - https://pubmed.ncbi.nlm.nih.gov/32146280/

    print('Check VDJ gene usage across conditions')
    dfdO = {x: dfd[x].loc[dfd[x]['CellName'].isin(adata.obs_names)] for x in dfd}
    for x, dfo in dfdO.items():
        for col in ['Timepoint', 'Treatment']:
            dfo[col] = ''
            for idx, row in dfo.iterrows():
                cname = row['CellName']
                dfo.at[idx, col] = adata.obs.at[cname, col]

    for x in dfdO:
        for gtype in ['v', 'j']:
            col = f'{gtype}_call'
            tmp = dfdO[x].groupby([col, 'Timepoint', 'Treatment']).size().unstack(col, fill_value=0).T
            fig, ax = plt.subplots()
            sns.heatmap(tmp, ax=ax)
            ax.set_title(f'{x}, {gtype} gene')
            fig.tight_layout()
    plt.ion(); plt.show()

    # NOTE: delta chains are very noisy, ignore for now
    def get_entropies(subsample=True):
        entropies = {}
        
        for x in ['A', 'B']:
            for gtype in ['v', 'j']:
                col = f'{gtype}_call'
                tmp = dfdO[x].groupby([col, 'Timepoint', 'Treatment']).size().unstack(col, fill_value=0).T

                if subsample:
        # subsample if afraid we are running into sampling issues
                    for tp in ['P7', 'P21']:
                        nmin = tmp[tp].sum(axis=0).min()
                        for tr in ['H', 'N']:
                            tmpi = tmp[(tp, tr)]
                            if tmpi.sum() > nmin:
                                tmp_ext = sum([[name] * n for name, n in tmpi.items()], [])
                                np.random.shuffle(tmp_ext)
                                tmpi_sub = pd.Series(tmp_ext[:nmin]).value_counts()
                                tmpi[:] = 0
                                for name, n in tmpi_sub.items():
                                    tmpi.loc[name] = n
                    # end of subsampling

                tmp_frac = tmp / tmp.sum(axis=0)
                entropy = -(tmp_frac * np.log2(tmp_frac + 1e-5)).sum(axis=0)
                entropies[(x, gtype)] = entropy
        entropies = pd.DataFrame(entropies).T
        return entropies

    entropies = get_entropies()

    fig, ax = plt.subplots()
    sns.heatmap(entropies, ax=ax)
    fig.tight_layout()
    plt.ion(); plt.show()

    deltaS = {tp: entropies[(tp, 'H')] - entropies[(tp, 'N')] for tp in ['P7', 'P21']}
    deltaS = pd.DataFrame(deltaS)

    fig, ax = plt.subplots()
    vmax = np.abs(deltaS).values.max()
    sns.heatmap(deltaS, ax=ax, center=0, vmin=-1.1 * vmax, vmax=1.1 * vmax, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    ax.set_title('Gene usage entropy (HO - N)')
    ax.set_ylabel('TCR chain - genetic locus')
    ticklabels = []
    for tl in ax.get_yticklabels():
        txt = tl.get_text()
        label = {'A': '$\\alpha$', 'B': '$\\beta$', 'D': '$\\delta$'}[txt[0]] + '-' + txt[2].upper()
        ticklabels.append(label)
    ax.set_yticklabels(ticklabels)
    fig.tight_layout()
    plt.ion(); plt.show()

    print('Bootstrap entropies and compute distribution of deltaS')
    deltaSs = []
    for i in range(100):
        entropies_i = get_entropies(subsample=True)
        deltaSi = {tp: entropies_i[(tp, 'H')] - entropies_i[(tp, 'N')] for tp in ['P7', 'P21']}
        deltaSi = pd.DataFrame(deltaSi).stack()
        deltaSi.name = str(i)
        deltaSs.append(deltaSi)
    deltaSs = pd.concat(deltaSs, axis=1).T

    fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(3, 6))
    ylabels = []
    xlabels = ['P7', 'P21']
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap_clip = 0.2
    cmap_tr = lambda x: cmap((np.clip(x, -cmap_clip, cmap_clip) + cmap_clip) / 2 / cmap_clip)
    for i, chain in enumerate(['A', 'B']):
        for il, locus in enumerate(['v', 'j']):
            txt = f'{chain}-{locus}'
            label = {'A': '$\\alpha$', 'B': '$\\beta$', 'D': '$\\delta$'}[txt[0]] + '-' + txt[2].upper()
            ylabels.append(label)
            ax = axs[i * 2 + il, 0].set_ylabel(label)
            for it, tp in enumerate(xlabels):
                ax = axs[i * 2 + il, it]
                datum = deltaSs[(chain, locus, tp)]
                mean = datum.mean()
                color = cmap_tr(mean)
                viod = ax.violinplot(datum)
                for key in ['bodies', 'cmins', 'cmaxes', 'cbars']:
                    if key in viod:
                        tt = viod[key]
                        if isinstance(tt, list):
                            tt = tt[0]
                        tt.set_color(color)
                if i + il == 0:
                    ax.set_title(tp)
                ax.set_xticks([])
                ax.grid(True, axis='y')
    fig.tight_layout()
    plt.ion(); plt.show()


