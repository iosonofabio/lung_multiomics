# vim: fdm=indent
'''
author:     Fabio Zanini
date:       20/06/23
content:    Pilot snapatac tutorial for our data.
'''
import os
import pathlib
import numpy as np
import pandas as pd

import snapatac2 as snap



if __name__ == '__main__':

    fdn_data = pathlib.Path('data/6samples_June2023')
    fn_atac = fdn_data / 'p7_multiome_atac.gz.h5ad'
    fn_fragments = fdn_data / 'atac_fragments.tsv.gz'
    fn_gff = pathlib.Path('data/gene_annotations/gencode.v41.basic.annotation.gff3.gz')
    fn_snap = fdn_data / 'snapatac2_data.h5ad'

    # Import fragments
    data = snap.pp.import_data(
        fn_fragments,
        # Genome annotation and fasta file, cached in ~/.cache/snapatac2
        genome=snap.genome.mm10,
        file=fn_snap,
        sorted_by_barcode=False,
    )

    #print("Load data from file")
    #data = snap.read(fn_atac, backed=None)
