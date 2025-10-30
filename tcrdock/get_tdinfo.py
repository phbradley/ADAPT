######################################################################################88
from collections import OrderedDict, Counter
#import os
#import sys
#from os.path import exists
#from .tcrdist.all_genes import all_genes
#from .tcrdist.amino_acids import amino_acids
#from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
#from Bio.pairwise2 import format_alignment
#from .util import path_to_db
#from .docking_geometry import DockingGeometry
#from .tcrdock_info import TCRdockInfo
#from .tcrdock_info import TCRdockInfo
#from . import docking_geometry
from . import tcrdock_info
from . import tcrdist
from . import mhc_util
from . import sequtil
#from . import superimpose
#from . import pdblite
import pandas as pd
import numpy as np
#import random
#import copy
#from numpy.linalg import norm



_mhc_seq2core_positions = {}
def get_mhc_core_positions_0x_cached(mhc_class, mhc_seq, verbose=True):
    global _mhc_seq2core_positions
    # if mhc_class==2 mhc_seq is a tuple
    if (mhc_class, mhc_seq)  not in _mhc_seq2core_positions:
        if verbose:
            print('get mhc core:', mhc_class, mhc_seq)
        if mhc_class==1:
            posl = mhc_util.get_mhc_core_positions_class1(mhc_seq)
        else:
            posl = mhc_util.get_mhc_core_positions_class2(*mhc_seq)
        _mhc_seq2core_positions[(mhc_class,mhc_seq)] = posl

    return _mhc_seq2core_positions[(mhc_class, mhc_seq)]

def get_model_tdinfo(
        organism,
        mhc_class,
        mhc_allele,
        cb_seq,
        va,
        ja,
        cdr3a,
        vb,
        jb,
        cdr3b,
        verbose = False,
        trust_tcr_positions = False,
):
    ''' return tdinfo

    meant to be faster than starting from sequences

    but falls back on slow parsing if the tcr framework sequences dont match
    expectation

    '''
    from .sequtil import ALL_GENES_GAP_CHAR
    from .tcrdist.all_genes import all_genes
    from .tcrdist.parsing import parse_tcr_sequence

    is_unbound_tcr = mhc_allele is None or pd.isna(mhc_allele)
    is_unbound_pmhc = va is None or pd.isna(va)

    if is_unbound_pmhc:
        assert vb is None or pd.isna(vb)

    core_len = 13 # tcr core len

    if is_unbound_tcr:
        mhc_seqs = []
        pep_seq = ''
        tcra_seq, tcrb_seq = cb_seq.split('/')
    elif is_unbound_pmhc:
        *mhc_seqs, pep_seq = cb_seq.split('/')
        tcra_seq = ''
        tcrb_seq = ''
    else:
        *mhc_seqs, pep_seq, tcra_seq, tcrb_seq = cb_seq.split('/')
        
    if not is_unbound_tcr:
        assert mhc_class == len(mhc_seqs)

    mhc_len = sum(len(x) for x in mhc_seqs)
    pmhc_len = mhc_len + len(pep_seq)
    pmhc_tcra_len = pmhc_len + len(tcra_seq)
    
    if is_unbound_pmhc:
        tcr_core_positions = []
        tcr_cdrs = []

    else:
        tcra_prefix = sequtil.get_v_seq_up_to_cys(organism, va)[:-1] # dont require Cys
        tcrb_prefix = sequtil.get_v_seq_up_to_cys(organism, vb)[:-1]

        if trust_tcr_positions:
            assert tcra_seq[len(tcra_prefix)] == 'C'
            assert tcrb_seq[len(tcrb_prefix)] == 'C'

        if (trust_tcr_positions or
            (tcra_seq.startswith(tcra_prefix) and tcrb_seq.startswith(tcrb_prefix))):
            # yay! we got a match to 'clean' model tcr seqs up to the Cys
            tcr_core_positions = (
                [x+pmhc_len      for x in sequtil.get_core_positions_0x(organism, va)]+
                [x+pmhc_tcra_len for x in sequtil.get_core_positions_0x(organism, vb)])

            tcr_cdrs = []
            for v,preseq,seq,cdr3,offset in [
                    [va,tcra_prefix,tcra_seq,cdr3a,pmhc_len],
                    [vb,tcrb_prefix,tcrb_seq,cdr3b,pmhc_tcra_len]]:
                g = all_genes[organism][v]
                for gapcdr in g.cdrs[:3]+[cdr3]:
                    cdr = gapcdr.replace(ALL_GENES_GAP_CHAR,'')
                    if cdr != cdr3 and cdr in cdr3:
                        # special wonky case
                        start = seq[:seq.index(cdr3)].index(cdr) + offset
                    elif cdr==cdr3:
                        assert seq.count(cdr) == 1
                        start = seq.index(cdr) + offset
                    else:
                        assert preseq.count(cdr) == 1
                        start = preseq.index(cdr) + offset
                    tcr_cdrs.append([start, start+len(cdr)-1])
        else:
            # have to do it the slow and painful way
            print('WARNING get_model_tdinfo: slow and painful tcr parsing')
            aparse = parse_tcr_sequence(organism, 'A', tcra_seq)
            bparse = parse_tcr_sequence(organism, 'B', tcrb_seq)
            assert (aparse is not None) and (bparse is not None),\
                'get_model_tdinfo tcr seq parsing failed!'
            tcr_core_positions = (
                [x+pmhc_len      for x in aparse['core_positions']]+
                [x+pmhc_tcra_len for x in bparse['core_positions']]
            )
            tcr_cdrs = (
                [(x+pmhc_len     , y+pmhc_len     ) for x,y in aparse['cdr_loops']]+
                [(x+pmhc_tcra_len, y+pmhc_tcra_len) for x,y in bparse['cdr_loops']]
            )

        sequence = cb_seq.replace('/','')
        tcr_coreseq = ''.join(sequence[x] for x in tcr_core_positions)
        cys_seq = tcr_coreseq[1]+tcr_coreseq[12]+tcr_coreseq[14]+tcr_coreseq[25]
        if verbose or cys_seq != 'CCCC':
            print('tcr_coreseq:', cys_seq, tcr_coreseq[:core_len],
                  tcr_coreseq[core_len:])


    if is_unbound_tcr:
        mhc_core_positions_0x = []
    else:
        mhc_seq = mhc_seqs[0] if mhc_class==1 else tuple(mhc_seqs)
        mhc_core_positions_0x = get_mhc_core_positions_0x_cached(mhc_class, mhc_seq,
                                                                 verbose=False)

    tdinfo = tcrdock_info.TCRdockInfo().from_dict(
        dict(organism = organism,
             mhc_class = mhc_class,
             mhc_allele = mhc_allele,
             mhc_core = mhc_core_positions_0x,
             pep_seq = pep_seq,
             tcr = ((va, ja, cdr3a), (vb, jb, cdr3b)),
             tcr_core = tcr_core_positions,
             tcr_cdrs = tcr_cdrs,
             valid = True,
        ))

    return tdinfo

def get_row_tdinfo(row, **kwargs):
    if hasattr(row, 'tdinfo') and not pd.isna(row.tdinfo):
        return tcrdock_info.TCRdockInfo().from_string(row.tdinfo)
    elif hasattr(row, 'target_tdinfo') and not pd.isna(row.target_tdinfo):
        return tcrdock_info.TCRdockInfo().from_string(row.target_tdinfo)

    return get_model_tdinfo(
        row.organism, row.mhc_class, row.mhc, row.chainseq, row.va, row.ja, row.cdr3a,
        row.vb, row.jb, row.cdr3b, **kwargs,
    )


