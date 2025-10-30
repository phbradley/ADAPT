''' Evaluate some loop designs by

* rescoring with alphafold
* relaxing with rosetta

inputs:

--targets:  tsvfile with required columns:

* targetid
* chainseq
* model_pdbfile
* something to define what the "designable" positions were

'''

required_cols = 'targetid chainseq model_pdbfile'.split()
need_one_of_cols = 'designable_positions template_0_target_to_template_alignstring'\
                   .split()

import os
import design_paths

import argparse
parser = argparse.ArgumentParser(description="evaluate designs")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--relax_rescored_model', action='store_true')
parser.add_argument('--skip_alphafold', action='store_true')
parser.add_argument('--skip_rosetta', action='store_true')
parser.add_argument('--skip_sasas', action='store_true')
parser.add_argument('--skip_mpnn_peptide_probs', action='store_true')
parser.add_argument('--norelax', action='store_true')
parser.add_argument('--ex_flags', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--extend_flex', type=int, default=1)
parser.add_argument('--num_batches', type=int, default=1)
parser.add_argument('--batch_num', type=int, default=0)


args = parser.parse_args()

if design_paths.FRED_HUTCH_HACKS and not args.skip_alphafold:
    assert os.environ['LD_LIBRARY_PATH'].startswith(
        '/home/pbradley/anaconda2/envs/af2/lib:'),\
        'export LD_LIBRARY_PATH=/home/pbradley/anaconda2/envs/af2/lib:$LD_LIBRARY_PATH'

## more imports ####################
design_paths.setup_import_paths()
import sys
from sys import exit
import copy
import itertools as it
from pathlib import Path
from os.path import exists, isdir
import os
import tcrdock as td2
from tcrdock.tcrdist.amino_acids import amino_acids
import pandas as pd
import numpy as np
import random
from os import system, popen, mkdir
from glob import glob
from collections import Counter, OrderedDict, namedtuple
import scipy
import json

import design_stats
import wrapper_tools


######################################################################################88
## functions
######################################################################################88


def run_alphafold_rescoring(
        targets,
        outprefix,
        model_name='model_2_ptm',
        extend_flex=args.extend_flex,
        gapchar='-',
):
    ''' will mask out with gapchar the peptide and all the unaligned residues in the
    template_0_target_to_template_alignstring's plus extend_flex rsds on either side
    of each loop segment
    '''
    dfl = []
    for l in targets.itertuples():
        sequence = l.chainseq.replace('/','')
        nres = len(sequence)
        cbs = [0]+list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        nres_mhc, nres_pmhc = cbs[-4:-2]

        loop_flex_posl = design_stats.get_designable_positions(
            row=l, extend_flex=extend_flex)

        full_alignstring = ';'.join(f'{i}:{i}' for i in range(nres))

        flex_posl = loop_flex_posl + list(range(nres_mhc, nres_pmhc)) # all of pep

        alt_seq = ''.join(gapchar if i in flex_posl else x
                          for i,x in enumerate(sequence))
        dfl.append(dict(
            targetid = f'{l.targetid}_rescore',
            chainseq = l.chainseq,
            template_0_template_pdbfile = l.model_pdbfile,
            template_0_target_to_template_alignstring = full_alignstring,
            template_0_alt_template_sequence = alt_seq,
        ))
    rescore_targets = pd.DataFrame(dfl)
    rescored_targets = wrapper_tools.run_alphafold(
        rescore_targets, outprefix, model_name=model_name).set_index('targetid')

    # compute stats using the paes/plddts for the rescoring run
    # add those to targets df
    #
    dfl = []
    for _, l in targets.iterrows():
        l2 = rescored_targets.loc[l.targetid+'_rescore']
        # pass 0 here since we already extended when we set up alt_template_sequence
        l2 = design_stats.add_info_to_rescoring_row(l2, model_name, extend_flex=0)
        if hasattr(l,'peptide') and not pd.isna(l.peptide):
            assert l.peptide == l2.peptide
        # not sure about this next one if we are tweaking designable_positions
        # if hasattr(l,'loop_seq'):
        #     assert l.loop_seq == l2.loop_seq

        outl = l.copy()
        outl['rescore_peptide_plddt'] = l2.peptide_plddt
        outl['rescore_loop_plddt'] = l2.loop_plddt
        outl['rescore_peptide_loop_pae'] = l2.peptide_loop_pae
        outl['rescore_pmhc_tcr_pae'] = l2.pmhc_tcr_pae
        outl['rescore_peptide'] = l2.peptide
        outl['rescore_loop_seq'] = l2.loop_seq
        outl['rescore_model_pdbfile'] = l2.model_pdbfile
        dfl.append(outl)

    return pd.DataFrame(dfl)


######################################################################################88
## main
######################################################################################88

targets = pd.read_table(args.targets, low_memory=False)

for col in required_cols:
    assert col in targets.columns, f'Need {col} column in {args.targets}'
assert any(col in targets.columns for col in need_one_of_cols),\
    f'Need one of {" ".join(need_one_of_cols)} in {args.targets}'

if targets.targetid.value_counts().max() >1:
    targets['targetid'] = [f'{x}_{i}' for i,x in enumerate(targets.targetid)]

if args.num_batches > 1: # subset to the batch targets
    mask = np.arange(targets.shape[0])%args.num_batches == args.batch_num
    print(f'num_batches: {args.num_batches} batch_num: {args.batch_num} '
          f'subset to {mask.sum()} out of {targets.shape[0]} targets')
    targets = targets[mask].copy()

# this doesn't really work for natives! at least not for run590 inputs
#if 'model_plddtfile' in targets.columns: # also model_paefile
#    # redo this because maybe it didn't have strength score, for example
#    targets = design_stats.compute_simple_stats(targets)

if not args.skip_alphafold: # run alphafold rescoring
    outprefix = f'{args.outfile_prefix}_afold_rescore'
    targets = run_alphafold_rescoring(targets, outprefix)

if not args.skip_rosetta: # run rosetta relax
    outprefix = f'{args.outfile_prefix}_relax'
    targets = wrapper_tools.run_rosetta_relax(
        targets, outprefix, ex_flags=args.ex_flags,
        relax_rescored_model= args.relax_rescored_model,
        norelax=args.norelax,
    )

if not args.skip_sasas: # run rosetta relax
    outprefix = f'{args.outfile_prefix}_sasas'
    targets = wrapper_tools.run_sasas_and_contacts(
        targets, outprefix, verbose=args.verbose,
    )

if not args.skip_mpnn_peptide_probs:
    # calc peptide probs
    outprefix = f'{args.outfile_prefix}_mpnn_pprobs'
    targets = wrapper_tools.run_mpnn_peptide_probs(
        targets, outprefix, num_mpnn_seqs=5, sampling_temp=1.0)
    targets['pepspec_delta'] = (targets.total_wt_pep_prob_full -
                                targets.total_wt_pep_prob_pmhc)

# write results
targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')




