######################################################################################88

import argparse
parser = argparse.ArgumentParser(
    description="alphafold dock design",
    epilog='''
Flexible-dock tcr design using the "random loops" approach

required arguments:
--pmhc_targets: TSV file with peptide-MHC target info (allele name, peptide seq)
--abids: string list of antibody ids to use as frameworks
--num_designs: total number of designs to generate
--outfile_prefix: string prefix prepended to outputs

--design_cdrs: int list of loops to design, default is both cdr3s
    0=CDR1L 1=CDR2L 2=CDR2.5L 3=CDR3L
    4=CDR1H 5=CDR2H 6=CDR2.5H 7=CDR3H

The script repeats the following steps '--num_designs' times:

STEP 1. pick a random pmhc target from list

STEP 2. pick a template antibody from the --abids list

STEP 3. pick CDR3 loops from a random paired TCR. Mutate the first 3 and last 2 aas
to match the template ab from STEP 2.

STEP 4. Provide this information (peptide,MHC,ab framework) to a
modified alphafold TCR docking protocol

STEP 5. Re-design the CDR loops (excluding first 3 and last 2 residues) using MPNN

STEP 6. Re-dock the TCR to the pMHC using the same alphafold docking protocol used
in step 4.

STEP 7. Re-dock the TCR to the pMHC using rf-antibody.

STEP 8. Compute final stats like pmhc_tcr_pae, peptide_loop_pae, and dock-rmsd between
first and second alphafold models.



Example command line:

python dock_design_ig.py --pmhc_targets my_pmhc_targets.tsv \\
    --abids 7sg5HL 7kqlHL 4hpyHL --design_cdrs 0 1 3 4 5 7 \\
    --num_designs 10  --outfile_prefix dock_design_test1

The --pmhc_targets file should have these columns:
    * organism ('human' or 'mouse')
    * mhc_class (1 or 2)
    * mhc (e.g. "A*02:01")
    * peptide (e.g. "GILGFVFTL")

$ head my_pmhc_targets.tsv
organism	mhc_class	mhc	peptide
human	1	A*01:01	EVDPIGHLY

email pbradley@fredhutch.org with questions

''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--pmhc_targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--num_designs', type=int, required=True)
parser.add_argument('--abids', nargs='*', required=True)
parser.add_argument('--design_cdrs', type=int, nargs='*')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--random_state', type=int)
parser.add_argument('--skip_rf_antibody', action='store_true',
                    help='dont run rf_antibody to evaluate the designs')
parser.add_argument('--reverse_rf_antibody_chains', action='store_true',
                    help='deprecated, now this is set based on reverse_dock')
parser.add_argument('--reverse_dock', action='store_true',
                    help='deprecated, now you can put this in --pmhc_targets '
                    'or use --both_orientations')
parser.add_argument('--both_orientations', action='store_true',
                    help='randomly choose forward or reverse docking orientation '
                    'for each design independently')

parser.add_argument('--model_name', default='model_2_ptm_ft_binder',
                    help='this doesnt really matter but it has to start with '
                    '"model_2_ptm_ft"')

parser.add_argument('--rf_antibody_chkpt', default=None,
                    help='Specify specific RFantibody model checkpoint, otherwise will use a default')

parser.add_argument('--model_params_file',
                    help='The default is a binder-fine-tuned model that was trained '
                    'on structures and a new distillation set')

args = parser.parse_args()


# other imports
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import pandas as pd
from os.path import exists
from os import mkdir, makedirs
import random
from collections import Counter, OrderedDict
from timeit import default_timer as timer
import copy
import numpy as np
from sys import exit

import design_stats
import wrapper_tools

if args.random_state is not None:
    random.seed(args.random_state)

if args.model_params_file is None:
    if args.model_name == 'model_2_ptm':
        print('using classic AlphaFold params:', args.model_name)
    else:
        args.model_params_file = design_paths.AF2_BINDER_FT_PARAMS
else:
    assert args.model_name != 'model_2_ptm'


## hard-coded -- these control how much sequence is retained from tcr template cdr3s
nterm_seq_stem = 3
cterm_seq_stem = 2
## defaults ##########


######################################################################################88
######################################################################################88
######################################################################################88

def make_templates_for_alphafold_ig(
        organism, # for pmhc
        abid, # like 7lxnHL    <pdbid><Hchain><Lchain>
        cdr3a,
        cdr3b,
        mhc_class,
        mhc_allele,
        peptide,
        outfile_prefix,
        verbose = False,
        next_best_identity_threshold_mhc=0.98,
        pmhc_template_pdbfile = None,
        reverse_dock = False,
):
    ''' Borrowed from tcrdock/sequtil.py, modified
    '''

    from tcrdock.pdblite import (
        apply_transform_Rx_plus_v, delete_chains, append_chains,
        dump_pdb, set_chainbounds_and_renumber, pose_from_pdb,
    )
    from tcrdock.sequtil import (
        BAD_PMHC_PDBIDS, ALL_GENES_GAP_CHAR, ternary_info, new_ternary_info,
        pmhc_info, new_pmhc_info, get_mhc_class_1_alseq, get_mhc_class_2_alseq,
        align_cdr3s
    )
    from tcrdock.tcrdock_info import TCRdockInfo
    from numpy.linalg import norm

    assert organism in ['human','mouse'] # organism for mhc
    num_templates_per_run = 4

    # check arguments
    if mhc_class == 2:
        assert len(peptide) == CLASS2_PEPLEN
        assert mhc_allele.count(',') == 1

    my_ternary_info = pd.concat([ternary_info, new_ternary_info])
    badmask = new_ternary_info.pdbid.isin(set(pmhc_info.pdbid))
    my_pmhc_info = pd.concat([pmhc_info, new_ternary_info[~badmask], new_pmhc_info])

    # read antibody pose and info
    ig_pdbfile = f'{design_paths.SABDAB_DIR}ig_only/{abid}_ig_only.pdb'
    ig_tdifile = ig_pdbfile+'.tcrdock_info.json'
    assert exists(ig_pdbfile) and exists(ig_tdifile)
    ig_pose = pose_from_pdb(ig_pdbfile)
    with open(ig_tdifile,'r') as f:
        ig_tdinfo = TCRdockInfo().from_string(f.read())



    def show_alignment(al,seq1,seq2):
        if verbose:
            for i,j in sorted(al.items()):
                a,b = seq1[i], seq2[j]
                star = '*' if a==b else ' '
                print(f'{i:4d} {j:4d} {a} {star} {b}')
            idents = sum(seq1[i] == seq2[j] for i,j in al.items())/len(seq1)
            print(f'idents: {idents:6.3f}')

    # setup the antibody alignment
    ig_align = {} # from trg to tmp
    trg_ig_seq = ''
    tmp_ig_seq = ''
    trg_ig_chainseq = []
    for ii, trg_cdr3 in enumerate([cdr3a, cdr3b]):
        start, stop = ig_tdinfo.tcr_cdrs[ii*4+3]
        tmp_cdr3 = ig_pose['sequence'][start:stop+1]
        assert tmp_cdr3[0] == 'C' == trg_cdr3[0] # might fail for wonky ab pdb
        tmp_seq = ig_pose['chainseq'].split('/')[ii]
        before_seq = tmp_seq[:start-len(tmp_ig_seq)]
        after_seq = tmp_seq[stop-len(tmp_ig_seq)+1:]
        assert tmp_seq == before_seq + tmp_cdr3 + after_seq
        for aseq, bseq in zip([before_seq, trg_cdr3, after_seq],
                              [before_seq, tmp_cdr3, after_seq]):
            if aseq == bseq:
                al = {i:i for i in range(len(aseq))}
            else:
                assert aseq[0] == 'C' == bseq[0]
                al = align_cdr3s(aseq, bseq)
            ig_align.update({i+len(trg_ig_seq):j+len(tmp_ig_seq) for i,j in al.items()})
            trg_ig_seq += aseq
            tmp_ig_seq += bseq
        trg_ig_chainseq.append(before_seq+trg_cdr3+after_seq)
    trg_ig_chainseq = '/'.join(trg_ig_chainseq)
    assert tmp_ig_seq == ig_pose['sequence']
    show_alignment(ig_align, trg_ig_seq, tmp_ig_seq)




    if pmhc_template_pdbfile is not None: # use the provided pmhc structure
        assert mhc_class == 1 # for the moment
        pose = td2.pdblite.pose_from_pdb(pmhc_template_pdbfile)
        pose = td2.pdblite.set_chainbounds_and_renumber(pose, list(pose['chainbounds']))

        assert len(pose['chains']) == 2
        tmp_mhc_seq, tmp_peptide = pose['chainseq'].split('/')
        tmp_allele = td2.mhc_util.get_mhc_allele(tmp_mhc_seq, organism)
        tmp_mhc_core = td2.mhc_util.get_mhc_core_positions_class1(tmp_mhc_seq)
        tmp_alignseq = td2.sequtil.get_mhc_class_1_mhc_alignseq_from_chainseq(
            tmp_allele, tmp_mhc_seq)
        tmp_pdbfile = outfile_prefix+'_user_pmhc_tmpl.pdb'
        pose = td2.mhc_util.orient_pmhc_pose(pose, mhc_core_positions=tmp_mhc_core)
        td2.pdblite.dump_pdb(pose, tmp_pdbfile)
        my_pmhc_info = pd.DataFrame([
            dict(organism = organism,
                 pdbid = 'user',
                 mhc_class = 1,
                 mhc_allele = tmp_allele,
                 pep_seq = tmp_peptide,
                 pdbfile = tmp_pdbfile,
                 chainseq = pose['chainseq'],
                 mhc_alignseq = tmp_alignseq,
                 mhc_total_chainbreak = 0.0, # doesnt really matter
            )]).set_index('pdbid', drop=False)
        # make tdifile
        tdinfo = td2.tcrdock_info.TCRdockInfo()
        tdinfo.mhc_core = tmp_mhc_core
        tdifile = tmp_pdbfile+'.tcrdock_info.json'
        with open(tdifile, 'w') as f:
            f.write(tdinfo.to_string()+'\n')
        print('make fake my_pmhc_info:')
        print(my_pmhc_info.iloc[0])


    if mhc_class == 1:
        if organism=='human':
            # now adding HLA-E 2022-05-03
            assert mhc_allele[0] in 'ABCE' and mhc_allele[1]=='*' and ':' in mhc_allele
            mhc_allele = ':'.join(mhc_allele.split(':')[:2]) # just the 4 digits
        else:
            assert mhc_allele.startswith('H2') and mhc_allele in mhc_class_1_alfas

        # first: MHC part
        trg_mhc_alseq = get_mhc_class_1_alseq(mhc_allele)
        trg_mhc_seq = trg_mhc_alseq.replace(ALL_GENES_GAP_CHAR,'')

        sortl = []

        # use new pmhc-only data
        # plus maybe some new ternary data
        for l in my_pmhc_info.itertuples():
            if (l.organism!=organism or l.mhc_class!=mhc_class or
                l.pdbid in BAD_PMHC_PDBIDS):
                continue

            tmp_mhc_alseq = l.mhc_alignseq
            assert len(trg_mhc_alseq) == len(tmp_mhc_alseq)

            mhc_idents = sum(a==b and a!=ALL_GENES_GAP_CHAR
                             for a,b in zip(trg_mhc_alseq, tmp_mhc_alseq))

            if len(peptide) == len(l.pep_seq):
                pep_idents = sum(a==b for a,b in zip(peptide,l.pep_seq))
            else:
                pep_idents = sum(a==b for a,b in zip(peptide[:3]+peptide[-3:],
                                                     l.pep_seq[:3]+l.pep_seq[-3:]))

            total = len(peptide)+len(trg_mhc_seq)
            frac = (mhc_idents+pep_idents)/total - 0.01*l.mhc_total_chainbreak
            sortl.append((frac, l.Index))

        sortl.sort(reverse=True)
        max_idents, tmp_pdbid = sortl[0]
        print(f'mhc max_idents: {max_idents:.3f}', 'target=', mhc_allele, peptide,
              'top_template=', tmp_pdbid,
              my_pmhc_info.loc[tmp_pdbid, 'mhc_allele'],
              my_pmhc_info.loc[tmp_pdbid, 'pep_seq'])

        pmhc_alignments = []
        for (idents, pdbid) in sortl[:num_templates_per_run]:
            if idents < next_best_identity_threshold_mhc*max_idents:
                break
            templatel = my_pmhc_info.loc[pdbid]
            tmp_mhc_alseq = templatel.mhc_alignseq
            tmp_seql = templatel.chainseq.split('/')[:2] # slice to allow ternary pdbs
            assert len(tmp_seql)==2 and tmp_seql[1] == templatel.pep_seq
            tmp_mhc_seq = tmp_seql[0] # class 1
            tmp_mhc_alseq_seq = tmp_mhc_alseq.replace(ALL_GENES_GAP_CHAR,'')
            assert tmp_mhc_alseq_seq in tmp_mhc_seq
            npad = tmp_mhc_seq.index(tmp_mhc_alseq_seq)
            #al1 = blosum_align(tmp_mhc_seq, tmp_mhc_alseq_seq)
            al1 = {i+npad:i for i in range(len(tmp_mhc_alseq_seq))}

            al2 = {i-tmp_mhc_alseq[:i].count(ALL_GENES_GAP_CHAR):
                   i-trg_mhc_alseq[:i].count(ALL_GENES_GAP_CHAR)
                   for i,(a,b) in enumerate(zip(tmp_mhc_alseq, trg_mhc_alseq))
                   if a != ALL_GENES_GAP_CHAR and b != ALL_GENES_GAP_CHAR}

            tmp_to_trg = {x:al2[y] for x,y in al1.items() if y in al2}
            trg_to_tmp = {y:x for x,y in tmp_to_trg.items()}

            trg_offset = len(trg_mhc_seq)
            tmp_offset = len(tmp_mhc_seq)
            trg_peplen, tmp_peplen = len(peptide), len(templatel.pep_seq)
            if trg_peplen == tmp_peplen:
                for i in range(trg_peplen):
                    trg_to_tmp[trg_offset+i] = tmp_offset+i
            else:
                for i in range(3):
                    trg_to_tmp[trg_offset+i] = tmp_offset+i
                for i in [-3,-2,-1]:
                    trg_to_tmp[trg_offset+trg_peplen+i] = tmp_offset+tmp_peplen+i
            trg_pmhc_seq = trg_mhc_seq + peptide
            tmp_pmhc_seq = tmp_mhc_seq + templatel.pep_seq
            identities = sum(trg_pmhc_seq[i] == tmp_pmhc_seq[j]
                             for i,j in trg_to_tmp.items())/len(trg_pmhc_seq)
            identities_for_sorting = (identities - 0.01*templatel.mhc_total_chainbreak)
            assert abs(identities_for_sorting-idents)<1e-3

            if verbose:
                print(f'oldnew mhc_idents: {idents:6.3f} {identities:6.3f} {pdbid}')
            show_alignment(trg_to_tmp, trg_pmhc_seq, tmp_pmhc_seq)
            pmhc_alignments.append((identities_for_sorting,
                                    pdbid,
                                    trg_to_tmp,
                                    trg_pmhc_seq,
                                    tmp_pmhc_seq,
                                    identities,
            ))
    else: # class II
        #assert force_pmhc_pdbids is None # not implemented here yet...
        #
        trg_mhca_alseq = get_mhc_class_2_alseq('A', mhc_allele.split(',')[0])
        trg_mhcb_alseq = get_mhc_class_2_alseq('B', mhc_allele.split(',')[1])
        trg_mhca_seq = trg_mhca_alseq.replace(ALL_GENES_GAP_CHAR,'')
        trg_mhcb_seq = trg_mhcb_alseq.replace(ALL_GENES_GAP_CHAR,'')
        trg_pmhc_seq = trg_mhca_seq + trg_mhcb_seq + peptide

        sortl = []
        for l in my_ternary_info.itertuples():
            if (l.organism!=organism or l.mhc_class!=mhc_class or
                l.pdbid in BAD_PMHC_PDBIDS or l.pdbid in exclude_pdbids):
                continue
            mismatches_for_excluding = min(
                count_peptide_mismatches(x, l.pep_seq)
                for x in [peptide]+alt_self_peptides)

            if mismatches_for_excluding < min_pmhc_peptide_mismatches:
                if verbose:
                    print('peptide too close:', peptide, l.pep_seq,
                          mismatches_for_excluding, alt_self_peptides)
                continue
            tmp_mhca_alseq, tmp_mhcb_alseq = l.mhc_alignseq.split('/')
            idents = 0
            for a,b in zip([trg_mhca_alseq, trg_mhcb_alseq, peptide],
                           [tmp_mhca_alseq, tmp_mhcb_alseq, l.pep_seq]):
                assert len(a) == len(b)
                idents += sum(x==y for x,y in zip(a,b) if x!=ALL_GENES_GAP_CHAR)
            sortl.append((idents/len(trg_pmhc_seq), l.pdbid))
        sortl.sort(reverse=True)
        max_idents = sortl[0][0]
        print(f'mhc max_idents: {max_idents:.3f}', 'target=', mhc_allele, peptide,
              'top_template=', sortl[0][1],
              my_ternary_info.loc[sortl[0][1], 'mhc_allele'],
              my_ternary_info.loc[sortl[0][1], 'pep_seq'])

        pmhc_alignments = []
        for (idents, pdbid) in sortl[:num_templates_per_run]:
            if idents < next_best_identity_threshold_tcr*max_idents:
                break
            templatel = my_ternary_info.loc[pdbid]
            tmp_mhca_alseq, tmp_mhcb_alseq = templatel.mhc_alignseq.split('/')
            mhca_part = tmp_mhca_alseq.replace(ALL_GENES_GAP_CHAR,'')
            mhcb_part = tmp_mhcb_alseq.replace(ALL_GENES_GAP_CHAR,'')
            tmp_mhca_seq, tmp_mhcb_seq = templatel.chainseq.split('/')[:2]
            assert mhca_part in tmp_mhca_seq and mhcb_part in tmp_mhcb_seq
            mhca_npad = tmp_mhca_seq.find(mhca_part)
            mhcb_npad = tmp_mhcb_seq.find(mhcb_part)

            trg_offset, tmp_offset = 0, mhca_npad
            al1 = {i-trg_mhca_alseq[:i].count(ALL_GENES_GAP_CHAR)+trg_offset:
                   i-tmp_mhca_alseq[:i].count(ALL_GENES_GAP_CHAR)+tmp_offset
                   for i,(a,b) in enumerate(zip(trg_mhca_alseq, tmp_mhca_alseq))
                   if a != ALL_GENES_GAP_CHAR and b != ALL_GENES_GAP_CHAR}
            trg_offset = len(trg_mhca_seq)
            tmp_offset = len(tmp_mhca_seq)+mhcb_npad
            al2 = {i-trg_mhcb_alseq[:i].count(ALL_GENES_GAP_CHAR)+trg_offset:
                   i-tmp_mhcb_alseq[:i].count(ALL_GENES_GAP_CHAR)+tmp_offset
                   for i,(a,b) in enumerate(zip(trg_mhcb_alseq, tmp_mhcb_alseq))
                   if a != ALL_GENES_GAP_CHAR and b != ALL_GENES_GAP_CHAR}
            trg_offset = len(trg_mhca_seq)+len(trg_mhcb_seq)
            tmp_offset = len(tmp_mhca_seq)+len(tmp_mhcb_seq)
            al3 = {i+trg_offset:i+tmp_offset for i in range(CLASS2_PEPLEN)}
            trg_to_tmp = {**al1, **al2, **al3}
            tmp_pmhc_seq = tmp_mhca_seq + tmp_mhcb_seq + templatel.pep_seq
            idents_redo = sum(trg_pmhc_seq[i] == tmp_pmhc_seq[j]
                              for i,j in trg_to_tmp.items())/len(trg_pmhc_seq)
            #print(f'oldnew mhc_idents: {idents:6.3f} {idents_redo:6.3f} {pdbid}')
            assert abs(idents-idents_redo)<1e-4
            show_alignment(trg_to_tmp, trg_pmhc_seq, tmp_pmhc_seq)
            pmhc_alignments.append((idents,
                                    pdbid,
                                    trg_to_tmp,
                                    trg_pmhc_seq,
                                    tmp_pmhc_seq,
                                    idents,
            ))


    # setup a tdinfo object for the target sequence
    mhc_seq = trg_mhc_seq if mhc_class==1 else (trg_mhca_seq, trg_mhcb_seq)
    mhc_core_positions_0x = design_stats.get_mhc_core_positions_0x_cached(
        mhc_class, mhc_seq, verbose=False)
    trg_pmhc_seq = pmhc_alignments[0][3]
    assert trg_pmhc_seq.endswith(peptide)
    trg_pmhc_len = len(trg_pmhc_seq)
    ig_align_rev = {j:i for i,j in ig_align.items()} # now from tmp to trg
    trg_tcr_core = [ig_align_rev[x]+trg_pmhc_len for x in ig_tdinfo.tcr_core]
    trg_tcr_cdrs = [[ig_align_rev[x]+trg_pmhc_len, ig_align_rev[y]+trg_pmhc_len]
                    for x,y in ig_tdinfo.tcr_cdrs]

    trg_tdinfo = td2.tcrdock_info.TCRdockInfo().from_dict(
        dict(organism = organism,
             mhc_class = mhc_class,
             mhc_allele = mhc_allele,
             mhc_core = mhc_core_positions_0x,
             pep_seq = peptide,
             tcr = ig_tdinfo.tcr,
             tcr_core = trg_tcr_core,
             tcr_cdrs = trg_tcr_cdrs,
             valid = True,
        ))


    rep_dgeoms = td2.docking_geometry.load_opt_dgeoms(mhc_class)
    assert len(rep_dgeoms) == 4 == num_templates_per_run


    trg_pmhc_seq = pmhc_alignments[0][3] # check for consistency below
    trg_tcra_seq, trg_tcrb_seq = trg_ig_chainseq.split('/')

    # now make the template pdbs

    dfl = []
    for itmp in range(num_templates_per_run):
        pmhc_al = pmhc_alignments[itmp%len(pmhc_alignments)]
        #tcra_al = tcr_alignments['A'][itmp%len(tcr_alignments['A'])]
        #tcrb_al = tcr_alignments['B'][itmp%len(tcr_alignments['B'])]
        dgeom = rep_dgeoms[itmp]
        dgeom_row = None

        pmhc_pdbid = pmhc_al[1]
        pmhc_pdbfile = my_pmhc_info.loc[pmhc_pdbid, 'pdbfile']
        if pmhc_pdbfile[0] != '/' and pmhc_template_pdbfile is None:
            pmhc_pdbfile = str(td2.util.path_to_db) + '/' + pmhc_pdbfile
        pmhc_pose = td2.pdblite.pose_from_pdb(pmhc_pdbfile)
        with open(pmhc_pdbfile+'.tcrdock_info.json', 'r') as f:
            pmhc_tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(f.read())

        tcra_pose = copy.deepcopy(ig_pose)
        tcra_tdinfo = copy.deepcopy(ig_tdinfo)

        # these might not be true, actually!
        assert tcra_pose['sequence'][tcra_tdinfo.tcr_cdrs[3][0]]=='C'
        assert tcra_pose['sequence'][tcra_tdinfo.tcr_cdrs[7][0]]=='C'
        assert tcra_pose['sequence'][tcra_tdinfo.tcr_core[ 1]]=='C'
        assert tcra_pose['sequence'][tcra_tdinfo.tcr_core[12]]=='C'
        assert tcra_pose['sequence'][tcra_tdinfo.tcr_core[14]]=='C'
        assert tcra_pose['sequence'][tcra_tdinfo.tcr_core[25]]=='C'

        if reverse_dock:
            print('reverse_dock: align heavy chain with TCRA and light chain with TCRB')
            core_len = 13
            assert len(tcra_tdinfo.tcr_core) == 2*core_len
            fake_tdinfo = copy.deepcopy(tcra_tdinfo)
            fake_tdinfo.tcr_core = (fake_tdinfo.tcr_core[core_len:]+
                                    fake_tdinfo.tcr_core[:core_len])
            old_tcr_stub = td2.tcr_util.get_tcr_stub(tcra_pose, fake_tdinfo)
        else:
            old_tcr_stub = td2.tcr_util.get_tcr_stub(tcra_pose, tcra_tdinfo)

        new_tcr_stub = td2.docking_geometry.stub_from_docking_geometry(
            dgeom)
        # R @ old_tcr_stub['axes'].T = new_tcr_stub['axes'].T
        R = new_tcr_stub['axes'].T @ old_tcr_stub['axes']
        # R @ old_tcr_stub['origin'] + v = new_tcr_stub['origin']
        v = new_tcr_stub['origin'] - R@old_tcr_stub['origin']
        tcra_pose = apply_transform_Rx_plus_v(tcra_pose, R, v)

        #
        # copy tcr from tcra_pose into pmhc_pose
        num_pmhc_chains = mhc_class+1
        if len(pmhc_pose['chains']) > num_pmhc_chains:
            del_chains = list(range(num_pmhc_chains, len(pmhc_pose['chains'])))
            pmhc_pose = delete_chains(pmhc_pose, del_chains)
        pmhc_pose = append_chains(pmhc_pose, tcra_pose, [0,1])
        assert len(pmhc_pose['chains'])==2+num_pmhc_chains
        offset = pmhc_pose['chainbounds'][num_pmhc_chains]
        pmhc_tdinfo.tcr_core = (
            [x+offset for x in tcra_tdinfo.tcr_core])
        pmhc_tdinfo.tcr_cdrs = (
            [[x+offset,y+offset] for x,y in tcra_tdinfo.tcr_cdrs])
        assert pmhc_pose['sequence'][pmhc_tdinfo.tcr_cdrs[3][0]] == 'C'
        assert pmhc_pose['sequence'][pmhc_tdinfo.tcr_cdrs[7][0]] == 'C'

        # should be the same as new_tcr_stub!
        if reverse_dock:
            fake_tdinfo = copy.deepcopy(pmhc_tdinfo)
            fake_tdinfo.tcr_core = (fake_tdinfo.tcr_core[core_len:]+
                                    fake_tdinfo.tcr_core[:core_len])
            redo_tcr_stub = td2.tcr_util.get_tcr_stub(pmhc_pose, fake_tdinfo)
            redo_dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(
                pmhc_pose, fake_tdinfo)
        else:
            redo_tcr_stub = td2.tcr_util.get_tcr_stub(pmhc_pose, pmhc_tdinfo)
            redo_dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(
                pmhc_pose, pmhc_tdinfo)
        v_dev = norm(redo_tcr_stub['origin']-new_tcr_stub['origin'])
        M_dev = norm(new_tcr_stub['axes'] @ redo_tcr_stub['axes'].T - np.eye(3))
        if max(v_dev, M_dev)>5e-2:
            print('ERROR devs:', v_dev, M_dev)
        assert v_dev<5e-2
        assert M_dev<5e-2

        # could also recompute the dgeom
        dgeom_dev = td2.docking_geometry.compute_docking_geometries_distance_matrix(
            [dgeom], [redo_dgeom])[0,0]
        assert dgeom_dev<1e-1, f'ERROR big dgeom_dev: {dgeom_dev}'

        trg_to_tmp = dict(pmhc_al[2])
        tmp_pmhc_seq = pmhc_al[4]
        tmp_tcra_seq, tmp_tcrb_seq = tcra_pose['chainseq'].split('/')
        tmp_fullseq = tmp_pmhc_seq + tmp_tcra_seq + tmp_tcrb_seq
        assert pmhc_pose['sequence'] == tmp_fullseq

        assert len(pmhc_pose['chains']) == num_pmhc_chains+2

        assert trg_pmhc_seq == pmhc_al[3]
        #assert trg_tcra_seq == tcra_al[3]
        #assert trg_tcrb_seq == tcrb_al[3]

        trg_fullseq = trg_pmhc_seq + trg_ig_seq
        trg_offset = len(trg_pmhc_seq)
        tmp_offset = len(tmp_pmhc_seq)
        trg_to_tmp.update({i+trg_offset:j+tmp_offset
                           for i,j in ig_align.items()})
        assert Counter(trg_to_tmp.values()).most_common(1)[0][1]==1

        identities = sum(trg_fullseq[i]==tmp_fullseq[j]
                         for i,j in trg_to_tmp.items())
        overall_idents = identities/len(trg_fullseq)
        if pmhc_pdbid in my_pmhc_info.index: # this is just for debugging
            pmhc_allele=my_pmhc_info.loc[pmhc_pdbid, 'mhc_allele']
            pmhc_peptide=my_pmhc_info.loc[pmhc_pdbid, 'pep_seq']
        else:
            pmhc_allele=my_ternary_info.loc[pmhc_pdbid, 'mhc_allele']
            pmhc_peptide=my_ternary_info.loc[pmhc_pdbid, 'pep_seq']
        print(f'tmplt_{itmp} overall_idents: {overall_idents:.3f} {abid} {mhc_allele} '
              f'{peptide} pmhc_template: {pmhc_pdbid} {pmhc_allele} {pmhc_peptide}')
        #show_alignment(trg_to_tmp, trg_fullseq, tmp_fullseq)

        run=0
        outpdbfile = f'{outfile_prefix}_{run}_{itmp}.pdb'
        pmhc_pose = set_chainbounds_and_renumber(
            pmhc_pose, list(pmhc_pose['chainbounds'])) # new 2023-10-29
        dump_pdb(pmhc_pose, outpdbfile)
        #print('made:', outpdbfile)

        trg_pmhc_seqs = ([trg_mhc_seq, peptide] if mhc_class==1 else
                         [trg_mhca_seq, trg_mhcb_seq, peptide])
        trg_cbseq = '/'.join(trg_pmhc_seqs+[trg_tcra_seq, trg_tcrb_seq])

        alignstring = ';'.join(f'{i}:{j}' for i,j in trg_to_tmp.items())

        cys_posl = [trg_tdinfo.tcr_core[1], trg_tdinfo.tcr_core[12],
                    trg_tdinfo.tcr_core[14], trg_tdinfo.tcr_core[25],
                    trg_tdinfo.tcr_cdrs[3][0], trg_tdinfo.tcr_cdrs[7][0]]
        cys_poslstring = ''.join(trg_fullseq[x] for x in cys_posl)
        assert cys_poslstring == 'CCCCCC', \
            f'bad cysteines? {abid} {cdr3a} {cdr3b} {cys_poslstring}'

        outl = OrderedDict(
            run=run,
            template_no=itmp,
            target_chainseq=trg_cbseq,
            overall_idents=overall_idents,
            pmhc_pdbid=pmhc_pdbid,
            pmhc_idents=pmhc_al[-1],
            pmhc_allele=pmhc_allele,#my_pmhc_info.loc[pmhc_pdbid, 'mhc_allele'],
            # tcra_pdbid=tcra_pdbid,
            # tcra_idents=tcra_al[-1],
            # tcra_v   =my_tcr_info.loc[(tcra_pdbid,'A'), 'v_gene'],
            # tcra_j   =my_tcr_info.loc[(tcra_pdbid,'A'), 'j_gene'],
            # tcra_cdr3=my_tcr_info.loc[(tcra_pdbid,'A'), 'cdr3'],
            # tcrb_pdbid=tcrb_pdbid,
            # tcrb_idents=tcrb_al[-1],
            # tcrb_v   =my_tcr_info.loc[(tcrb_pdbid,'B'), 'v_gene'],
            # tcrb_j   =my_tcr_info.loc[(tcrb_pdbid,'B'), 'j_gene'],
            # tcrb_cdr3=my_tcr_info.loc[(tcrb_pdbid,'B'), 'cdr3'],
            # dgeom_pdbid=f'opt{itmp}' if dgeom_row is None else dgeom_row.pdbid,
            template_pdbfile=outpdbfile,
            target_to_template_alignstring=alignstring,
            identities=identities,
            target_len=len(trg_fullseq),
            template_len=len(tmp_fullseq),
            target_tdinfo=trg_tdinfo.to_string(),
        )
        dfl.append(outl)
    assert len(dfl) == num_templates_per_run
    return pd.DataFrame(dfl)

def setup_for_alphafold_ig(
        tcr_db,
        outdir,
        **kwargs,
):
    ''' Borrowed from tcrdock/sequtil.py, modified
    '''
    assert outdir.endswith('/')
    required_cols = 'organism mhc_class mhc peptide abid cdr3a cdr3b'.split()
    for col in required_cols:
        assert col in tcr_db.columns, f'Need {col} column in tcr_db'

    num_runs = 1
    makedirs(outdir, exist_ok=True)

    tcr_db_outfile = outdir+'tcr_db.tsv'
    tcr_db.to_csv(tcr_db_outfile, sep='\t', index=False)

    #print('check genes for modeling', tcr_db.shape[0])
    #assert check_genes_for_modeling(tcr_db)

    targets_dfl = []
    for index, targetl in tcr_db.reset_index().iterrows():
        targetid_prefix = f'T{index:05d}_{targetl.mhc}_{targetl.peptide}'
        targetid_prefix = targetid_prefix.replace('*','').replace(':','')
        print('START', index, tcr_db.shape[0], targetid_prefix)
        outfile_prefix = f'{outdir}{targetid_prefix}'

        pmhc_template_pdbfile = (targetl.pmhc_pdbfile if hasattr(targetl,'pmhc_pdbfile')
                                 else None)
        if 'reverse_dock' in tcr_db.columns:
            kwargs['reverse_dock'] = targetl.reverse_dock

        all_run_info = make_templates_for_alphafold_ig(
            targetl.organism, targetl.abid, targetl.cdr3a, targetl.cdr3b,
            targetl.mhc_class, targetl.mhc, targetl.peptide, outfile_prefix,
            pmhc_template_pdbfile = pmhc_template_pdbfile,
            **kwargs,
        )

        for run in range(num_runs):
            info = all_run_info[all_run_info.run==run]
            assert info.shape[0] == 4#num templates
            targetid = f'{targetid_prefix}_{run}'
            trg_cbseq = set(info.target_chainseq).pop()
            alignfile = f'{outdir}{targetid}_alignments.tsv'
            info.to_csv(alignfile, sep='\t', index=False)
            outl = targetl.copy()
            outl['targetid'] = targetid
            outl['target_chainseq'] = trg_cbseq
            outl['templates_alignfile'] = alignfile
            outl['target_tdinfo'] = info.target_tdinfo.iloc[0]
            targets_dfl.append(outl)

    outfile = outdir+'targets.tsv'
    targets = pd.DataFrame(targets_dfl)
    targets.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)
    return targets

######################################################################################88
######################################################################################88
######################################################################################88


# read the targets
pmhc_targets = pd.read_table(args.pmhc_targets)
required_cols = 'organism mhc_class mhc peptide'.split()
for col in required_cols:
    assert col in pmhc_targets.columns, f'Need {col} in --pmhc_targets'

if args.both_orientations:
    assert 'reverse_dock' not in pmhc_targets.columns
    df1 = pmhc_targets.copy()
    df1['reverse_dock'] = False
    df2 = pmhc_targets.copy()
    df2['reverse_dock'] = True
    pmhc_targets = pd.concat([df1,df2])
elif args.reverse_dock:
    if 'reverse_dock' not in pmhc_targets.columns:
        pmhc_targets['reverse_dock'] = args.reverse_dock
    else:
        assert all(pmhc_targets.reverse_dock == args.reverse_dock)
else:
    if 'reverse_dock' not in pmhc_targets.columns:
        pmhc_targets['reverse_dock'] = False # make explicit

assert 'reverse_dock' in pmhc_targets.columns # new

# read the templates info -- right now, use this to get the template cdr3 sequences
fname = design_paths.SABDAB_DIR+'sabdab_summary_2024-01-26_abid_info.tsv'
#fname = design_paths.SABDAB_DIR+'sabdab_summary_abid_info.tsv'
print('reading:', fname)
ab_templates = pd.read_table(fname, low_memory=False)
ab_templates.set_index('abid', inplace=True, drop=False)

known_abids = set(ab_templates.abid)
assert all(x in known_abids for x in args.abids)

# read the big paired tcr database, this provides the random cdr3a/cdr3b pairs
tcrs_file = design_paths.PAIRED_TCR_DB
#tcrs_file = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2022-01-22.tsv.top9'
print('reading:', tcrs_file)
big_tcrs_df = pd.read_table(tcrs_file, low_memory=False)

# exclude extreme len cdr3s
badmask = ((big_tcrs_df.cdr3a.str.len()<9)|
           (big_tcrs_df.cdr3b.str.len()<9)|
           (big_tcrs_df.cdr3a.str.len()>17)|
           (big_tcrs_df.cdr3b.str.len()>17))
big_tcrs_df = big_tcrs_df[~badmask]


# sample --num_designs pmhcs and cdr3a/b pairs
pmhcs = pmhc_targets.sample(n=args.num_designs, replace=True,
                            random_state=args.random_state)

cdr3s = big_tcrs_df.sample(n=args.num_designs, replace=True,
                           random_state=args.random_state)


outdir = f'{args.outfile_prefix}_tmp/'
makedirs(outdir, exist_ok=True)


dfl = []
for (_,lpmhc), lcdr3 in zip(pmhcs.iterrows(), cdr3s.itertuples()):
    assert lpmhc.mhc_class==1 # for the time being...

    outl = lpmhc.copy()
    abid = random.choice(args.abids)
    outl['abid'] = abid
    old_cdr3a = ab_templates.loc[abid, 'cdr3a']
    old_cdr3b = ab_templates.loc[abid, 'cdr3b']
    cdr3a = lcdr3.cdr3a
    cdr3b = lcdr3.cdr3b
    # preserve 1st 3 and last 2 cdr3 rsds from template tcr (set by n/cterm_seq_stem)
    cdr3a = (old_cdr3a[:nterm_seq_stem] +
             cdr3a[nterm_seq_stem:-cterm_seq_stem] +
             old_cdr3a[-cterm_seq_stem:])
    cdr3b = (old_cdr3b[:nterm_seq_stem] +
             cdr3b[nterm_seq_stem:-cterm_seq_stem] +
             old_cdr3b[-cterm_seq_stem:])
    outl['cdr3a'] = cdr3a
    outl['cdr3b'] = cdr3b
    outl['old_cdr3a'] = old_cdr3a
    outl['old_cdr3b'] = old_cdr3b

    dfl.append(outl)

tcrs = pd.DataFrame(dfl)


assert 'reverse_dock' in tcrs.columns # new boolean column

targets = setup_for_alphafold_ig(tcrs, outdir)

if args.design_cdrs:
    which_cdrs = args.design_cdrs[:]
else:
    which_cdrs = [3,7]


targets.rename(columns={'target_chainseq':'chainseq',
                        'templates_alignfile':'alignfile'}, inplace=True)
dfl = []
for l in targets.itertuples():
    posl = []
    tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(l.target_tdinfo)
    for ii in which_cdrs:
        loop = tdinfo.tcr_cdrs[ii]
        npad, cpad = (nterm_seq_stem, cterm_seq_stem) if ii in [3,7] else \
                     (0,0)
        posl.extend(range(loop[0]+npad, loop[1]+1-cpad))
    dfl.append(','.join([str(x) for x in posl]))
targets['designable_positions'] = dfl


# run alphafold
outprefix = f'{outdir}afold1'
start = timer()
targets = wrapper_tools.run_alphafold(
    targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    dry_run = args.debug,
)
af2_time = timer()-start

# compute stats; most will be over-written but this saves docking geometry info
# so at the end we will get an rmsd between the mpnn-input pose and the final
# alphafold re-docked pose
targets = design_stats.compute_simple_stats(targets, extend_flex='barf')

# run mpnn
outprefix = f'{outdir}mpnn'
start = timer()
targets = wrapper_tools.run_mpnn(
    targets,
    outprefix,
    extend_flex='barf',
    dry_run=args.debug,
)
mpnn_time = timer()-start

# run alphafold again
outprefix = f'{outdir}afold2'
start = timer()
targets = wrapper_tools.run_alphafold(
    targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    dry_run = args.debug,
    ignore_identities = True, # since mpnn changed sequence...
)
af2_time += timer()-start

# compute stats again. this should compute docking rmsd to the original mpnn dock
targets = design_stats.compute_simple_stats(targets, extend_flex='barf')

# calc peptide probs
outprefix = f'{outdir}mpnn_pprobs'
start = timer()
targets = wrapper_tools.run_mpnn_peptide_probs(
    targets, outprefix, num_mpnn_seqs=5, sampling_temp=1.0)
mpnn_time += timer()-start
targets['pepspec_delta'] = (targets.total_wt_pep_prob_full -
                            targets.total_wt_pep_prob_pmhc)


if not args.skip_rf_antibody:
    start = timer()

    outprefix = f'{outdir}_rf2'
    rf_targets = wrapper_tools.run_rf_antibody_on_designs(
        targets, outprefix, delete_old_results=True, # in case of restart for partial
        model_path=args.rf_antibody_chkpt,
        reverse_chains = args.reverse_rf_antibody_chains, # deprecated
    )
    rf2_time = timer()-start
    targets['rf2_time'] = rf2_time/args.num_designs

    for col in 'model_pdbfile rfab_pbind rfab_pmhc_tcr_pae'.split():
        targets['rf2_'+col] = rf_targets[col]

    df = design_stats.compare_models(targets, rf_targets)
    for col in 'dgeom_rmsd cdr3_rmsd cdr_rmsd'.split():
        targets['rf2_'+col] = df[col]
    targets['cdr_seq'] = df['model1_cdr_seq']

    targets['combo_score'] = (targets.pmhc_tcr_pae +
                              targets.rf2_rfab_pmhc_tcr_pae +
                              targets.rf2_cdr3_rmsd)

    targets['combo_score_wtd'] = (2.0 * targets.pmhc_tcr_pae +
                                  1.0 * targets.rf2_rfab_pmhc_tcr_pae +
                                  0.5 * targets.rf2_cdr3_rmsd)

# write results
targets['af2_time'] = af2_time/args.num_designs
targets['mpnn_time'] = mpnn_time/args.num_designs

targets = design_stats.reorient_and_rechain_models(targets)

targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')
