######################################################################################88
from collections import OrderedDict, Counter
import os
import json
import sys
from os.path import exists
from os import makedirs
#from .tcrdist.all_genes import all_genes
#from .tcrdist.amino_acids import amino_acids
#from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
#from Bio.pairwise2 import format_alignment
from .util import path_to_db
#from . import docking_geometry
#from .docking_geometry import DockingGeometry
#from .tcrdock_info import TCRdockInfo
#from . import tcrdist
#from . import superimpose
#from . import tcr_util
from . import util
from . import pdblite
from . import sequtil
from . import get_tdinfo
import pandas as pd
import numpy as np
import random
import copy
import string
import itertools as it
#from numpy.linalg import norm


def make_af3_templates_from_alignfile(
        alignfile,
        chainseq,
        #resample_dgeoms=None,
        #tdinfo=None, # for the target seq
        ignore_target_chainseq_in_alignfile = False,
):
    ''' This assumes that AF3 and TCRdock had the same chainseq

    returns all_templates a list of lists of templates

    this is borrowed/modified from scripts/shared.py

    '''
    import io

    aldf = pd.read_table(alignfile)
    assert aldf.shape[0] == 4

    tcrdock_templates = []
    for ind,l in aldf.iterrows():
        if ignore_target_chainseq_in_alignfile:
            assert len(l.target_chainseq) == len(chainseq)
        else:
            assert l.target_chainseq == chainseq
        pose = pdblite.pose_from_pdb(l.template_pdbfile)
        al = {int(x.split(':')[0]):int(x.split(':')[1])
              for x in l.target_to_template_alignstring.split(';')}

        tcrdock_templates.append((pose, al))

    cs = chainseq.split('/')
    num_chains = len(cs)
    chainbounds = [0] + list(it.accumulate(len(x) for x in cs))

    all_templates = []

    for ichain, seq in enumerate(cs):

        templates = []

        for tmpnum, (pose, align) in enumerate(tcrdock_templates):
            assert len(pose['chains']) == num_chains

            other_chains = [x for x in range(num_chains) if x != ichain]
            new_pose = copy.deepcopy(pose)
            new_pose = pdblite.delete_chains(new_pose, other_chains)

            full_template_chainbounds = [0] + list(it.accumulate(
                len(x) for x in pose['chainseq'].split('/')))
            target_nres_before = chainbounds[ichain]
            template_nres_before = full_template_chainbounds[ichain]

            # the tcrdock target --> template alignment, just for this chain
            chain_align = {x-target_nres_before : y-template_nres_before
                           for x,y in align.items()
                           if (chainbounds[ichain] <= x < chainbounds[ichain+1])}
            assert all(0 <= x < len(new_pose['sequence'])
                       for x in chain_align.values())

            out = io.StringIO()
            pdblite.dump_cif(
                new_pose, None, out=out, name=f'fake_{ichain}_{tmpnum}')

            query_inds    = [i for i,j in sorted(chain_align.items())]
            template_inds = [j for i,j in sorted(chain_align.items())]

            assert query_inds and template_inds

            template = {'mmcif': out.getvalue(),
                        'queryIndices': query_inds,
                        'templateIndices': template_inds}
            templates.append(template)
        all_templates.append(templates)

    return all_templates # list of lists

def unalign_templates_at_masked_positions(res, masked_positions):

    offset = 0

    for chain in range(len(res['sequences'])):
        info = res['sequences'][chain]['protein']
        seq = info['sequence']

        posl = [x-offset for x in masked_positions if x>=offset and x<offset+len(seq)]

        if posl:
            print('masking:', chain, posl)
            new_templates = []
            for template in info['templates']:
                qinds, tinds = template['queryIndices'], template['templateIndices']

                new_qinds = [x for x in qinds if x not in posl]
                new_tinds = [y for x,y in zip(qinds, tinds) if x not in posl]
                assert len(new_qinds) == len(new_tinds)

                new_templates.append({
                    'mmcif': template['mmcif'],
                    'queryIndices':new_qinds,
                    'templateIndices':new_tinds})
            info['templates'] = new_templates

            assert not any(x in y['queryIndices'] for x in posl
                           for y in res['sequences'][chain]['protein']['templates'])
        offset += len(seq)
    return res


def make_af3_templates_from_template_pdbfile(
        target_chainseq,
        template_pdbfile,
        is_unbound_pmhc=False,
        is_unbound_tcr=False,
):
    ''' This assumes that target_chainseq and template_pdbfile have the same number
    of chains (unless is_unbound_tcr or is_unbound_pmhc)

    allows for sequence and length differences via blosum_align

    '''
    import io

    cs = target_chainseq.split('/')
    num_chains = len(cs)
    chainbounds = [0] + list(it.accumulate(len(x) for x in cs))

    all_templates = []

    pose = pdblite.pose_from_pdb(template_pdbfile)
    old_num_chains = len(pose['chains'])
    if is_unbound_tcr: # delete pmhc from pose
        pose = pdblite.delete_chains(pose, list(range(old_num_chains-2)))
    elif is_unbound_pmhc: # delete tcr from pose
        assert old_num_chains == num_chains+2
        pose = pdblite.delete_chains(pose, list(range(num_chains, num_chains+2)))

    assert len(pose['chains']) == num_chains

    for ichain, trg_seq in enumerate(cs):

        other_chains = [x for x in range(num_chains) if x != ichain]
        new_pose = copy.deepcopy(pose)
        new_pose = pdblite.delete_chains(new_pose, other_chains)

        tmp_seq = new_pose['sequence']

        if trg_seq == tmp_seq:
            al = {i:i for i in range(len(trg_seq))}
        else:
            al = sequtil.blosum_align(trg_seq, tmp_seq)

        out = io.StringIO()
        pdblite.dump_cif(
            new_pose, None, out=out, name=f'fake_{ichain}')

        query_inds    = [i for i,j in sorted(al.items())]
        template_inds = [j for i,j in sorted(al.items())]

        assert query_inds and template_inds

        template = {'mmcif': out.getvalue(),
                    'queryIndices': query_inds,
                    'templateIndices': template_inds}

        all_templates.append([template])

    return all_templates # list of lists

def make_core_anchored_alignment(aseq, acore, bseq, bcore, verbose=True):
    ''' align aseq and bseq by forcing acore and bcore positions to align
    and inserting at most one gap between each pair of aligned core positions
    '''
    from .tcrdist.tcr_distances import align_cdr3_regions
    assert len(acore) == len(bcore)

    align = {}
    
    mygap = '-'
    for ipos, (apos, bpos) in enumerate(zip(acore+[None], bcore+[None])):
        # we align from astart to apos, inclusive (roughly)
        if ipos:
            astart, bstart = acore[ipos-1]+1, bcore[ipos-1]+1
        else:
            astart, bstart = 0,0
            
        if apos is None: # Cterm
            asuf = len(aseq)-astart # rsds after last core rsd, including astart
            bsuf = len(bseq)-bstart # rsds after last core rsd
            for ii in range(min(asuf, bsuf)):
                align[astart+ii] = bstart+ii
        elif ipos == 0: # Nterm
            for ii in range(min(apos, bpos)+1):
                align[apos-ii] = bpos-ii
        else: # somewhere in the middle
            alen = apos-astart+1
            blen = bpos-bstart+1
            if alen == 0 or blen == 0:
                assert align[apos] == bpos
            elif alen == 1 or blen == 1:
                align[apos] = bpos # have to make a choice, start or pos
            elif alen == blen:
                for ii in range(alen):
                    align[astart+ii] = bstart+ii
            else:
                agapped, bgapped = align_cdr3_regions(aseq[astart:apos+1],
                                                      bseq[bstart:bpos+1], mygap)
                for i,(a,b) in enumerate(zip(agapped, bgapped)):
                    if a!=mygap and b!=mygap:
                        ia = astart+i-agapped[:i].count(mygap)
                        ib = bstart+i-bgapped[:i].count(mygap)
                        align[ia] = ib

    if verbose:
        bshow = ''.join(bseq[align[x]] if x in align else '-'
                        for x in range(len(aseq)))
        cshow = ''.join('*' if x in acore else ' '
                        for x in range(len(aseq)))

        print('make_core_anchored_alignment:')
        print(aseq)
        print(cshow)
        print(bshow)
        print([x for x in acore if x not in align])
    
    if not all(align[x] == y for x,y in zip(acore, bcore)):
        print(aseq, acore, Counter(acore).most_common())
        print(bseq, bcore, Counter(bcore).most_common())
        print([(ii,x,y) for ii, (x,y) in enumerate(zip(acore, bcore)) if align[x] != y])
            
    assert all(align[x] == y for x,y in zip(acore, bcore))
    return align


_cached_abinfo = None

def make_abid_msas(abid, aseq, bseq): #cdr3a, bseq, cdr3b):
    ''' returns all_msas which is a list containing two strings

    assumes that aseq and bseq differ from the respective chain sequences for abid
    by mutations anywhere and an indel only in cdr3
    
    the first string is the full msa (with newline characters) for the light chain,
    the second string is the full msa for the heavy chain

    note that the light and heavy msas are not aligned, ie they do not
    preserve heavy-light pairing info in the templates
    '''
    global _cached_abinfo
    if _cached_abinfo is None:
        fname = util.path_to_db / 'sabdab_summary_2024-01-26_abid_info_w_tdinfo.tsv'
        abinfo = pd.read_table(fname).set_index('abid', drop=False)
        cols = 'va ja cdr3a vb jb cdr3b'.split()
        nr_abinfo = abinfo.sort_values('resolution').drop_duplicates(cols)
        #nr_abinfo = nr_abinfo[nr_abinfo.abid == '8ahnHL'].copy()
        _cached_abinfo = abinfo, nr_abinfo
    abinfo, nr_abinfo = _cached_abinfo

    trg_row = abinfo.loc[abid]

    all_msas = []

    for ii,ab in enumerate('ab'):
        # compute seqid to all abids in nr_abinfo
        old_seq = trg_row['chainseq_'+ab]
        new_seq = aseq if ab=='a' else bseq
        old_cdr3 = trg_row['cdr3'+ab]
        assert old_cdr3 in old_seq
        start = old_seq.index(old_cdr3)
        #assert new_seq[:start] == old_seq[:start]
        suflen = len(old_seq)-start-len(old_cdr3)
        assert new_seq[-suflen:] == old_seq[-suflen:]

        new_cdr3 = new_seq[start:-suflen]
        print('new_cdr3:', new_cdr3)
        old_stop = start+len(old_cdr3)
        new_stop = start+len(new_cdr3)
        
        old_alcore = [int(x) for x in trg_row['alcore_'+ab].split(',')]
        cdr3_start_anchor = start+1
        assert all(x<=cdr3_start_anchor or x>=old_stop for x in old_alcore)
        shift = new_stop-old_stop
        new_alcore = [x if x<=cdr3_start_anchor else x+shift for x in old_alcore]
        sortl = []
        for _,tmp_row in nr_abinfo.iterrows():
            tmp_seq = tmp_row['chainseq_'+ab]
            tmp_alcore = [int(x) for x in tmp_row['alcore_'+ab].split(',')]
            align = make_core_anchored_alignment(
                new_seq, new_alcore, tmp_seq, tmp_alcore, verbose=False)
            idents = sum(new_seq[i] == tmp_seq[j] for i,j in align.items())
            sortl.append((idents, tmp_seq, tmp_row.abid, align))
        sortl.sort(reverse=True)

        msa = [f'>query\n{new_seq}\n']
        for idents, tmp_seq, tmp_abid, align in sortl:
            alseq = make_target_anchored_alseq(align, new_seq, tmp_seq)
            msa.append(f'>{tmp_abid}\n{alseq}\n')
        all_msas.append(''.join(msa))
    return all_msas

    

def align_tcr_chainseqs(
        org0, v0, j0, cdr30,
        org1, v1, j1, cdr31,
):
    ''' returns align, seq0, seq1
    '''
    from .sequtil import align_vgene_to_structure_msas, get_v_seq_up_to_cys, \
        get_j_seq_after_cdr3, align_cdr3s
    from .tcrdist.tcr_distances import align_cdr3_regions

    msa_type = 'human' if org0 == 'human' and org1 == 'human' else 'both'

    v_align0 = align_vgene_to_structure_msas(org0, v0)[msa_type]

    v_align1 = align_vgene_to_structure_msas(org1, v1)[msa_type]

    v0_seq = get_v_seq_up_to_cys(org0, v0)
    j0_seq = get_j_seq_after_cdr3(org0, j0)
    v1_seq = get_v_seq_up_to_cys(org1, v1)
    j1_seq = get_j_seq_after_cdr3(org1, j1)

    assert v0_seq[-1] == 'C' and v1_seq[-1] == 'C', \
        f'bad v seqs: {v0} {v0_seq[-1]} {v1} {v1_seq[-1]}'

    # allow variable gap positions...
    assert len(cdr30) >= 5 and len(cdr31) >= 5
    minlen = min(len(cdr30), len(cdr31))
    npad = min(4, minlen-minlen//2)
    cpad = min(4, minlen//2)
    assert npad + cpad <= minlen
    assert npad >= 3
    assert cpad >= 2
    mygap = '-'
    a,b = align_cdr3_regions(cdr30[npad:-cpad], cdr31[npad:-cpad], mygap)
    a = cdr30[:npad]+a+cdr30[-cpad:]
    b = cdr31[:npad]+b+cdr31[-cpad:]
    assert len(a) == len(b)

    #cdr3_align = align_cdr3s(cdr30, cdr31)
    cdr3_align = {i-a[:i].count(mygap) : i-b[:i].count(mygap)
                  for i,(x,y) in enumerate(zip(a,b))
                  if x != mygap and y != mygap}
    #print(cdr3_align)

    j_align = {i:i for i in range(min(len(j0_seq), len(j1_seq)))}

    v_revalign1 = {j:i for i,j in v_align1.items()}

    align = {}
    for i,j in v_align0.items():
        if j in v_revalign1:
            align[i] = v_revalign1[j]

    for i,j in cdr3_align.items():
        align[i+len(v0_seq)-1] = j+len(v1_seq)-1 # -1 to drop the 'C'

    for i,j in j_align.items():
        align[i+len(v0_seq)-1+len(cdr30)] = j+len(v1_seq)-1+len(cdr31)

    return align, v0_seq[:-1] + cdr30 + j0_seq, v1_seq[:-1] + cdr31 + j1_seq



_deletion_table = str.maketrans('', '', string.ascii_lowercase) # sanity
def make_target_anchored_alseq(align, seq0, seq1, ADD_INSERTS=True):
    jprev = None
    alseq = ''
    for i in range(len(seq0)):
        if i in align:
            j = align[i]
            if jprev is not None:
                assert j>jprev
                if j != jprev+1 and ADD_INSERTS:
                    alseq += seq1[jprev+1:j].lower()
            alseq += seq1[j]
            jprev = j
        else:
            alseq += '-'

    if ADD_INSERTS:
        nogaps = alseq.translate(_deletion_table)
        assert len(nogaps) == len(seq0)
    else:
        assert len(alseq) == len(seq0)

    return alseq



def align_mhc_class_1_chainseqs(org0, mhc0, org1, mhc1):
    ''' returns align, seq0, seq1
    '''
    from .sequtil import get_mhc_class_1_alseq

    GAP = sequtil.ALL_GENES_GAP_CHAR

    # guarantee consistency with sequtil.make_pmhc_chainseq_for_alphafold
    mhc0 = sequtil.trim_and_validate_class1_mhc(org0, mhc0)
    mhc1 = sequtil.trim_and_validate_class1_mhc(org1, mhc1)

    alseq0 = get_mhc_class_1_alseq(mhc0)
    alseq1 = get_mhc_class_1_alseq(mhc1)

    assert len(alseq0) in [175, 181]
    assert len(alseq1) in [175, 181]

    alseq0 = alseq0 + GAP*(181-len(alseq0))
    alseq1 = alseq1 + GAP*(181-len(alseq1))


    align = {i-alseq0[:i].count(GAP) : i-alseq1[:i].count(GAP)
             for i,(x,y) in enumerate(zip(alseq0,alseq1))
             if x != GAP and y != GAP}

    return align, alseq0.replace(GAP,''), alseq1.replace(GAP,'')



def align_mhc_class_2_chainseqs(org0, mhc0, org1, mhc1):
    ''' returns mhc_alseqs, mhc_idents

    mhc_alseqs is a list of org1-mhc sequences that can go right into AF3 MSA
    '''
    from .sequtil import get_mhc_class_2_alseq

    GAP = sequtil.ALL_GENES_GAP_CHAR

    mhc_alseqs, mhc_idents = [], []
    for ab, mhc0part, mhc1part in zip('AB', mhc0.split(','), mhc1.split(',')):

        alseq0 = get_mhc_class_2_alseq(ab, mhc0part)
        alseq1 = get_mhc_class_2_alseq(ab, mhc1part)

        assert len(alseq0) == len(alseq1)

        align = {i-alseq0[:i].count(GAP) : i-alseq1[:i].count(GAP)
                 for i,(x,y) in enumerate(zip(alseq0,alseq1))
                 if x != GAP and y != GAP}

        mhc_idents.append(sum((x==y and x != GAP) for x,y in zip(alseq0, alseq1)))

        mhc_alseqs.append(make_target_anchored_alseq(
            align, alseq0.replace(GAP,''), alseq1.replace(GAP,'')))

    return mhc_alseqs, mhc_idents



def align_peptides(pep0, pep1):
    'returns align'
    from .tcrdist.tcr_distances import align_cdr3_regions
    assert len(pep0) >= 6 and len(pep1) >= 6
    npad, cpad = 2,2
    mygap = '-'
    # align_cdr3_regions doesnt allow a gap at the beginning, or the end
    # also, it uses blosum matrix so can't handle non-canonicals
    a,b = align_cdr3_regions(
        util.map_sequence_with_mods(pep0[npad:-cpad]),
        util.map_sequence_with_mods(pep1[npad:-cpad]), mygap)
    assert mygap not in (a[0]+a[-1]+b[0]+b[-1])
    a = pep0[:npad]+a+pep0[-cpad:]
    b = pep1[:npad]+b+pep1[-cpad:]
    assert len(a) == len(b)

    pep_align = {i-a[:i].count(mygap) : i-b[:i].count(mygap)
                 for i,(x,y) in enumerate(zip(a,b))
                 if x != mygap and y != mygap}
    #print(pep_align)
    return pep_align





def make_af3_msas_for_class_1_row(row0, debug=False):

    required_cols = 'organism mhc chainseq peptide va ja cdr3a vb jb cdr3b'.split()
    assert all(hasattr(row0,x) for x in required_cols)

    is_unbound_pmhc = row0.va is None or pd.isna(row0.vb)
    if is_unbound_pmhc:
        assert row0.vb is None or pd.isna(row0.vb)

    df = pd.read_table(
        path_to_db / 'run669_run670_comp_pdb_dgeom_rmsds_dgeom_zscores.tsv')

    if debug:
        df = df.sample(n=50).reset_index()

    cs0 = row0.chainseq.split('/')

    sortl = []

    for ind, row1 in df.iterrows():
        cs1 = row1.chainseq.split('/')

        mhc_align, seq0, seq1 = align_mhc_class_1_chainseqs(
            row0.organism, row0.mhc, row1.organism, row1.mhc,
        )
        assert cs0[0] == seq0
        assert cs1[0] == seq1
        mhc_idents = sum(seq0[i] == seq1[j] for i,j in mhc_align.items())
        mhc_alseq = make_target_anchored_alseq(mhc_align, seq0, seq1)

        pep_align = align_peptides(row0.peptide, row1.peptide)
        pep_idents = sum(row0.peptide[i] == row1.peptide[j]
                         for i,j in pep_align.items())
        pep_alseq = make_target_anchored_alseq(pep_align, row0.peptide, row1.peptide)


        if is_unbound_pmhc:
            tcra_idents, tcrb_idents = 0, 0
            tcra_alseq, tcrb_alseq = '', ''
        else:
            tcra_align, seq0, seq1 = align_tcr_chainseqs(
                row0.organism, row0.va, row0.ja, row0.cdr3a,
                row1.organism, row1.va, row1.ja, row1.cdr3a,
            )
            if seq0 != cs0[-2]:
                assert len(seq0) == len(cs0[-2])
                seq0 = cs0[-2]
            assert seq1 == cs1[-2]
            tcra_idents = sum(seq0[i] == seq1[j] for i,j in tcra_align.items())
            tcra_alseq = make_target_anchored_alseq(tcra_align, seq0, seq1)

            tcrb_align, seq0, seq1 = align_tcr_chainseqs(
                row0.organism, row0.vb, row0.jb, row0.cdr3b,
                row1.organism, row1.vb, row1.jb, row1.cdr3b,
            )
            if seq0 != cs0[-1]:
                assert len(seq0) == len(cs0[-1])
                seq0 = cs0[-1]
            assert seq1 == cs1[-1]
            tcrb_idents = sum(seq0[i] == seq1[j] for i,j in tcrb_align.items())
            tcrb_alseq = make_target_anchored_alseq(tcrb_align, seq0, seq1)

        wtd_idents = mhc_idents + 4*pep_idents + tcra_idents + tcrb_idents
        wtd_pid = wtd_idents/(sum(len(x) for x in cs0) + 3*len(row0.peptide))

        sortl.append((wtd_pid, mhc_alseq, pep_alseq, tcra_alseq, tcrb_alseq, ind))

        if ind%25==0:
            print(f'{wtd_pid:.3f}', ind, row1.organism, row1.mhc, row1.peptide,
                  row1.va, row1.ja, row1.cdr3a, row1.vb, row1.jb, row1.cdr3b)


    sortl.sort(reverse=True)

    if is_unbound_pmhc:
        # drop dups
        seen = set()
        new_sortl = []
        for l in sortl:
            alseqs = (l[1], l[2])
            if alseqs not in seen:
                seen.add(alseqs)
                new_sortl.append(l)

        print('make_af3_msas_for_class_1_row:: is_unbound_pmhc: drop dups:',
              len(sortl), '-->', len(new_sortl))
        sortl = new_sortl


    all_msas = []

    inds = [x[-1] for x in sortl]

    tags = 'mhc peptide'.split() if is_unbound_pmhc else 'mhc peptide tcra tcrb'.split()
    for ii,tag in enumerate(tags):
        seq0 = cs0[ii]
        seqs = [seq0] + [x[ii+1] for x in sortl]
        ids = ['>query'] + [f'>distil_{tag}_{x}' for x in inds]
        msa = ''.join(f'{x}\n{y}\n' for x,y in zip(ids,seqs))
        all_msas.append(msa)

    return all_msas


def make_af3_msas_for_class_2_row(row0, debug=False):

    required_cols = 'organism mhc chainseq peptide va ja cdr3a vb jb cdr3b'.split()
    assert all(hasattr(row0,x) for x in required_cols)

    is_unbound_pmhc = row0.va is None or pd.isna(row0.vb)
    if is_unbound_pmhc:
        assert row0.vb is None or pd.isna(row0.vb)

    df = pd.read_table(path_to_db / 'test_class2_align_info.tsv')

    if debug:
        df = df.sample(n=50).reset_index()

    cs0 = row0.chainseq.split('/')

    sortl = []
    for ind, row1 in df.iterrows():

        if pd.isna(row1.peptide):
            mhc_alseqs = ['-'*len(cs0[0]), '-'*len(cs0[1])]
            mhc_idents = [0,0]

            pep_alseq = '-'*len(row0.peptide)
            pep_idents = 0

        else:
            mhc_alseqs, mhc_idents = align_mhc_class_2_chainseqs(
                row0.organism, row0.mhc, row1.organism, row1.mhc)


            assert len(row0.peptide) == len(row1.peptide) == 11
            pep_idents = sum(row0.peptide[i] == row1.peptide[i]
                             for i in range(len(row0.peptide)))
            pep_alseq = row1.peptide # no indels

        if is_unbound_pmhc:
            tcra_idents, tcrb_idents = 0, 0
            tcra_alseq, tcrb_alseq = '', ''
        else:
            tcra_align, seq0, seq1 = align_tcr_chainseqs(
                row0.organism, row0.va, row0.ja, row0.cdr3a,
                row1.organism, row1.va, row1.ja, row1.cdr3a,
            )
            if seq0 != cs0[-2]:
                assert len(seq0) == len(cs0[-2])
                seq0 = cs0[-2]
            tcra_idents = sum(seq0[i] == seq1[j] for i,j in tcra_align.items())
            tcra_alseq = make_target_anchored_alseq(tcra_align, seq0, seq1)

            tcrb_align, seq0, seq1 = align_tcr_chainseqs(
                row0.organism, row0.vb, row0.jb, row0.cdr3b,
                row1.organism, row1.vb, row1.jb, row1.cdr3b,
            )
            if seq0 != cs0[-1]:
                assert len(seq0) == len(cs0[-1])
                seq0 = cs0[-1]
            tcrb_idents = sum(seq0[i] == seq1[j] for i,j in tcrb_align.items())
            tcrb_alseq = make_target_anchored_alseq(tcrb_align, seq0, seq1)

        wtd_idents = sum(mhc_idents) + 4*pep_idents + tcra_idents + tcrb_idents
        wtd_pid = wtd_idents/(sum(len(x) for x in cs0) + 3*len(row0.peptide))

        sortl.append((wtd_pid, *mhc_alseqs, pep_alseq, tcra_alseq, tcrb_alseq, ind))

        if ind%25==0:
            print(f'{wtd_pid:.3f}', ind, row1.organism, row1.mhc, row1.peptide,
                  row1.va, row1.ja, row1.cdr3a, row1.vb, row1.jb, row1.cdr3b)


    sortl.sort(reverse=True)

    if is_unbound_pmhc:
        # drop dups
        seen = set()
        new_sortl = []
        for l in sortl:
            alseqs = (l[1], l[2], l[3])
            if alseqs not in seen:
                seen.add(alseqs)
                new_sortl.append(l)

        print('make_af3_msas_for_class_2_row:: is_unbound_pmhc: drop dups:',
              len(sortl), '-->', len(new_sortl))
        sortl = new_sortl

    all_msas = []

    inds = [x[-1] for x in sortl]

    tags = 'mhca mhcb peptide'.split() if is_unbound_pmhc else \
           'mhca mhcb peptide tcra tcrb'.split()

    for ii,tag in enumerate(tags):
        seq0 = cs0[ii]
        seqs = [util.map_sequence_with_mods(seq0)] + [x[ii+1] for x in sortl]
        ids = ['>query'] + [f'>distil_{tag}_{x}' for x in inds]
        msa = ''.join(f'{x}\n{y}\n' for x,y in zip(ids,seqs))
        all_msas.append(msa)

    return all_msas



def make_af3_msas_for_unbound_tcr(row0, debug=False):

    required_cols = 'organism chainseq va ja cdr3a vb jb cdr3b'.split()
    assert all(hasattr(row0,x) for x in required_cols)

    df = pd.read_table(
        path_to_db / 'run669_run670_comp_pdb_dgeom_rmsds_dgeom_zscores.tsv')

    if debug:
        df = df.sample(n=50).reset_index()

    cs0 = row0.chainseq.split('/')

    sortl = []

    for ind, row1 in df.iterrows():
        cs1 = row1.chainseq.split('/')

        tcra_align, seq0, seq1 = align_tcr_chainseqs(
            row0.organism, row0.va, row0.ja, row0.cdr3a,
            row1.organism, row1.va, row1.ja, row1.cdr3a,
        )
        if seq0 != cs0[-2]:
            assert len(seq0) == len(cs0[-2])
            seq0 = cs0[-2]
        assert seq1 == cs1[-2]
        tcra_idents = sum(seq0[i] == seq1[j] for i,j in tcra_align.items())
        tcra_alseq = make_target_anchored_alseq(tcra_align, seq0, seq1)

        tcrb_align, seq0, seq1 = align_tcr_chainseqs(
            row0.organism, row0.vb, row0.jb, row0.cdr3b,
            row1.organism, row1.vb, row1.jb, row1.cdr3b,
        )
        if seq0 != cs0[-1]:
            assert len(seq0) == len(cs0[-1])
            seq0 = cs0[-1]
        assert seq1 == cs1[-1]
        tcrb_idents = sum(seq0[i] == seq1[j] for i,j in tcrb_align.items())
        tcrb_alseq = make_target_anchored_alseq(tcrb_align, seq0, seq1)

        wtd_idents = tcra_idents + tcrb_idents
        wtd_pid = wtd_idents/(sum(len(x) for x in cs0))

        sortl.append((wtd_pid, tcra_alseq, tcrb_alseq, ind))

        if ind%25==0:
            print(f'{wtd_pid:.3f}', ind, row1.organism,
                  row1.va, row1.ja, row1.cdr3a, row1.vb, row1.jb, row1.cdr3b)


    sortl.sort(reverse=True)

    all_msas = []

    inds = [x[-1] for x in sortl]

    for ii,tag in enumerate('tcra tcrb'.split()):
        seq0 = cs0[ii]
        seqs = [seq0] + [x[ii+1] for x in sortl]
        ids = ['>query'] + [f'>distil_{tag}_{x}' for x in inds]
        msa = ''.join(f'{x}\n{y}\n' for x,y in zip(ids,seqs))
        all_msas.append(msa)

    return all_msas


def make_af3_msas_for_class_1_tcrm_row(row0, debug=False):
    '''
    '''

    required_cols = 'organism mhc chainseq peptide abid'.split()
    assert all(hasattr(row0,x) for x in required_cols)

    # is_unbound_pmhc = row0.va is None or pd.isna(row0.vb)
    # if is_unbound_pmhc:
    #     assert row0.vb is None or pd.isna(row0.vb)

    df = pd.read_table(
        path_to_db / 'run669_run670_comp_pdb_dgeom_rmsds_dgeom_zscores.tsv')

    df.drop_duplicates('mhc peptide'.split(), inplace=True)

    if debug:
        df = df.sample(n=50).reset_index()

    cs0 = row0.chainseq.split('/')
    assert len(cs0) == 4 # mhc,pep,igl,igh

    sortl = []

    for ind, row1 in df.iterrows():
        cs1 = row1.chainseq.split('/')

        mhc_align, seq0, seq1 = align_mhc_class_1_chainseqs(
            row0.organism, row0.mhc, row1.organism, row1.mhc,
        )
        assert cs0[0] == seq0
        assert cs1[0] == seq1
        mhc_idents = sum(seq0[i] == seq1[j] for i,j in mhc_align.items())
        mhc_alseq = make_target_anchored_alseq(mhc_align, seq0, seq1)

        pep_align = align_peptides(row0.peptide, row1.peptide)
        pep_idents = sum(row0.peptide[i] == row1.peptide[j]
                         for i,j in pep_align.items())
        pep_alseq = make_target_anchored_alseq(pep_align, row0.peptide, row1.peptide)

        # tcra_idents, tcrb_idents = 0, 0
        # tcra_alseq, tcrb_alseq = '', ''

        wtd_idents = mhc_idents + 4*pep_idents
        wtd_pid = wtd_idents/(len(cs0[0]) + 4*len(row0.peptide))

        sortl.append((wtd_pid, mhc_alseq, pep_alseq, ind))

        if ind%25==0:
            print(f'{wtd_pid:.3f}', ind, row1.organism, row1.mhc, row1.peptide)


    sortl.sort(reverse=True)

    if True: #is_unbound_pmhc:
        # drop dups
        seen = set()
        new_sortl = []
        for l in sortl:
            alseqs = (l[1], l[2])
            if alseqs not in seen:
                seen.add(alseqs)
                new_sortl.append(l)

        print('make_af3_msas_for_class_1_tcrm_row:: drop pmhc dups:',
              len(sortl), '-->', len(new_sortl))
        sortl = new_sortl


    all_msas = []

    inds = [x[-1] for x in sortl]

    tags = 'mhc peptide'.split()
    for ii,tag in enumerate(tags):
        seq0 = cs0[ii]
        seqs = [seq0] + [x[ii+1] for x in sortl]
        ids = ['>query'] + [f'>distil_{tag}_{x}' for x in inds]
        msa = ''.join(f'{x}\n{y}\n' for x,y in zip(ids,seqs))
        all_msas.append(msa)

    ab_msas = make_abid_msas(row0.abid, cs0[-2], cs0[-1])
    all_msas.extend(ab_msas)

    return all_msas





def sanitised_name(s):
    """Returns sanitised version of the name that can be used as a filename.
    PHB borrowed from AF3 folding_input.py"""
    lower_spaceless_name = s.lower().replace(' ', '_')
    allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
    return ''.join(l for l in lower_spaceless_name if l in allowed_chars)

def setup_for_alphafold3(
        targets,
        outdir,
        num_seeds = 1,
        setup_for_af2_kwargs = {},
        debug = False,
):
    ''' Now this can handle va_mutations, vb_mutations
    '''
    required_cols = 'organism mhc_class mhc peptide va ja cdr3a vb jb cdr3b'.split()

    for col in required_cols:
        assert col in targets.columns, f'Need {col} in targets'

    targets['mhc_class'] = targets.mhc_class.astype(int)

    dropcols = [x for x in 'chainseq alignfile'.split() if x in targets.columns]
    if dropcols:
        targets.drop(columns=dropcols, inplace=True)

    # these need to be unique
    if 'targetid' not in targets.columns:
        targets['targetid'] = [f'T{ii:05d}_{x.organism}_{x.mhc}_{x.peptide}'
                               for ii,x in enumerate(targets.itertuples())]

    targets['targetid'] = targets.targetid.map(sanitised_name)

    if targets.targetid.value_counts().max() > 1:
        targets['targetid'] = [f'{x}_{i}' for i,x in enumerate(targets.targetid)]


    af2_outdir = outdir+'af2/'
    makedirs(af2_outdir, exist_ok=True)

    targets = sequtil.setup_for_alphafold(
        targets, af2_outdir, num_runs=1, use_opt_dgeoms=True, clobber=True,
        use_new_templates=True,
        **setup_for_af2_kwargs,
    )

    targets.rename(columns={'target_chainseq':'chainseq',
                            'templates_alignfile':'alignfile'}, inplace=True)

    ## now apply any v mutations
    # uses va_mutations and vb_mutations columns, if present
    targets = sequtil.update_chainseqs_from_v_mutations(targets)
    
    ## now we need to make a JSON file
    ## MSA templates, etc

    jsonfiles = []

    
    for _,row in targets.iterrows():
        v_mutations_present = 'va_mutations' in targets.columns
        all_templates = make_af3_templates_from_alignfile(
            row.alignfile, row.chainseq,
            ignore_target_chainseq_in_alignfile = v_mutations_present,
        )

        # msas
        if row.mhc_class==1:
            all_msas = make_af3_msas_for_class_1_row(row, debug=debug)
        else:
            all_msas = make_af3_msas_for_class_2_row(row, debug=debug)


        assert len(all_templates) == len(all_msas) == 3+row.mhc_class #

        res = dict(
            dialect= 'alphafold3',
            version= 1,
            name= row.targetid,
            sequences= [],
            modelSeeds= [random.randint(0, 2**32 - 1) for _ in range(num_seeds)],
            bondedAtomPairs= None,
            userCCD= None,
        )

        cs = row.chainseq.split('/')
        for ii, (id, seq, templates, msas) in enumerate(zip(
                'ABCDE', cs, all_templates, all_msas)):
            info = dict(
                id= id,
                sequence= util.map_sequence_with_mods(seq),
                modifications= [],
                unpairedMsa= msas,
                pairedMsa= '',
                templates= templates,
            )

            for jj, aa in enumerate(seq):
                if aa in util.mod_ccd_map:
                    print('add modification:', ii, jj, aa, util.mod_ccd_map[aa])
                    info['modifications'].append({'ptmType': util.mod_ccd_map[aa],
                                                  'ptmPosition': jj+1})

            res['sequences'].append({'protein': info})
        outfile = f'{outdir}{row.targetid}_data.json'
        with open(outfile,'w') as out:
            json.dump(res, out)
        print('made:', outfile)

        jsonfiles.append(outfile)

    targets['af3_input_jsonfile'] = jsonfiles

    return targets

def simple_setup_for_alphafold3(
        targets,
        outdir,
        num_seeds = 1,
        debug = False,
):
    ''' Sets templates to [] in the jsonfile if template_pdbfile not in targets

    saves location of the jsonfile in the column 'af3_input_jsonfile'

    fills in modelSeeds with num_seeds random numbers

    does handle v gene mutations

    this is used, for example, in design/dock_spectest3.py where we might have
    a template pdbfile to work from

    '''
    required_cols = 'organism mhc_class mhc peptide va ja cdr3a vb jb cdr3b'.split()
    # and maybe also 'template_pdbfile'
    for col in required_cols:
        assert col in targets.columns, f'Need {col} in targets'

    makedirs(outdir, exist_ok=True)

    dropcols = [x for x in 'chainseq target_tdinfo tdinfo'.split()
                if x in targets.columns]
    if dropcols:
        targets.drop(columns=dropcols, inplace=True)

    # these need to be unique
    if 'targetid' not in targets.columns:
        targets['targetid'] = [f'T{ii:05d}_{x.organism}_{x.mhc}_{x.peptide}'
                               for ii,x in enumerate(targets.itertuples())]

    targets['targetid'] = targets.targetid.map(sanitised_name)

    if targets.targetid.value_counts().max() > 1:
        targets['targetid'] = [f'{x}_{i}' for i,x in enumerate(targets.targetid)]


    # start with an unmutated chainseq
    targets['chainseq'] = [sequtil.make_pmhc_tcr_chainseq_for_alphafold(
        x.organism, x.mhc_class, x.mhc, x.peptide,
        x.va, x.ja, x.cdr3a, x.vb, x.jb, x.cdr3b,
        va_mutations = x.va_mutations if hasattr(x,'va_mutations') else [],
        vb_mutations = x.vb_mutations if hasattr(x,'vb_mutations') else [])
                           for x in targets.itertuples()]

    targets['target_tdinfo'] = [
        get_tdinfo.get_row_tdinfo(x, trust_tcr_positions=True).to_string()
        for _,x in targets.iterrows()]

    ## now we need to make a JSON file
    ## MSA templates, etc

    jsonfiles = []

    for _, row in targets.iterrows():

        is_unbound_tcr = row.mhc is None or pd.isna(row.mhc)
        is_unbound_pmhc = row.va is None or pd.isna(row.va)

        # msas
        if is_unbound_tcr:
            all_msas = make_af3_msas_for_unbound_tcr(row, debug=debug)
        elif row.mhc_class==1:
            all_msas = make_af3_msas_for_class_1_row(row, debug=debug)
        else:
            all_msas = make_af3_msas_for_class_2_row(row, debug=debug)

        if 'template_pdbfile' in targets.columns:
            all_templates = make_af3_templates_from_template_pdbfile(
                row.chainseq, row.template_pdbfile,
                is_unbound_pmhc = is_unbound_pmhc,
                is_unbound_tcr = is_unbound_tcr,
            )
        else:
            num_chains = row.chainseq.count('/')+1
            all_templates = [[] for _ in range(num_chains)]

        if is_unbound_tcr:
            assert len(all_msas) == len(all_templates) == 2
        elif is_unbound_pmhc:
            assert len(all_msas) == len(all_templates) == 1+row.mhc_class #
        else:
            assert len(all_msas) == len(all_templates) == 3+row.mhc_class #

        res = dict(
            dialect= 'alphafold3',
            version= 1,
            name= row.targetid,
            sequences= [],
            modelSeeds= [random.randint(0, 2**32 - 1) for _ in range(num_seeds)],
            bondedAtomPairs= None,
            userCCD= None,
        )

        cs = row.chainseq.split('/')
        for ii, (id, seq, templates, msas) in enumerate(zip(
                'ABCDE', cs, all_templates, all_msas)):
            info = dict(
                id= id,
                sequence= util.map_sequence_with_mods(seq),
                modifications= [],
                unpairedMsa= msas,
                pairedMsa= '',
                templates= templates,
            )

            for jj, aa in enumerate(seq):
                if aa in util.mod_ccd_map:
                    print('add modification:', ii, jj, aa, util.mod_ccd_map[aa])
                    info['modifications'].append({'ptmType': util.mod_ccd_map[aa],
                                                  'ptmPosition': jj+1})

            res['sequences'].append({'protein': info})
        outfile = f'{outdir}{row.targetid}_data.json'
        with open(outfile,'w') as out:
            json.dump(res, out)
        print('made:', outfile)

        jsonfiles.append(outfile)

    targets['af3_input_jsonfile'] = jsonfiles

    return targets



def update_af3_jsonfiles_to_match_designed_sequences(
        targets,
        outdir,
):
    # need to update
    # - sequence
    # - first line of msa
    #
    assert targets.targetid.value_counts().max() == 1

    makedirs(outdir, exist_ok=True)

    dfl = [] # list of new jsonfiles
    for _,row in targets.iterrows():
        with open(row.af3_input_jsonfile) as f:
            res = json.load(f)
        res['name'] = sanitised_name(row.targetid) # NOTE NOTE NOTE UPDATE name
        num_chains = len(res['sequences'])
        cs = row.chainseq.split('/')
        assert len(cs) == num_chains
        for ii, newseq_w_mods in enumerate(cs):
            newseq = util.map_sequence_with_mods(newseq_w_mods)
            oldseq = res['sequences'][ii]['protein']['sequence']
            assert len(oldseq) == len(newseq)
            if oldseq != newseq:
                print('changes to chain:', ii, 'sequence:',
                      [(i,a,b) for i,(a,b) in enumerate(zip(oldseq,newseq))
                       if a!=b])
                res['sequences'][ii]['protein']['sequence'] = newseq
                l = res['sequences'][ii]['protein']['unpairedMsa'].split('\n')
                assert l[0] == '>query' and l[1] == oldseq
                l[1] = newseq
                res['sequences'][ii]['protein']['unpairedMsa'] = '\n'.join(l)

        outfile = f'{outdir}{row.targetid}_data.json'
        with open(outfile,'w') as out:
            json.dump(res, out)
        print('made:', outfile)
        dfl.append(outfile)
    targets['af3_input_jsonfile'] = dfl
    return targets


