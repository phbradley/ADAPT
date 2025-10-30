import os
import sys
from sys import exit
import copy
import itertools as it
from pathlib import Path
from os.path import exists, isdir
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
from tcrdock.tcrdist.amino_acids import amino_acids
from tcrdock.docking_geometry import compute_docking_geometries_distance_matrix
import pandas as pd
import numpy as np
import random
from os import system, popen, mkdir
from glob import glob
from collections import Counter, OrderedDict, namedtuple
import scipy
import json

strength_aas = frozenset(list('CFILMVWY'))

def get_total_strength(seq):
    return sum(x in strength_aas for x in seq)


all_template_poses = {} # global dict to avoid reloading...
def get_const_pose_and_tdinfo(fname):
    global all_template_poses
    if fname not in all_template_poses:
        pose = td2.pdblite.pose_from_pdb(fname)
        tdifile = fname+'.tcrdock_info.json'
        if exists(tdifile):
            with open(tdifile, 'r') as f:
                tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(f.read())
        else:
            tdinfo = None
        all_template_poses[fname] = (pose, tdinfo)

    return all_template_poses[fname] # NOT making a copy!

rmsd_atoms = [' N  ', ' CA ', ' C  ', ' O  ']

def get_designable_positions(
        alignstring=None,
        row=None,
        extend_flex=1,
        nres=None,
        reverse=False,
        num_contigs=2, # for debugging
):
    ''' this is way too complicated

    moving toward just using row.designable_positions

    old comment:
    the unaligned position plus extend_flex rsds on either side of each contig
    '''
    if hasattr(row, 'designable_positions'):
        posl = [int(x) for x in row.designable_positions.split(',')]
        return sorted(posl) # want sorted!

    if alignstring is None:
        alignstring = row.template_0_target_to_template_alignstring

    align = dict(tuple(map(int, x.split(':'))) for x in alignstring.split(';'))
    if reverse:
        align = {y:x for x,y in align.items()}
    fixed_posl = sorted(align.keys())
    if nres is None: # assume no terminal designable positions
        nres = max(fixed_posl)+1
    flex_posl = [x for x in range(nres) if x not in fixed_posl]
    if extend_flex:
        new_flex_posl = set()
        for pos in flex_posl:
            new_flex_posl.update([
                pos+i for i in range(-extend_flex, extend_flex+1)])
        assert len(new_flex_posl) == len(flex_posl) + 2*extend_flex*num_contigs
        flex_posl = sorted(new_flex_posl)
    return flex_posl

def get_rmsd_coords(pose, flex_posl):
    '''
    '''
    coords = pose['coords']
    resids = pose['resids']

    return np.stack([coords[resids[i]][a] for i in flex_posl for a in rmsd_atoms])

def filled_in_alignment(align_in):
    'returns a new dict, does not change input'
    align = copy.deepcopy(align_in)
    nres = max(align.keys())+1
    nat_nres = max(align.values())+1
    #new_align = deepcopy(align)
    gap_starts = [i for i in align if i+1 not in align and i+1<nres]
    gap_stops  = [i for i in align if i-1 not in align and i>0]
    assert len(gap_starts) == 2 and len(gap_stops) == 2
    for start, stop in zip(sorted(gap_starts), sorted(gap_stops)):
        nstart, nstop = align[start], align[stop]
        mingap = min(stop-start-1, nstop-nstart-1)
        nterm, cterm = mingap//2, mingap-mingap//2
        for i in range(nterm):
            align[start+1+i] = nstart+1+i
        for i in range(cterm):
            align[stop-1-i] = nstop-1-i
    #print(len(align), nres, nat_nres)
    #assert len(align) == min(nres, nat_nres)
    assert len(set(align.values())) == len(align)
    return align


def compute_stats(
        targets,
        extend_flex=1,
):
    ''' returns targets with stats like peptide_loop_pae, rmsds, recovery filled in
    (assuming the template pdbfiles are the native structures)
    '''
    required_cols = ('chainseq template_0_template_pdbfile '
                     ' template_0_target_to_template_alignstring '
                     ' model_pdbfile model_plddtfile model_paefile'.split())
    for col in required_cols:
        assert col in targets.columns

    dfl = []
    for _,l in targets.iterrows():
        nres = len(l.chainseq.replace('/',''))
        alignstring = l.template_0_target_to_template_alignstring
        nat_pose, tdinfo = get_const_pose_and_tdinfo(l.template_0_template_pdbfile)
        a,b,c = nat_pose['chainbounds'][:3] # a = 0
        nat_sequence = nat_pose['sequence']
        sequence = l.chainseq.replace('/','')

        plddts = np.load(l.model_plddtfile)[:nres]
        paes = np.load(l.model_paefile)[:nres,:][:,:nres]

        align = dict(tuple(map(int,x.split(':'))) for x in alignstring.split(';'))
        align_rev = {y:x for x,y in align.items()}
        mod_flex_posl = get_designable_positions(
            alignstring=alignstring, extend_flex=extend_flex)
        nat_flex_posl = get_designable_positions(
            alignstring=alignstring, extend_flex=extend_flex, reverse=True)

        if tdinfo is None:
            nat_cdr3_bounds = [[0,1],[2,3]] # hacking
        else:
            nat_cdr3_bounds = [tdinfo.tcr_cdrs[3], tdinfo.tcr_cdrs[7]]
        mod_cdr3_bounds = [[align_rev[x[0]], align_rev[x[1]]] for x in nat_cdr3_bounds]

        nat_cdr3s, mod_cdr3s = [], []
        for start,stop in nat_cdr3_bounds:
            nat_cdr3s.append(nat_sequence[start:stop+1])
            mod_cdr3s.append(sequence[align_rev[start]:align_rev[stop]+1])

        # actually two loops:
        loop_seq = ''.join(sequence[x] for x in mod_flex_posl)
        wt_loop_seq = ''.join(nat_sequence[x] for x in nat_flex_posl)


        # rmsds
        pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)
        align_full = filled_in_alignment(align)
        mod_coords = get_rmsd_coords(
            pose, [x for x in mod_flex_posl if x in align_full])
        nat_coords = get_rmsd_coords(
            nat_pose,
            [align_full[x] for x in mod_flex_posl if x in align_full])
        natoms = mod_coords.shape[0]
        assert nat_coords.shape == (natoms, 3)
        cdr3a_flex_coords_len = sum(
            x in align_full and x in mod_flex_posl for x in range(*mod_cdr3_bounds[0]))
        cdr3b_flex_coords_len = sum(
            x in align_full and x in mod_flex_posl for x in range(*mod_cdr3_bounds[1]))
        #print('cdr3a_flex_coords_len:', cdr3a_flex_coords_len,
        #      cdr3b_flex_coords_len, natoms//len(rmsd_atoms))
        assert ((cdr3a_flex_coords_len + cdr3b_flex_coords_len)*len(rmsd_atoms) ==
                natoms)

        nat_mhc_coords = nat_pose['ca_coords'][:b]
        mhc_coords = pose['ca_coords'][:b]

        R, v = td2.superimpose.superimposition_transform(
            nat_mhc_coords, mhc_coords)

        mod_coords = (R@mod_coords.T).T + v
        rmsd = np.sqrt(np.sum((nat_coords-mod_coords)**2)/natoms)
        split = cdr3a_flex_coords_len * len(rmsd_atoms)
        rmsda = np.sqrt(np.sum((nat_coords[:split]-mod_coords[:split])**2)/split)
        rmsdb = np.sqrt(np.sum((nat_coords[split:]-mod_coords[split:])**2)/
                        (natoms-split))


        outl = l.copy()
        outl['loop_plddt'] = plddts[mod_flex_posl].mean()
        outl['loop_rmsd'] = rmsd
        outl['aloop_rmsd'] = rmsda
        outl['bloop_rmsd'] = rmsdb
        outl['loop_seq'] = loop_seq
        outl['wt_loop_seq'] = wt_loop_seq
        outl['cdr3a'] = mod_cdr3s[0]
        outl['wt_cdr3a'] = nat_cdr3s[0]
        outl['ashift'] = len(mod_cdr3s[0])-len(nat_cdr3s[0])
        outl['cdr3b'] = mod_cdr3s[1]
        outl['wt_cdr3b'] = nat_cdr3s[1]
        outl['bshift'] = len(mod_cdr3s[1])-len(nat_cdr3s[1])
        outl['peptide'] = l.chainseq.split('/')[1]
        outl['wt_peptide'] = nat_pose['chainseq'].split('/')[1]
        outl['peptide_plddt'] = plddts[b:c].mean()
        outl['peptide_loop_pae'] = 0.5*(
            paes[b:c,:][:,mod_flex_posl].mean() +
            paes[mod_flex_posl,:][:,b:c].mean())

        dfl.append(outl)


    targets = pd.DataFrame(dfl)

    return targets



mhc_seq2core_positions = {}
def get_mhc_core_positions_0x_cached(mhc_class, mhc_seq, verbose=True):
    global mhc_seq2core_positions
    # if mhc_class==2 mhc_seq is a tuple
    if (mhc_class, mhc_seq)  not in mhc_seq2core_positions:
        if verbose:
            print('get mhc core:', mhc_class, mhc_seq)
        if mhc_class==1:
            posl = td2.mhc_util.get_mhc_core_positions_class1(mhc_seq)
        else:
            posl = td2.mhc_util.get_mhc_core_positions_class2(*mhc_seq)
        mhc_seq2core_positions[(mhc_class,mhc_seq)] = posl

    return mhc_seq2core_positions[(mhc_class, mhc_seq)]

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
    from tcrdock.sequtil import ALL_GENES_GAP_CHAR
    from tcrdock.tcrdist.all_genes import all_genes
    from tcrdock.tcrdist.parsing import parse_tcr_sequence

    core_len = 13 # tcr core len

    sequence = cb_seq.replace('/','')
    nres = len(sequence)
    *mhc_seqs, pep_seq, tcra_seq, tcrb_seq = cb_seq.split('/')
    chainlens = [len(x) for x in cb_seq.split('/')]
    assert mhc_class == len(mhc_seqs)

    mhc_len = sum(chainlens[:-3])
    pmhc_len = sum(chainlens[:-2])
    pmhc_tcra_len = sum(chainlens[:-1])

    tcra_prefix = td2.sequtil.get_v_seq_up_to_cys(organism, va)[:-1] # dont require Cys
    tcrb_prefix = td2.sequtil.get_v_seq_up_to_cys(organism, vb)[:-1]

    if trust_tcr_positions:
        assert tcra_seq[len(tcra_prefix)] == 'C'
        assert tcrb_seq[len(tcrb_prefix)] == 'C'

    if (trust_tcr_positions or
        (tcra_seq.startswith(tcra_prefix) and tcrb_seq.startswith(tcrb_prefix))):
        # yay! we got a match to 'clean' model tcr seqs up to the Cys
        tcr_core_positions = (
            [x+pmhc_len      for x in td2.sequtil.get_core_positions_0x(organism, va)]+
            [x+pmhc_tcra_len for x in td2.sequtil.get_core_positions_0x(organism, vb)])

        tcr_cdrs = []
        for v,preseq,seq,cdr3,offset in [[va,tcra_prefix,tcra_seq,cdr3a,pmhc_len],
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


    tcr_coreseq = ''.join(sequence[x] for x in tcr_core_positions)
    cys_seq = tcr_coreseq[1]+tcr_coreseq[12]+tcr_coreseq[14]+tcr_coreseq[25]
    if verbose or cys_seq != 'CCCC':
        print('tcr_coreseq:', cys_seq, tcr_coreseq[:core_len], tcr_coreseq[core_len:])


    mhc_seq = mhc_seqs[0] if mhc_class==1 else tuple(mhc_seqs)
    mhc_core_positions_0x = get_mhc_core_positions_0x_cached(mhc_class, mhc_seq,
                                                             verbose=False)

    tdinfo = td2.tcrdock_info.TCRdockInfo().from_dict(
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
        return td2.tcrdock_info.TCRdockInfo().from_string(row.tdinfo)
    elif hasattr(row, 'target_tdinfo') and not pd.isna(row.target_tdinfo):
        return td2.tcrdock_info.TCRdockInfo().from_string(row.target_tdinfo)

    return get_model_tdinfo(
        row.organism, row.mhc_class, row.mhc, row.chainseq, row.va, row.ja, row.cdr3a,
        row.vb, row.jb, row.cdr3b, **kwargs,
    )


def compute_docking_geometry_info(
        l,
        pose=None,
        trust_tcr_positions=False,
        compare_to_old_dgeom=False,
        default_organism = 'human', #legacy, used to compute dgeom distances
):
    ''' l is a Series

    returns None if failure

    otherwise returns dict with some info including dgeom_rmsd to prev dgeom if present
    '''

    # maybe compute docking geometry
    if hasattr(l,'tdinfo') or hasattr(l,'target_tdinfo'):
        required_cols = ['chainseq']
    else:
        required_cols = ('organism mhc_class mhc chainseq va ja cdr3a '
                         ' vb jb cdr3b').split()
    if pose is None:
        required_cols.append('model_pdbfile')

    if not all(hasattr(l,x) for x in required_cols):
        print('WARNING compute_docking_geometry_info missing required cols:',
              [x for x in required_cols if not hasattr(l,x)])
        return None
    else:
        organism = l.organism if hasattr(l,'organism') else default_organism
        tdinfo = get_row_tdinfo(l, trust_tcr_positions=trust_tcr_positions,
                                verbose=False)
        if tdinfo.mhc_class==0:
            print('compute_docking_geometry_info: mhc_class=0, returning None')
            return None # cant define geometry
        if l.chainseq.count('/') not in [3,4]:
            print('compute_docking_geometry_info: bad num chains=',
                  l.chainseq.count('/')+1, 'returning None')
            return None
            
        cbs = [0] + list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        if pose is None:
            pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)
        assert pose['sequence'] == l.chainseq.replace('/','')

        pose = td2.pdblite.set_chainbounds_and_renumber(pose, cbs)

        dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(pose, tdinfo)

        dgcols = 'torsion d tcr_unit_y tcr_unit_z mhc_unit_y mhc_unit_z'.split()

        outl = {}
        if compare_to_old_dgeom and hasattr(l, 'old_torsion'):
            d = {k:l['old_'+k] for k in dgcols}
            old_dgeom = td2.docking_geometry.DockingGeometry().from_dict(d)
            outl['dgeom_rmsd'] = compute_docking_geometries_distance_matrix(
                [dgeom], [old_dgeom], organism=organism)[0,0]

        elif hasattr(l, dgcols[0]):
            # already has docking geometry info, compute rmsd, save old info
            old_dgeom = td2.docking_geometry.DockingGeometry().from_dict(l)
            for col in dgcols:
                outl['old_'+col] = l[col]
            outl['dgeom_rmsd'] = compute_docking_geometries_distance_matrix(
                [dgeom], [old_dgeom], organism=organism)[0,0]

        for k,v in dgeom.to_dict().items():
            outl[k] = v

    return outl # either a dict or None

def compute_simple_stats(
        targets,
        extend_flex=1,
        trust_tcr_positions=False,
        compare_to_old_dgeom=False,
):
    ''' Not assuming a single 'native' template

    this assumes we can call:

      flex_posl = get_designable_positions(row=l, extend_flex=extend_flex)


    stats:

    loop_plddt
    loop_seq
    loop_seq2
    peptide
    peptide_plddt
    peptide_loop_pae
    pmhc_tcr_pae

    '''
    required_cols = ('chainseq model_plddtfile model_paefile'.split())
    for col in required_cols:
        assert col in targets.columns, f'Need {col} column in targets df'

    dfl = []
    for _, l in targets.iterrows():
        sequence = l.chainseq.replace('/','')
        nres = len(sequence)
        cbs = [0]+list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        chain_number = np.zeros((nres,), dtype=int)
        for pos in cbs[1:-1]:
            chain_number[pos:] += 1
        num_chains = len(cbs)-1
        assert chain_number[0] == 0 and chain_number[-1] == num_chains-1

        peptide = l.chainseq.split('/')[-3]
        nres_mhc, nres_pmhc = cbs[-4:-2]
        assert sequence[nres_mhc:nres_pmhc] == peptide
        flex_posl = get_designable_positions(row=l, extend_flex=extend_flex)
        flex_posl.sort() # probably not necessary

        if pd.isna(l.model_plddtfile) or pd.isna(l.model_paefile):
            #print('barf!')
            #print(l)
            plddts = np.full((nres,), np.nan)
            paes = np.full((nres,nres), np.nan)
        else:
            plddts = np.load(l.model_plddtfile)[:nres]
            paes = np.load(l.model_paefile)[:nres,:][:,:nres]

        # actually two (or more) loops:
        loop_seq = ''.join(sequence[x] for x in flex_posl)

        loop_seq2 = sequence[flex_posl[0]]
        for i,j in zip(flex_posl[:-1], flex_posl[1:]):
            if chain_number[i] != chain_number[j]:
                loop_seq2 += '/'
            elif j!=i+1:
                loop_seq2 += '-'
            loop_seq2 += sequence[j]

        total_loop_strength = get_total_strength(loop_seq)
        mean_loop_strength = total_loop_strength / len(loop_seq)

        loopseqs = loop_seq2.replace('/','-').split('-')
        total_loop_strengths = [get_total_strength(x) for x in loopseqs]
        mean_loop_strengths = [x/len(y) for x,y in zip(total_loop_strengths,loopseqs)]

        outl = l.copy()
        outl['loop_plddt'] = plddts[flex_posl].mean()
        outl['loop_seq'] = loop_seq
        outl['loop_seq2'] = loop_seq2
        outl['total_loop_strength'] = total_loop_strength
        outl['mean_loop_strength'] = mean_loop_strength
        outl['total_loop_strengths'] = ','.join([str(x) for x in total_loop_strengths])
        outl['mean_loop_strengths']= ','.join([f'{x:.2f}' for x in mean_loop_strengths])
        outl['peptide'] = peptide
        outl['peptide_plddt'] = plddts[nres_mhc:nres_pmhc].mean()
        outl['peptide_loop_pae'] = 0.5*(
            paes[nres_mhc:nres_pmhc,:][:,flex_posl].mean() +
            paes[flex_posl,:][:,nres_mhc:nres_pmhc].mean())
        outl['pmhc_tcr_pae'] = 0.5*(
            paes[:nres_pmhc,:][:,nres_pmhc:].mean() +
            paes[nres_pmhc:,:][:,:nres_pmhc].mean())
        if hasattr(l, 'loop_seq'):
            outl['old_loop_seq'] = l.loop_seq
        if hasattr(l, 'loop_seq2'):
            outl['old_loop_seq2'] = l.loop_seq2
        if hasattr(l, 'pmhc_tcr_pae'):
            outl['old_pmhc_tcr_pae'] = l.pmhc_tcr_pae

        # maybe compute docking geometry
        dginfo = compute_docking_geometry_info(
            l, trust_tcr_positions=trust_tcr_positions,
            compare_to_old_dgeom=compare_to_old_dgeom,
        )
        if dginfo is not None:
            for k,v in dginfo.items():
                outl[k] = v
        dfl.append(outl)

    targets = pd.DataFrame(dfl)

    return targets


def reduce_pae_matrix_to_residues(
        paes,
        token_chain_ids,
        token_res_ids,
):
    ''' May be called recursively
    '''
    n0 = len(token_chain_ids)
    assert len(token_res_ids) == n0
    assert paes.shape == (n0,n0)

    resid_counts = Counter(zip(token_chain_ids, token_res_ids))

    resid, count = resid_counts.most_common(1)[0]


    if count==1:
        #print('already done!')
        return paes, token_chain_ids, token_res_ids  # already done
    
    print('reduce_pae_matrix_to_residues:', n0, resid, count)

    mask = (token_chain_ids==resid[0]) & (token_res_ids==resid[1])
    assert mask.sum() == count

    inds = np.nonzero(mask)[0]
    start, stop0 = inds[0], inds[-1]+1
    assert stop0-start == count # contiguous
    stop1 = start+1

    n1 = n0-count+1
    newpaes = np.zeros((n1,n1))
    newpaes[:start, :start] = paes[:start, :start]
    newpaes[:start, stop1:] = paes[:start, stop0:]
    newpaes[stop1:, :start] = paes[stop0:, :start]
    newpaes[stop1:, stop1:] = paes[stop0:, stop0:]

    newpaes[:start, start] = paes[:start, start:stop0].mean(axis=1)
    newpaes[stop1:, start] = paes[stop0:, start:stop0].mean(axis=1)
    
    newpaes[start, :start] = paes[start:stop0, :start].mean(axis=0)
    newpaes[start, stop1:] = paes[start:stop0, stop0:].mean(axis=0)
    
    newpaes[start,start] = paes[start:stop0,:][:,start:stop0].mean() # self pae

    new_token_chain_ids = np.concatenate([token_chain_ids[:start],
                                          [token_chain_ids[start]],
                                          token_chain_ids[stop0:]])
    
    new_token_res_ids = np.concatenate([token_res_ids[:start],
                                        [token_res_ids[start]],
                                        token_res_ids[stop0:]])

    assert newpaes.shape == (n1,n1)
    assert new_token_chain_ids.shape == (n1,)
    assert new_token_res_ids.shape == (n1,)
    
    return reduce_pae_matrix_to_residues( # recursive call
        newpaes, new_token_chain_ids, new_token_res_ids,
    )

def read_af3_pose_plddts_and_paes(ciffile, full_conffile):
    ''' returns pose, plddts, residue_indices, atoms, paes
     which are (1d-np.array, 1d-np.array, list, 2d-np.array)

    atoms is a list of tuples: (resindex, chain, resnumstring, atomname)
    
    '''
    pose = td2.pdblite.pose_from_cif(ciffile)
    with open(full_conffile,'r') as f:
        confs = json.load(f)

    plddts = np.array(confs['atom_plddts'])
    paes = np.array(confs['pae'])
    chain_ids = confs['atom_chain_ids']

    token_chain_ids = np.array(confs['token_chain_ids'])
    token_res_ids = np.array(confs['token_res_ids'])

    paes, token_chain_ids, token_res_ids = reduce_pae_matrix_to_residues(
        paes, token_chain_ids, token_res_ids)

    coords = pose['coords']
    atoms = []
    for ind, r in enumerate(pose['resids']):
        chain, resnumstring = r
        for atomname,_ in coords[r].items():
            atom = (ind, chain, resnumstring, atomname)
            assert chain_ids[len(atoms)] == chain
            atoms.append(atom)

    residue_indices = np.array([x[0] for x in atoms])
    return pose, plddts, residue_indices, atoms, paes


def calc_iptm_combo(d):
    'd is a dict or pandas series'
    return (0.5 * d['iptm'] +
            0.5 * d['peptide_tcr_iptm'] -
            0.01 * d['peptide_tcr_pae'])
    

def compute_af3_stats_for_row(row):
    ''' returns dict with info
    '''
    required_cols = ('chainseq af3_ciffile af3_conffile target_tdinfo'.split())
    
    assert all(hasattr(row,x) for x in required_cols), \
        f'compute_af3_stats_for_row: required_cols: {" ".join(required_cols)}'

    assert 'summary_confidences' in row.af3_conffile
    full_conffile = row.af3_conffile.replace('summary_confidences','confidences')
    assert exists(full_conffile)

    pose, plddts, residue_indices, atoms, paes = read_af3_pose_plddts_and_paes(
        row.af3_ciffile, full_conffile)
    tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(row.target_tdinfo)

    cbs = [0] + list(it.accumulate(len(x) for x in row.chainseq.split('/')))
    assert cbs == pose['chainbounds']

    num_chains = len(cbs)-1

    is_unbound_tcr = len(tdinfo.mhc_core)==0
    is_unbound_pmhc = len(tdinfo.tcr_cdrs)==0

    # is_unbound_tcr = row.mhc is None or pd.isna(row.mhc)
    # is_unbound_pmhc = row.va is None or pd.isna(row.va)

    # if is_unbound_tcr:
    #     assert len(tdinfo.mhc_core)==0
    # if is_unbound_pmhc:
    #     assert len(tdinfo.tcr_cdrs)==0
        
    if is_unbound_tcr:
        nres_mhc, nres_pmhc = 0,0
    else:
        mhc_class = num_chains-1 if is_unbound_pmhc else num_chains-3
        assert mhc_class in [1,2]
        nres_mhc = cbs[mhc_class]
        nres_pmhc = cbs[mhc_class+1]
        mhc_chains = list(range(mhc_class))
        pmhc_chains = list(range(mhc_class+1))
        peptide_chain = mhc_class
        
    nres = cbs[-1]

    assert paes.shape == (nres,nres)

    peptide_inds = list(range(nres_mhc, nres_pmhc))

    cdr_inds, cdr3_inds = [], []
    for ii, (start,stop) in enumerate(tdinfo.tcr_cdrs):
        if ii%4==2: # not cdr2.5
            continue
        elif ii%4==3:
            npad, cpad = 3,2
        else:
            npad, cpad = 0,0

        cdr_inds.extend(range(start+npad, stop+1-cpad))
        if ii%4==3:
            cdr3_inds.extend(range(start+npad, stop+1-cpad))


    peptide_inds = set(peptide_inds)
    cdr_inds = set(cdr_inds)
    cdr3_inds = set(cdr3_inds)

    assert (len(cdr_inds)==0) == is_unbound_pmhc
    assert (len(peptide_inds)==0) == is_unbound_tcr

    peptide_rsd_mask = (np.arange(nres) >= nres_mhc) & (np.arange(nres) < nres_pmhc)
    pmhc_rsd_mask = np.arange(nres) < nres_pmhc
    mhc_rsd_mask = np.arange(nres) < nres_mhc
    tcr_rsd_mask = np.arange(nres) >= nres_pmhc
    
    peptide_atom_mask = np.array(
        [x in peptide_inds for x in residue_indices])
    cdr_atom_mask = np.array(
        [x in cdr_inds for x in residue_indices])
    cdr3_atom_mask = np.array(
        [x in cdr3_inds for x in residue_indices])

    with open(row.af3_conffile) as f:
        sumconfs = json.load(f)

    cpiptm = sumconfs['chain_pair_iptm']
    cppaemin = sumconfs['chain_pair_pae_min']
    
    if is_unbound_tcr or is_unbound_pmhc:
        pmhc_tcr_pae = 0.
        peptide_tcr_pae = 0.
        pmhc_tcr_iptm = 0.
        peptide_tcr_iptm = 0.
        pmhc_tcr_paemin = 0.
        peptide_tcr_paemin = 0.
    else:
        pmhc_tcr_pae = 0.5 * (paes[pmhc_rsd_mask,:][:, tcr_rsd_mask].mean() +
                              paes[ tcr_rsd_mask,:][:,pmhc_rsd_mask].mean())

        peptide_tcr_pae = 0.5 * (paes[peptide_rsd_mask,:][:,tcr_rsd_mask].mean() +
                                 paes[tcr_rsd_mask,:][:,peptide_rsd_mask].mean())

        pmhc_tcr_iptm = 0.5 * np.mean(
            [cpiptm[i][j]+cpiptm[j][i] for i in pmhc_chains for j in [-2,-1]])

        peptide_tcr_iptm = 0.5 * np.mean(
            [cpiptm[i][j]+cpiptm[j][i] for i in [peptide_chain] for j in [-2,-1]])

        pmhc_tcr_paemin = 0.5 * min(
            [cppaemin[i][j]+cppaemin[j][i] for i in pmhc_chains for j in [-2,-1]])

        peptide_tcr_paemin = 0.5 * min(
            [cppaemin[i][j]+cppaemin[j][i] for i in [peptide_chain] for j in [-2,-1]])

    if is_unbound_tcr:
        mhc_peptide_pae = 0.
        mhc_peptide_iptm = 0.
        mhc_peptide_paemin = 0.
    else:
        mhc_peptide_pae = 0.5 * (paes[peptide_rsd_mask,:][:,mhc_rsd_mask].mean() +
                                 paes[mhc_rsd_mask,:][:,peptide_rsd_mask].mean())

        mhc_peptide_iptm = 0.5 * np.mean(
            [cpiptm[i][j]+cpiptm[j][i] for i in mhc_chains for j in [peptide_chain]])

        mhc_peptide_paemin = 0.5 * min(
            [cppaemin[i][j]+cppaemin[j][i] for i in mhc_chains for j in [peptide_chain]]
        )

        
    
    
    outl = dict(
        iptm= sumconfs['iptm'],
        ptm= sumconfs['ptm'],
        ranking_score= sumconfs['ranking_score'],
        pmhc_tcr_pae= pmhc_tcr_pae,
        pmhc_tcr_paemin= pmhc_tcr_paemin,
        pmhc_tcr_iptm= pmhc_tcr_iptm,
        peptide_tcr_pae= peptide_tcr_pae,
        peptide_tcr_paemin= peptide_tcr_paemin,
        peptide_tcr_iptm= peptide_tcr_iptm,
        mhc_peptide_pae= mhc_peptide_pae,
        mhc_peptide_paemin= mhc_peptide_paemin,
        mhc_peptide_iptm= mhc_peptide_iptm,
    )
    outl['iptm_combo'] = calc_iptm_combo(outl)
    outl['plddt'] = plddts.mean() # new
    for tag, mask in zip('peptide cdr cdr3'.split(),
                         [peptide_atom_mask, cdr_atom_mask, cdr3_atom_mask]):
        if mask.sum():
            outl[tag+'_plddt'] = plddts[mask].mean()
        else:
            outl[tag+'_plddt'] = 0.

    if not (is_unbound_tcr or is_unbound_pmhc):
        dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(pose, tdinfo)
        for k,v in dgeom.to_dict().items():
            outl[k] = v

        
    return outl

def add_plddt_pae_iptm_combo_to_df(results, verbose=False):
    ''' adds (or overwrites) the column "plddt_pae_iptm_combo"

    BIGGER IS BETTER for this score
    
    '''
    weights = { # will be multiplied by 1/std
        'peptide_plddt':5,
        'cdr_plddt':1,
        'cdr3_plddt':3,
        'peptide_tcr_pae':-3, # since smaller paes are better
        'iptm_combo':0.1,
    }
    
    stds = { # based on a single analysis!
        'peptide_plddt':1.5,
        'cdr_plddt':2.5,
        'cdr3_plddt':4.5,
        'peptide_tcr_pae':0.35,
        'iptm_combo':0.02, # actually 0.016040
    }

    results['plddt_pae_iptm_combo'] = 0.
    for col,wt in weights.items():
        std = stds[col]
        if verbose:
            true_std = results[col].std()
            print(f'true_std: {true_std:.3f} std: {std:.3f} wt: {wt:6.2f} {col}')
        results['plddt_pae_iptm_combo'] += wt * results[col] / std
        
    return results


def compute_af3_stats(
        targets,
        verbose=False,
):
    '''
    paes, plddts, docking geometry
    '''
    required_cols = ('chainseq af3_ciffile af3_conffile target_tdinfo'.split())
    for col in required_cols:
        assert col in targets.columns, f'Need {col} column in targets df'

    dfl = []
    oldcols = set(targets.columns)
    for ind, (_,l) in enumerate(targets.iterrows()):
        res = compute_af3_stats_for_row(l)
        outl = l.copy()
        for k,v in res.items():
            if 'older_'+k in oldcols:
                outl['oldest_'+k] = outl['older_'+k]
            if 'old_'+k in oldcols:
                outl['older_'+k] = outl['old_'+k]
            if k in oldcols:
                outl['old_'+k] = outl[k]
            outl[k] = v
        dfl.append(outl)

        if verbose and ind%25==0:
            print('compute_af3_stats:', ind, targets.shape[0])

    targets = pd.DataFrame(dfl)
    add_plddt_pae_iptm_combo_to_df(targets)

    return targets



def add_info_to_rescoring_row(l, model_name, extend_flex=1):
    ''' l is a pandas Series
    uses chainseq, plddt-file, pae-file, template_0_alt_template_sequence
    YES extending of the loop definition by +/- extend_flex for the TCR
    '''
    #if model_name is None:
    #    model_name = l.model_name
    sequence = l.chainseq.replace('/','')
    nres = len(sequence)
    plddts = np.load(l[f'{model_name}_plddt_file'])[:nres]
    paes = np.load(l[f'{model_name}_predicted_aligned_error_file'])[:nres,:nres]

    cbs = [0] + list(it.accumulate(len(x) for x in l.chainseq.split('/')))
    nres_mhc, nres_pmhc = cbs[-4:-2]
    gap_posl = [i for i,x in enumerate(l.template_0_alt_template_sequence)
                if x not in amino_acids]
    pep_inds = [x for x in gap_posl if x < nres_pmhc]
    loop_inds = [x for x in gap_posl if x >= nres_pmhc]
    assert all(x>=nres_mhc for x in pep_inds)
    #gap_starts = [i for i in gap_posl if i-1 not in gap_posl]
    #gap_stops  = [i+1 for i in gap_posl if i+1 not in gap_posl]
    #assert len(gap_starts) == 3 and len(gap_stops) == 3
    #bounds = list(zip(gap_starts, gap_stops))
    #pep_inds = list(range(*bounds[0]))
    peptide = ''.join(sequence[x] for x in pep_inds)
    peptide_plddt = plddts[pep_inds].mean()
    #loop_inds = list(it.chain(range(*bounds[1]), range(*bounds[2])))
    if extend_flex:
        new_loop_inds = set()
        for pos in loop_inds:
            new_loop_inds.update([
                pos+i for i in range(-extend_flex, extend_flex+1)])
        assert len(new_loop_inds) == len(loop_inds) + 4*extend_flex
        loop_inds = sorted(new_loop_inds)
    #new_loop_inds = [i-1 for i in loop_inds] + [i+1 for i in loop_inds]
    loop_seq = ''.join(sequence[x] for x in loop_inds)
    loop_plddt = plddts[loop_inds].mean()
    peptide_loop_pae = 0.5 * (
        paes[pep_inds,:][:,loop_inds].mean() +
        paes[loop_inds,:][:,pep_inds].mean())
    pmhc_tcr_pae = 0.5*(
        paes[:nres_pmhc,:][:,nres_pmhc:].mean() +
        paes[nres_pmhc:,:][:,:nres_pmhc].mean())
    mask_char = set(x for x in l.template_0_alt_template_sequence
                    if x not in amino_acids)
    assert len(mask_char) == 1
    mask_char = mask_char.pop()
    outl = l.copy()
    outl['model_name'] = model_name
    outl['peptide_plddt'] = peptide_plddt
    outl['peptide_len'] = len(pep_inds)
    outl['peptide'] = peptide
    outl['loop_plddt'] = loop_plddt
    outl['loop_len'] = len(loop_inds)
    outl['loop_seq'] = loop_seq
    outl['peptide_loop_pae'] = peptide_loop_pae
    outl['mask_char'] = mask_char
    outl['pmhc_tcr_pae'] = pmhc_tcr_pae

    return outl



def compare_models(
        amodels,
        bmodels,
        default_organism = 'human', #legacy, used to compute dgeom distances
        trust_tcr_positions = False,
):
    ''' return a dataframe with the results

    dgeom rmsd (dock rmsd)

    rmsd over all 3 cdr loops (after superimposing on MHC)
    rmsd over just the cdr3 loop (after superimposing on MHC)


    '''
    from tcrdock.docking_geometry import compute_docking_geometries_distance_matrix
    assert amodels.shape[0] == bmodels.shape[0]

    required_cols = 'chainseq model_pdbfile'.split()
    has_tdinfo = any(col in amodels.columns and col in bmodels.columns
                     for col in 'tdinfo target_tdinfo'.split())
    if not has_tdinfo:
        required_cols.extend('organism mhc_class mhc va ja cdr3a vb jb cdr3b'.split())

    assert all(x in amodels.columns for x in required_cols)
    assert all(x in bmodels.columns for x in required_cols)

    def read_row(row):
        tdinfo = get_row_tdinfo(row, trust_tcr_positions=trust_tcr_positions)
        pose = td2.pdblite.pose_from_pdb(row.model_pdbfile)
        cbs = [0] + list(it.accumulate(len(x) for x in row.chainseq.split('/')))
        pose = td2.pdblite.set_chainbounds_and_renumber(pose, cbs)
        pose = td2.mhc_util.orient_pmhc_pose(pose, tdinfo=tdinfo)
        dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(pose, tdinfo)

        cdr_coords = {}
        N, CA, C  = ' N  ', ' CA ', ' C  '
        cdr_posl = []
        for ii, loop in enumerate(tdinfo.tcr_cdrs):
            coords = []
            for pos in range(loop[0], loop[1]+1):
                coords.append(pose['coords'][pose['resids'][pos]][N])
                coords.append(pose['coords'][pose['resids'][pos]][CA])
                coords.append(pose['coords'][pose['resids'][pos]][C])
                cdr_posl.append(pos)
            cdr_coords[ii] = np.array(coords)

        sequence = row.chainseq.replace('/','')
        chain_number = np.zeros((len(sequence),), dtype=int)
        for pos in cbs[1:-1]:
            chain_number[pos:] += 1
        assert chain_number[0] == 0 and chain_number[-1] == len(pose['chains'])-1
        cdr_seq = sequence[cdr_posl[0]]
        for i,j in zip(cdr_posl[:-1], cdr_posl[1:]):
            if chain_number[i] != chain_number[j]:
                cdr_seq += '/'
            elif j!=i+1:
                cdr_seq += '-'
            cdr_seq += sequence[j]


        return dict(
            pose=pose,
            tdinfo=tdinfo,
            cdr_coords=cdr_coords,
            dgeom=dgeom,
            cdr_seq=cdr_seq,
        )

    dfl = []
    for (_,row1), (_,row2) in zip(amodels.iterrows(), bmodels.iterrows()):

        d1 = read_row(row1)
        d2 = read_row(row2)

        samelen_cdrs = [i for i in range(8) if d1['cdr_coords'][i].shape ==
                        d2['cdr_coords'][i].shape]
        samelen_cdr3s = [i for i in [3,7] if i in samelen_cdrs]

        if len(samelen_cdrs) != 8:
            for i in set(range(8))-set(samelen_cdrs):
                print(f'WARNING compare_models:: cdr {i} length mismatch:',
                      d1['cdr_coords'][i].shape[0], d2['cdr_coords'][i].shape[0])

        outl = {}
        for tag, inds in [['cdr',samelen_cdrs], ['cdr3',samelen_cdr3s]]:
            coords1 = np.concatenate([d1['cdr_coords'][x] for x in inds])
            coords2 = np.concatenate([d2['cdr_coords'][x] for x in inds])
            natoms = coords1.shape[0]
            assert coords2.shape == (natoms,3)
            outl[tag+'_rmsd'] = np.sqrt(np.sum((coords1-coords2)**2)/natoms)

        organism = row1.organism if hasattr(row1, 'organism') else default_organism
        outl['dgeom_rmsd'] = compute_docking_geometries_distance_matrix(
            [d1['dgeom']], [d2['dgeom']], organism)[0,0]
        outl['model1_cdr_seq'] = d1['cdr_seq']
        outl['model2_cdr_seq'] = d2['cdr_seq']

        dfl.append(outl)
    return pd.DataFrame(dfl)


def reorient_and_rechain_models(results, keep_old_pdbs=False):
    ''' Returns a new results dataframe with the `model_pdbfile` column updated

    For this we need enough info to define the MHC core so we can orient
    '''

    required_columns = 'chainseq model_pdbfile'.split()
    assert all(x in results.columns for x in required_columns),\
        f'reorient_and_rechain_models:: required_columns= {required_columns}'

    dfl = []
    for _, l in results.iterrows():
        pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)

        *mhc_seqs, pep_seq, tcra_seq, tcrb_seq = l.chainseq.split('/')
        chainbounds = [0] + list(it.accumulate(len(x) for x in l.chainseq.split('/')))

        mhc_class = len(chainbounds)-4
        assert mhc_class in [1,2]

        assert pose['sequence'] == l.chainseq.replace('/','')

        pose = td2.pdblite.set_chainbounds_and_renumber(pose, chainbounds)


        mhc_seq = mhc_seqs[0] if mhc_class==1 else tuple(mhc_seqs)
        mhc_core_positions = get_mhc_core_positions_0x_cached(mhc_class, mhc_seq,
                                                              verbose=False)

        pose = td2.mhc_util.orient_pmhc_pose(
            pose, mhc_core_positions=mhc_core_positions)

        outfile = l.model_pdbfile[:-4]+'_orient.pdb'
        td2.pdblite.dump_pdb(pose, outfile)
        print('made:', outfile)

        if exists(outfile) and not keep_old_pdbs:
            os.remove(l.model_pdbfile)

        outl = l.copy()
        outl['model_pdbfile'] = outfile
        dfl.append(outl)

    return pd.DataFrame(dfl)


def superimpose_pose(
        mov_pose,
        fix_pose,
        align=None,
        atom=' CA ',
        verbose=True,
        return_rmsd=False,
):
    ''' returns mov_pose (AND OPTIONALLY superposition rmsd)

    align maps from mov_pose to fix_pose numbers

    uses C-alpha coords by default (see atom argument)
    '''

    if align is None:
        nres = len(mov_pose['sequence'])
        assert nres == len(fix_pose['sequence'])
        align = {i:i for i in range(nres)}

    rsd_pairs = sorted(align.items())
    mov_coords = np.stack([mov_pose['coords'][mov_pose['resids'][i]][atom]
                           for i,_ in rsd_pairs])

    fix_coords = np.stack([fix_pose['coords'][fix_pose['resids'][i]][atom]
                           for _,i in rsd_pairs])

    R, v = td2.superimpose.superimposition_transform(fix_coords, mov_coords)

    mov_pose = td2.pdblite.apply_transform_Rx_plus_v(mov_pose, R, v)

    if verbose or return_rmsd:
        new_mov_coords = np.stack([mov_pose['coords'][mov_pose['resids'][i]][atom]
                                   for i,_ in align.items()])
        npos = len(align.keys())
        rmsd = np.sqrt(np.sum((new_mov_coords-fix_coords)**2)/npos)
        if verbose:
            print(f'superimpose_pose: rmsd: {rmsd:7.2f} npos: {npos}')

    if return_rmsd:
        return mov_pose, rmsd
    else:
        return mov_pose


def make_fake_cbeta(n,ca,c):
    ''' n, ca, and c are numpy vectors of shape (3,)
    '''
    from numpy.linalg import norm
    from numpy import dot, cross
    mean_cb_coords = np.array([-0.53529231, -0.76736402,  1.20778869])

    origin = ca
    x = (n-ca)/norm(n-ca)
    y = (c-ca)
    y -= x * dot(x,y)
    y /= norm(y)
    z = cross(x,y)
    assert abs(norm(x)-1)<1e-3
    assert abs(norm(y)-1)<1e-3
    assert abs(norm(z)-1)<1e-3
    assert abs(dot(x,y)<1e-3)
    assert abs(dot(z,y)<1e-3)
    assert abs(dot(x,z)<1e-3)

    new_cb = origin + (mean_cb_coords[0] * x +
                       mean_cb_coords[1] * y +
                       mean_cb_coords[2] * z)
    return new_cb


def compute_burials(pose, burier_mask=None):
    from numpy.linalg import norm
    from numpy import dot, cross

    min_d, max_d = 4.5, 9
    min_d2, max_d2 = min_d**2, max_d**2

    nres = len(pose['sequence'])
    coords, resids = pose['coords'], pose['resids']

    if burier_mask is None:
        burier_mask = np.ones((nres,), dtype=bool)
    assert burier_mask.shape == (nres,)

    n_coords  = np.stack([coords[x][' N  '] for x in resids])
    ca_coords = np.stack([coords[x][' CA '] for x in resids])
    c_coords  = np.stack([coords[x][' C  '] for x in resids])

    cb_coords = np.stack(list(map(make_fake_cbeta, n_coords, ca_coords, c_coords)))

    ca_cb_vectors = (cb_coords - ca_coords)
    ca_cb_vectors /= np.linalg.norm(ca_cb_vectors, axis=-1, keepdims=True)

    D2 = np.sum((cb_coords[None,:,:] - cb_coords[:,None,:])**2,-1)

    burials = []
    for ii in range(nres):
        ii_burial=0.
        for jj,d2 in enumerate(D2[ii]):
            if ii==jj or d2 > max_d2 or not burier_mask[jj]:
                continue
            if d2 < 1e-2:
                print(f'residues on top of each other? {ii} {jj} {d2}')
                continue
            d = np.sqrt(d2)
            d_contribution = 1 + min(0, (min_d-d)/(max_d-min_d))
            assert -0.001 <= d_contribution <= 1.001
            v = cb_coords[jj]-cb_coords[ii]
            v/= norm(v)
            vi_dot = np.dot( v, ca_cb_vectors[ii])
            vj_dot = np.dot(-v, ca_cb_vectors[jj])
            assert -1.001 <= vi_dot <= 1.001
            assert -1.001 <= vj_dot <= 1.001
            vi_contribution = (2+vi_dot)/3
            vj_contribution = (3+vj_dot)/4
            contribution = d_contribution * vi_contribution * vj_contribution
            ii_burial += contribution
        burials.append(ii_burial)
    return np.array(burials)


def get_mhc_vgene_buried_strength(pose, tdinfo):
    cbs = [0] + list(it.accumulate(len(x) for x in pose['chainseq'].split('/')))
    num_chains = len(cbs)-1
    assert num_chains in [4,5]

    nres_mhc, nres_pmhc, _, nres = cbs[-4:]

    burier_mask = np.zeros((nres,), dtype=bool)
    burier_mask[:nres_mhc] = True

    burials = compute_burials(pose, burier_mask=burier_mask)

    seq = pose['sequence']

    total = 0.
    for ii, (start,stop) in enumerate(tdinfo.tcr_cdrs):
        if ii%4==3:
            continue # ignore cdr3
        for pos in range(start, stop+1):
            if seq[pos] in strength_aas:
                total += burials[pos]
    return total


def get_mhc_buried_strengths(pose, tdinfo, designable_positions=None):
    ''' returns dict indexed by 'cdrs', 'vgene', and (optional) 'designable'
    'vgene' is really just cdr1,2, and 2.5
    '''
    
    cbs = [0] + list(it.accumulate(len(x) for x in pose['chainseq'].split('/')))
    num_chains = len(cbs)-1
    assert num_chains in [4,5]

    nres_mhc, nres_pmhc, _, nres = cbs[-4:]

    burier_mask = np.zeros((nres,), dtype=bool)
    burier_mask[:nres_mhc] = True

    burials = compute_burials(pose, burier_mask=burier_mask)

    seq = pose['sequence']

    totals = {'cdrs':0., 'vgene': 0.}
    
    for ii, (start,stop) in enumerate(tdinfo.tcr_cdrs):
        for pos in range(start, stop+1):
            if seq[pos] in strength_aas:
                totals['cdrs'] += burials[pos]
                if ii%4 != 3:
                    totals['vgene'] += burials[pos]

    if designable_positions:
        totals['designable'] = sum(burials[x] for x in designable_positions
                                   if seq[x] in strength_aas)
        
    return totals




def get_designable_from_tdinfo(
        tdinfo,
        which_cdrs=[0,1,3,4,5,7],
        ntrim_cdr3=3,
        ctrim_cdr3=2,
):
    designable_positions = []
    for ii, loop in enumerate(tdinfo.tcr_cdrs):
        if ii not in which_cdrs:
            continue
        if ii%4==3: # cdr3
            npad, cpad = ntrim_cdr3, ctrim_cdr3
        else: # cdr1 or cdr2
            npad, cpad = 0, 0
        designable_positions.extend(range(loop[0]+npad, loop[1]+1-cpad))
    return designable_positions




######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88

if __name__ == '__main__':
    results = pd.read_table('/home/pbradley/csdat/tcrpepmhc/amir/run673b_results.tsv')

    results = compute_af3_stats(results.head(10))


    


    exit()

    
if __name__ == '__main__':
    # testing
    tsvfile = ('/home/pbradley/csdat/tcrpepmhc/amir/'
               'run409_results.tsv')
    targets = pd.read_table(tsvfile).head(3)

    targets = compute_simple_stats(targets, trust_tcr_positions=True,
                                   compare_to_old_dgeom=True)
    #targets = compute_simple_stats(targets)
    outfile = 'tmp.tsv'
    targets.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

    sys.exit()


if __name__ == '__main__':
    tsvfile = '/home/pbradley/csdat/tcrpepmhc/amir/run376_results_top_20.tsv'

    results = pd.read_table(tsvfile)

    results = reorient_and_rechain_models(results, keep_old_pdbs=True)

    exit()


if __name__ == '__main__':
    tsvfile = ('/home/pbradley/csdat/tcrpepmhc/amir/rf_ab_diff_test1/'
               'run7_results.tsv')
    targets = pd.read_table(tsvfile)

    targets = targets.head(10)

    common_cols = 'organism mhc_class mhc va ja cdr3a vb jb cdr3b'.split()

    rfdiff_targets = targets[common_cols+'model_pdbfile chainseq'.split()]
    af2_targets = targets[common_cols+'af2_model_pdbfile af2_chainseq'.split()]\
                  .rename(columns={'af2_model_pdbfile':'model_pdbfile',
                                   'af2_chainseq':'chainseq'})
    rf2_targets = targets[common_cols+'rf2_model_pdbfile rf2_chainseq'.split()]\
                  .rename(columns={'rf2_model_pdbfile':'model_pdbfile',
                                   'rf2_chainseq':'chainseq'})
    df = compare_models(rfdiff_targets, af2_targets)
    #df = compare_models(rfdiff_targets, rf2_targets)
    #df = compare_models(af2_targets, rf2_targets)




    exit()

if __name__ == '__main__':
    # testing

    print('hello world')

    tsvfile = ('/home/pbradley/csdat/tcrpepmhc/amir/'
               'run368run369_results_design_scores.tsv')
    targets = pd.read_table(tsvfile)

    targets = targets.head(10)

    targets = compute_simple_stats(targets)

    targets = compute_simple_stats(targets)

    if 0:
        for _,l in targets.iterrows():
            tdinfo = get_model_tdinfo(
                l.organism, l.mhc_class, l.mhc, l.chainseq,
                l.va, l.ja, l.cdr3a, l.vb, l.jb, l.cdr3b,
            )

            print(tdinfo)
