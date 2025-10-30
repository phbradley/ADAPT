######################################################################################88
''' Read TSV file with info on TCR:pMHC designs

* chainseq
* <--pdbfile_column>

'''


import argparse

parser = argparse.ArgumentParser(
    description="Evaluate alphafold designs with Rosetta")

parser.add_argument('--targets', nargs = '*', required=True)
parser.add_argument('--outfile', required=True)
parser.add_argument('--pdbfile_column', default='model_pdbfile')
parser.add_argument('--batch_num', type=int, default=0)
parser.add_argument('--num_batches', type=int, default=1)

parser.add_argument('--mute', action='store_true')
#parser.add_argument('--random_delay', type=int)
# parser.add_argument('--ex1', action='store_true')
# parser.add_argument('--ex2', action='store_true')
# parser.add_argument('--norelax', action='store_true')
# parser.add_argument('--beta_nov16_cart', action='store_true')
# parser.add_argument('--weights_filetag', default="ref2015_cart.wts")


args = parser.parse_args()

######################################################################################88

# if 'beta' in args.weights_filetag:
#     assert args.beta_nov16_cart

from os.path import exists

assert all(exists(x) for x in args.targets)

import pandas as pd

targets = pd.concat([pd.read_table(x) for x in args.targets]).reset_index(
    drop=True)

if args.num_batches>1:
    mask = targets.index%args.num_batches == args.batch_num
    print(f'subset to {mask.sum()} out of {mask.shape[0]}')
    targets = targets[mask].reset_index(drop=True)

required_cols = ['chainseq', args.pdbfile_column]

for col in required_cols:
    assert col in targets.columns, f'Need column {col} in targets TSV file'


### more imports
from timeit import default_timer as timer
import numpy as np
import sys
import pyrosetta
import itertools as it

from pyrosetta.rosetta import core, protocols, numeric, basic, utility
from pyrosetta.rosetta.protocols.sasa_scores import \
    compute_residue_sasas_for_sasa_scores

# local import s
from design_stats import get_designable_positions

# pyrosetta init
init_flags = '-ignore_unrecognized_res 1 -include_current -out:file:renumber_pdb'

# if args.random_delay is not None:
#     init_flags += f' -run:random_delay {args.random_delay}'
if args.mute:
    init_flags += f' -mute all'
# if args.ex1:
#     init_flags += f' -ex1'
# if args.ex2:
#     init_flags += f' -ex2'
# if args.beta_nov16_cart:
#     init_flags += f' -beta_nov16_cart'

if not args.mute:
    print('init_flags:', init_flags)

pyrosetta.init(init_flags)

################################################################################
# FUNCTIONS
################################################################################

# borrowed from pilot/brunette/tj_util.hh
# exposed_rsd_sasa = [0.0]*21
# exposed_rsd_sasa[  1]  = 170 # 1 A
# exposed_rsd_sasa[  2]  = 170 # 2 C
# exposed_rsd_sasa[  3]  = 210 # 3 D
# exposed_rsd_sasa[  4]  = 250 # 4 E
# exposed_rsd_sasa[  5]  = 290 # 5 F
# exposed_rsd_sasa[  6]  = 170 # 6 G
# exposed_rsd_sasa[  7]  = 220 # 7 H
# exposed_rsd_sasa[  8]  = 230 # 8 I
# exposed_rsd_sasa[  9]  = 260 # 9 K
# exposed_rsd_sasa[ 10]  = 230 # 10 L
# exposed_rsd_sasa[ 11]  = 240 # 11 M
# exposed_rsd_sasa[ 12]  = 190 # 12 N
# exposed_rsd_sasa[ 13]  = 220 # 13 P
# exposed_rsd_sasa[ 14]  = 220 # 14 Q
# exposed_rsd_sasa[ 15]  = 260 # 15 R
# exposed_rsd_sasa[ 16]  = 180 # 16 S
# exposed_rsd_sasa[ 17]  = 200 # 17 T
# exposed_rsd_sasa[ 18]  = 200 # 18 V
# exposed_rsd_sasa[ 19]  = 300 # 19 W
# exposed_rsd_sasa[ 20]  = 290 # 20 Y

def unbind_tcr(peptide_chain, pose, sep=50):
    ''' This function changes the pose so that the TCR is translated
    50 (sep) Angstroms away from the pMHC
    '''
    from pyrosetta.rosetta.numeric import xyzVector_double_t as Vector
    from pyrosetta.rosetta.numeric import xyzMatrix_double_t as Matrix
    posl = range(1,pose.size()+1)

    chains = np.array([pose.chain(x) for x in posl])
    calphas = np.array([pose.residue(x).xyz("CA") for x in posl])

    mhc_cen = np.mean(calphas[chains<peptide_chain], axis=0)
    tcr_cen = np.mean(calphas[chains>peptide_chain], axis=0)

    trans = tcr_cen - mhc_cen
    trans = Vector(*trans).normalized()
    trans *= sep

    pose2 = pose.clone()
    rotation = Matrix.I()
    pose2.apply_transform_Rx_plus_v(rotation, trans)

    for pos in range(pose.chain_begin(peptide_chain+1), pose.size()+1):
        pose.replace_residue(pos, pose2.residue(pos), False)




def read_alphafold_pose(filename):
    ''' This function reads an alphafold 3D model into the pose data structure.
    In Alphafold models from the TCR design pipeline, all the residues
    have the same chain identifier (A). But there are big gaps in the residue
    numbering between chains. This code looks for those gaps and increments the
    chain identifiers so that the individual chains (MHC,peptide,TCRa,TCRb) are
    reflected in the pose.

    '''
    pose = pyrosetta.pose_from_pdb(filename)

    pdbinfo = pose.pdb_info()

    # insert chainbreaks at residue numbering breaks
    for i in range(1,pose.size()):
        if pdbinfo.number(i+1) != pdbinfo.number(i)+1:
            pose.conformation().insert_chain_ending(i)
            core.pose.add_upper_terminus_type_to_pose_residue(pose, i)
            core.pose.add_lower_terminus_type_to_pose_residue(pose, i+1)

    return pose



def construct_ideal_cbeta(rsd):
    'you pass in a Rosetta residue object like from "pose.residue(i)"'
    from numpy.linalg import norm
    from numpy import dot, cross

    mean_cb_coords = np.array([-0.53529231, -0.76736402,  1.20778869])

    # convert from pyrosetta vectors to numpy vectors
    n  = np.array(rsd.xyz("N"))
    ca = np.array(rsd.xyz("CA"))
    c  = np.array(rsd.xyz("C"))

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

    # convert back to a pyrosetta vector
    new_cb = pyrosetta.rosetta.numeric.xyzVector_double_t(*new_cb)

    return new_cb


################################################################################
################################################################################


out_tsvfile = args.outfile

# scorefxn = pyrosetta.create_score_function(args.weights_filetag)
# eval_scorefxn = pyrosetta.create_score_function(args.weights_filetag)
# eval_scorefxn.set_weight(core.scoring.cart_bonded, 0.)


dfl = []

for ind, row in targets.iterrows():
    pose = read_alphafold_pose(row[args.pdbfile_column])
    cs = row.chainseq.split('/')
    cbs = [0] + list(it.accumulate(len(x) for x in cs))
    assert len(cs) == pose.num_chains()
    assert pose.num_chains() in [4,5]

    mhc_class = pose.num_chains()-3
    unbound_pose = pose.clone()
    peptide_chain = pose.num_chains()-2
    unbind_tcr(peptide_chain, unbound_pose)

    tcra_unbound_pose = unbound_pose.clone()
    tcrb_unbound_pose = unbound_pose.clone()


    mhc_end = pose.chain_end(mhc_class)
    pep_begin = pose.chain_begin(mhc_class+1)
    pep_end = pose.chain_end(mhc_class+1)
    tcr_begin = pose.chain_begin(mhc_class+2)
    tcrb_begin = pose.chain_begin(mhc_class+3)

    for i in range(tcr_begin, pose.size()+1):
        if i<tcrb_begin:
            tcrb_unbound_pose.replace_residue(i, pose.residue(i), False)
        else:
            tcra_unbound_pose.replace_residue(i, pose.residue(i), False)

    outl = row.copy()

    dis_thresholds = [9,8,7,6] # decreasing order!

    dis2_thresholds = [x**2 for x in dis_thresholds]
    atom_dis_threshold = 4.5
    atom_dis2_threshold = atom_dis_threshold**2
    hb_atom_dis_threshold = 3.3
    hb_atom_dis2_threshold = hb_atom_dis_threshold**2
    big_dis2 = (2*6.12 + atom_dis_threshold)**2

    mhc_contacts = [0]*len(dis2_thresholds)
    pep_contacts = [0]*len(dis2_thresholds)
    mhc_atom_contacts = 0
    pep_atom_contacts = 0
    tcra_pep_atom_contacts = 0
    tcrb_pep_atom_contacts = 0
    tcra_mhc_atom_contacts = 0
    tcrb_mhc_atom_contacts = 0
    pep_bb_hbonds = set()

    cbetas = [construct_ideal_cbeta(pose.residue(i))
              for i in range(1,pose.size()+1)]

    for i in range(tcr_begin, pose.size()+1):
        # the tcr position
        cb1 = cbetas[i-1] # annoying i is 1-indexed!!!
        rsd1 = pose.residue(i)

        for j in range(1, pep_end+1):
            # the pmhc position
            cb2 = cbetas[j-1]
            rsd2 = pose.residue(j)

            dis2 = cb1.distance_squared(cb2)
            for ii, dis2_threshold in enumerate(dis2_thresholds):
                if dis2 <= dis2_threshold:
                    if j <= mhc_end:
                        mhc_contacts[ii] += 1
                    else:
                        pep_contacts[ii] += 1
                else:
                    break

            if dis2<big_dis2:
                # look for atom-atom contacts
                for ii in range(1,rsd1.nheavyatoms()+1):
                    ii_xyz = rsd1.xyz(ii)
                    for jj in range(1,rsd2.nheavyatoms()+1):
                        atom_dis2 = ii_xyz.distance_squared(rsd2.xyz(jj))
                        if atom_dis2 <= atom_dis2_threshold:
                            if j <= mhc_end:
                                mhc_atom_contacts += 1
                                if i >= tcrb_begin:
                                    tcrb_mhc_atom_contacts += 1
                                else:
                                    tcra_mhc_atom_contacts += 1
                            else:
                                pep_atom_contacts += 1
                                if i >= tcrb_begin:
                                    tcrb_pep_atom_contacts += 1
                                else:
                                    tcra_pep_atom_contacts += 1
                                if (atom_dis2 <= hb_atom_dis2_threshold and
                                    rsd2.atom_is_backbone(jj) and
                                    (rsd1.atom_type(ii).is_donor() and
                                     rsd2.atom_type(jj).is_acceptor()) or
                                    (rsd1.atom_type(ii).is_acceptor() and
                                     rsd2.atom_type(jj).is_donor())):
                                    pep_bb_hbonds.add((j,jj))


    for dis, mhc, pep in zip(dis_thresholds, mhc_contacts, pep_contacts):
        outl[f'mhc_contacts_{dis}'] = mhc
        outl[f'pep_contacts_{dis}'] = pep

    outl[f'mhc_atom_contacts_{atom_dis_threshold}'] = mhc_atom_contacts
    outl[f'pep_atom_contacts_{atom_dis_threshold}'] = pep_atom_contacts
    outl[f'tcra_pep_atom_contacts_{atom_dis_threshold}'] = tcra_pep_atom_contacts
    outl[f'tcrb_pep_atom_contacts_{atom_dis_threshold}'] = tcrb_pep_atom_contacts
    outl[f'tcra_mhc_atom_contacts_{atom_dis_threshold}'] = tcra_mhc_atom_contacts
    outl[f'tcrb_mhc_atom_contacts_{atom_dis_threshold}'] = tcrb_mhc_atom_contacts
    outl[f'pep_bb_hbonds'] = len(pep_bb_hbonds)

    # compute delta sasa
    for prefix, ub_pose in [['',unbound_pose],
                            ['tcra_',tcra_unbound_pose],
                            ['tcrb_',tcrb_unbound_pose]]:
        for probe in [0.5, 1.0, 1.4]:
            bound_sasas = pyrosetta.rosetta.utility.vector1_double()
            unbound_sasas = pyrosetta.rosetta.utility.vector1_double()
            compute_residue_sasas_for_sasa_scores(probe, pose, bound_sasas)
            compute_residue_sasas_for_sasa_scores(probe, ub_pose, unbound_sasas)

            bound_sasas = np.array(bound_sasas)
            unbound_sasas = np.array(unbound_sasas) # now 0-indexed !

            mhc_bsasa = unbound_sasas[:mhc_end].sum() - bound_sasas[:mhc_end].sum()
            pep_sasa_unbound = unbound_sasas[mhc_end:pep_end].sum()
            pep_sasa_bound = bound_sasas[mhc_end:pep_end].sum()

            outl[f'{prefix}mhc_bsasa_{probe}'] = mhc_bsasa
            outl[f'{prefix}pep_sasa_unbound_{probe}'] = pep_sasa_unbound
            outl[f'{prefix}pep_sasa_bound_{probe}'] = pep_sasa_bound
            outl[f'{prefix}pep_bsasa_{probe}'] = pep_sasa_unbound - pep_sasa_bound
            outl[f'{prefix}pep_bsasa_frac_{probe}'] = (
                1.0 - pep_sasa_bound/pep_sasa_unbound)

    #print(outl)
    dfl.append(outl)

    if not args.mute:
        pd.DataFrame(dfl).to_csv(args.outfile, sep='\t', index=False)
        print('made temporary:', args.outfile)


pd.DataFrame(dfl).to_csv(args.outfile, sep='\t', index=False)


print('DONE')
