from .util import long2short
from . import tcrdist
import numpy as np
import itertools as it
from sys import exit
from collections import Counter, OrderedDict
import copy

long2short_MSE = dict(**long2short, MSE='M')
long2short_plus = dict(**long2short_MSE, DA='a', DC='c', DG='g', DT='t', SEP='s')
long2short_plus.update({' DA':'a', ' DC':'c', ' DG':'g', ' DT':'t'})

short_to_long_plus = dict(**tcrdist.amino_acids.short_to_long,
                          a=' DA', c=' DC', g=' DG', t=' DT',
                          s='SEP')

dna_name1s = ['a','c','g','t']
protein_name1s = [x for x in short_to_long_plus if x not in dna_name1s]
print('protein_name1s:', protein_name1s)


def load_pdb_coords(
        pdbfile,
        allow_chainbreaks=False,
        allow_skipped_lines=False,
        verbose=False,
        preserve_atom_name_whitespace=False,
        require_CA=False,
        require_bb=False,
        ignore_altloc=True,
        force_MODEL=None,
):
    ''' returns: chains, all_resids, all_coords, all_name1s
    '''

    chains = []
    all_resids = {}
    all_coords = {}
    all_name1s = {}

    if verbose:
        print('reading:', pdbfile)
    skipped_lines = False
    in_model = force_MODEL is None or force_MODEL is False
    if force_MODEL:
        assert type(force_MODEL) is int

    with open(pdbfile,'r') as data:
        for line in data:
            if line[:5] == 'MODEL' and not in_model:
                model_num = int(line.split()[1])
                in_model = (model_num == force_MODEL)
                if in_model:
                    print('FOUND the right MODEL:', model_num)
            if not in_model:
                continue
            if line[:6] == 'ENDMDL' and in_model:
                #print('stopping ENDMDL:', pdbfile)
                break
            if (line[:6] in ['ATOM  ','HETATM'] and line[17:20] != 'HOH' and
                (ignore_altloc or line[16] in ' A1')):
                if line[17:20] in long2short_plus:
                    resid = line[22:27]
                    chain = line[21]
                    if chain not in all_resids:
                        all_resids[chain] = []
                        all_coords[chain] = {}
                        all_name1s[chain] = {}
                        chains.append(chain)
                    if line.startswith('HETATM') and line[12:16] == ' CA ':
                        print('WARNING: HETATM', pdbfile, line[:-1])
                    if preserve_atom_name_whitespace:
                        atom = line[12:16]
                    else:
                        atom = line[12:16].strip()
                    if resid not in all_resids[chain]:
                        all_resids[chain].append(resid)
                        all_coords[chain][resid] = OrderedDict()
                        all_name1s[chain][resid] = long2short_plus[line[17:20]]

                    if atom in all_coords[chain][resid]:
                        if verbose:
                            print('WARNING: take first xyz for atom, ignore others:',
                                  chain, resid, atom, 'altloc:', line[16],
                                  pdbfile)
                    else:
                        all_coords[chain][resid][atom] = np.array(
                            [float(line[30:38]),float(line[38:46]), float(line[46:54])])
                else:
                    if verbose or line[12:16] == ' CA ':
                        print('skip ATOM line:', line[:-1], pdbfile)
                    skipped_lines = True

    # possibly subset to residues with CA
    if preserve_atom_name_whitespace:
        N, CA, C  = ' N  ', ' CA ', ' C  '
        C1p = " C1'"
    else:
        N, CA, C = 'N', 'CA', 'C'
        C1p = "C1'"

    require_atoms = [N,CA,C] if require_bb else [CA] if require_CA else []
    if require_atoms:
        chains = all_resids.keys()
        for chain in chains:
            bad_resids = [x for x,y in all_coords[chain].items()
                          if ((any(a not in y for a in require_atoms) and
                               all_name1s[chain][x] in protein_name1s) or
                              (C1p not in y and all_name1s[chain][x] in dna_name1s))]
            if bad_resids:
                print('missing one of', require_atoms, bad_resids)
                for r in bad_resids:
                    all_resids[chain].remove(r)
                    del all_coords[chain][r]
                    del all_name1s[chain][r]


    # check for chainbreaks
    maxdis = 1.75
    for chain in chains:
        for res1, res2 in zip(all_resids[chain][:-1], all_resids[chain][1:]):
            coords1 = all_coords[chain][res1]
            coords2 = all_coords[chain][res2]
            if C in coords1 and N in coords2:
                dis = np.sqrt(np.sum(np.square(coords1[C]-coords2[N])))
                if dis>maxdis:
                    if verbose or not allow_chainbreaks:
                        print('ERROR chainbreak:', chain, res1, res2, dis, pdbfile)
                    if not allow_chainbreaks:
                        print('STOP: chainbreaks', pdbfile)
                        #print('DONE')
                        exit()

    if skipped_lines and not allow_skipped_lines:
        print('STOP: skipped lines:', pdbfile)
        #print('DONE')
        exit()

    return chains, all_resids, all_coords, all_name1s

def load_pdb_coords_resids(
        pdbfile,
        **kwargs,
        #allow_chainbreaks=False,
        #allow_skipped_lines=False,
        #verbose=False,
):
    ''' returns: resids, coords, sequence

    resids is a list of (chain, resid) tuples

    coords is a dict indexed by (chain, resid)

    sequence is the full sequence, as a string
    '''

    chains, all_resids, all_coords, all_name1s = load_pdb_coords(
        pdbfile, **kwargs)#allow_chainbreaks, allow_skipped_lines, verbose)

    resids = list(it.chain(*[[(c,r) for r in all_resids[c]]
                               for c in chains]))
    coords = {(c,r):all_coords[c][r]
              for c in chains
              for r in all_resids[c]}

    sequence = ''.join(all_name1s[c][r] for c,r in resids)

    return resids, coords, sequence


def pose_from_pdb(filename, **kwargs):
    ''' calls update_derived_data(pose) before returning pose
    '''
    defaults = dict(
        allow_chainbreaks=True,
        allow_skipped_lines=True,
        preserve_atom_name_whitespace=True,
        require_CA=True,
        require_bb=True,
    )
    kwargs = {**defaults, **kwargs}
    resids, coords, sequence = load_pdb_coords_resids(filename, **kwargs)
    pose = {'resids':resids, 'coords':coords, 'sequence':sequence}
    pose = update_derived_data(pose)

    return pose


def save_pdb_coords(
        outfile,
        resids,
        coords,
        sequence,
        verbose=False,
        bfactors=None,
        out=None,
):

    ''' right now bfactors is a list of length = resids (all atoms in res have same)
    '''
    assert len(sequence) == len(resids)
    if out is None:
        close_file = True
        out = open(outfile, 'w')
    else:
        close_file = False
    last_chain = None
    counter=0
    if bfactors is None:
        bfactors = [50.0]*len(resids)
    if len(bfactors) == len(resids):
        atom_bfactors = False
    else:
        atom_bfactors = True
        # count atoms
        num_atoms = sum(len(coords[x].keys()) for x in resids)
        assert len(bfactors) == num_atoms
    for ind, (cr, name1) in enumerate(zip(resids, sequence)):
        (chain,resid) = cr
        name3 = short_to_long_plus[name1]
        if chain != last_chain and last_chain is not None:
            out.write('TER\n')
        last_chain = chain
        for atom,xyz in coords[cr].items():
            counter += 1
            a = atom.strip()
            assert len(atom) == 4
            assert len(resid) == 5
            element = 'H' if a[0].isdigit() else a[0]
            occ=1.
            if atom_bfactors:
                bfac = bfactors[counter-1]
            else:
                bfac = bfactors[ind]
            #                   6:12      12:16   17:20    21    22:27
            # oops screwed up the atom number position! bugfix: 2023-09-01
            outline = (f'ATOM  {counter%100000:5d} {atom} {name3} {chain}{resid}   '
                       f'{xyz[0]:8.2f}{xyz[1]:8.2f}{xyz[2]:8.2f}{occ:6.2f}{bfac:6.2f}'
                       f'{element:>12s}\n')
            assert (outline[12:16] == atom and outline[17:20] == name3 and
                    outline[22:27] == resid) # confirm register
            out.write(outline)
    if close_file:
        out.close()
    if verbose:
        print('made:', outfile)


def dump_pdb(pose, outfile, out=None, bfactors=None):
    save_pdb_coords(outfile, pose['resids'], pose['coords'], pose['sequence'],
                    out=out, bfactors=bfactors)

def dump_cif(pose, outfile, name='model', out=None, bfactors=None, date='2001-01-01'):
    atoms_header = '''loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
'''
#ATOM 1    N N   . LYS B 1 29  ? 37.148 8.654   -41.002 1.00 30.40 26  A 1
#ATOM 2    C CA  . LYS B 1 29  ? 37.020 10.035  -40.455 1.00 31.13 26  A 1
#ATOM 3    C C   . LYS B 1 29  ? 35.904 10.150  -39.435 1.00 30.99 26  A 1
#ATOM 4    O O   . LYS B 1 29  ? 36.153 10.160  -38.227 1.00 30.91 26  A 1
#ATOM 5    N N   . SER B 1 30  ? 34.670 10.232  -39.922 1.00 31.11 27  A 1
    if out is None:
        close_file = True
        out = open(outfile, 'w')
    else:
        close_file = False

    counter=0
    if bfactors is None:
        bfactors = it.repeat(50.0)
    else:
        assert len(bfactors) == len(resids)
    chains = [] # make ordered list of chains for defining entities, not quite right
    for (chain,_) in pose['resids']:
        if chain not in chains:
            chains.append(chain)

    out.write(f'data_{name}\n')
    out.write(f'''#
_pdbx_audit_revision_history.revision_date {date}
#
_pdbx_database_status.recvd_initial_deposition_date {date}
''')

    out.write(atoms_header)

    for cr, name1, bfac in zip(pose['resids'], pose['sequence'], bfactors):
        (chain,resid) = cr
        resnum = int(resid[:-1])
        ins_code = '?' if resid[-1] == ' ' else resid[-1]
        entity = chains.index(chain)
        name3 = short_to_long_plus[name1]
        last_chain = chain
        for atom,xyz in pose['coords'][cr].items():
            counter += 1
            a = atom.strip()
            assert len(atom) == 4
            assert len(resid) == 5
            element = 'H' if a[0].isdigit() else a[0]
            occ=1.
            pdb_model_num = 1
            outline = (f'ATOM {counter} {element} {a} . {name3} {chain} {entity} '
                       f'{resnum} {ins_code} '
                       f'{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f} {occ:.2f} {bfac:.2f} '
                       f'{resnum} {chain} {pdb_model_num}\n')

            out.write(outline)
    if close_file:
        out.close()





def check_coords_shape(pose):
    coords = pose['coords']
    for r in pose['resids']:
        for a,xyz in coords[r].items():
            if xyz.shape != (3,):
                print('check_coords_shape: FAIL:', r, a, xyz)
                assert False

def update_derived_data(pose):
    ''' pose is a python dict with keys

    'resids' - list of (chain,resid) tuples where resid is the pdb resid: line[22:27]
    'coords' - dict indexed by (chain,resid) tuples mapping to dicts from atom-->xyz
    'sequence' - string, sequence in 1-letter code

    invariants:

    len(resids) == len(sequence)
    set(coords.keys()) == set(resids)
    each chain id occurs in contiguous block of resids
    each (chain,resid) has N,CA,C, names = ' N  ', ' CA ', ' C  '

    sets up derived data:

    'ca_coords'
    'chains'
    'chainseq'
    'chainbounds'

    '''
    #check_coords_shape(pose)

    CA = ' CA '
    C1p = " C1'"

    resids, coords, sequence = pose['resids'], pose['coords'], pose['sequence']

    chains = [x[0] for x in it.groupby(resids, lambda x:x[0])]
    assert len(set(chains)) == len(chains) # not interspersed: each chain comes once

    chainseq = {c:'' for c in chains}
    for r,a in zip(resids, sequence):
        chainseq[r[0]] += a

    chain_lens = [len(chainseq[c]) for c in chains]
    chainbounds = [0] + list(it.accumulate(chain_lens))

    chainseq = '/'.join(chainseq[c] for c in chains)

    assert len(chainbounds) == len(chains) + 1 # 0 at the beginning, N at the end
    if resids:
        seqatoms = [C1p if x in dna_name1s else CA for x in sequence]
        pose['ca_coords'] = np.stack([coords[r][a] for r,a in zip(resids,seqatoms)])
    else: # empty
        pose['ca_coords'] = []
    pose['chains'] = chains
    pose['chainseq'] = chainseq
    pose['chainbounds'] = chainbounds

    return pose

def renumber(pose):
    ''' set resids from 0 ---> N-1
    set chains from A,B,C,...Z
    '''

    resids, coords = pose['resids'], pose['coords']

    old_chains = pose['chains']
    new_chains = [chr(ord('A')+i) for i in range(len(old_chains))]

    old2new = dict(zip(old_chains, new_chains))

    new_resids = []
    new_coords = {}

    for ii,r in enumerate(resids):
        new_r = (old2new[r[0]], f'{ii:4d} ')
        new_resids.append(new_r)
        new_coords[new_r] = copy.deepcopy(coords[r])

    pose['resids'] = new_resids
    pose['coords'] = new_coords

    return update_derived_data(pose)

def set_chainbounds_and_renumber(pose, chainbounds):
    ''' set resids from 0 ---> N-1
    set chains from A,B,C,...Z
    '''

    assert chainbounds[0] == 0 and chainbounds[-1] == len(pose['sequence'])

    resids, coords = pose['resids'], pose['coords']

    num_chains = len(chainbounds)-1
    new_chains = [chr(ord('A')+i) for i in range(num_chains)]

    new_resids = []
    new_coords = {}

    for c, chain in enumerate(new_chains):
        assert chainbounds[c] < chainbounds[c+1]
        for ind in range(chainbounds[c], chainbounds[c+1]):
            new_r = (chain, f'{ind:4d} ')
            r = resids[ind]
            new_resids.append(new_r)
            new_coords[new_r] = copy.deepcopy(coords[r])

    assert len(new_resids) == len(resids) == len(new_coords.keys())

    pose['resids'] = new_resids
    pose['coords'] = new_coords

    return update_derived_data(pose)


def apply_transform_Rx_plus_v(pose, R, v):
    assert R.shape==(3,3) and v.shape==(3,)
    resids, coords = pose['resids'], pose['coords']
    for r in resids:
        coords[r] = {a:R@xyz + v for a,xyz in coords[r].items()}

    pose['coords'] = coords # probably not necessary

    # update since coords changed
    return update_derived_data(pose)

def delete_chains(pose, chain_nums):
    ''' doesn't necessarily copy all the old data
    '''
    del_chains = [pose['chains'][c] for c in chain_nums]

    resids, coords, sequence = pose['resids'], pose['coords'], pose['sequence']

    new_resids = []
    new_sequence = []
    for r,a in zip(resids, sequence):
        if r[0] in del_chains:
            del coords[r]
        else:
            new_resids.append(r)
            new_sequence.append(a)

    pose['resids'] = new_resids
    pose['sequence'] = ''.join(new_sequence)
    pose['coords'] = coords

    return update_derived_data(pose)

def append_chains(pose, src_pose, src_chain_nums):
    assert pose is not src_pose


    resids, coords = pose['resids'], pose['coords']
    src_resids, src_coords, src_sequence = (src_pose['resids'], src_pose['coords'],
                                            src_pose['sequence'])

    if not pose['chains']:
        ord0 = ord('A')
    else:
        ord0 = ord(max(pose['chains']))+1

    sequence = list(pose['sequence'])
    for ii, chain_num in enumerate(src_chain_nums):
        old_chain = src_pose['chains'][chain_num]
        new_chain = chr(ord0+ii)
        for r,a in zip(src_resids, src_sequence):
            if r[0] == old_chain:
                new_r = (new_chain, r[1])
                resids.append(new_r)
                sequence.append(a)
                coords[new_r] = copy.deepcopy(src_coords[r])
    pose['resids'] = resids
    pose['coords'] = coords
    pose['sequence'] = ''.join(sequence)

    return update_derived_data(pose)



def delete_residue_range(pose, start, stop):
    ''' returns a new pose (that might share some data)

    deletes start through stop, inclusive of start but NOT stop!!!!!!!!!!!!

    start and stop are 0-indexed

    '''

    newpose = {
        'coords': pose['coords'],
        'sequence': pose['sequence'][:start] + pose['sequence'][stop:],
        'resids': pose['resids'][:start] + pose['resids'][stop:],
    }

    return update_derived_data(newpose)


def find_chainbreaks(
        pose,
        maxdis = 1.75,
        verbose=False,
        return_total_chainbreak_by_chain=False,
):
    ''' assumes atom names have the usual PDB extra whitespace in them
    '''
    N, C  = ' N  ', ' C  '

    resids, coords = pose['resids'], pose['coords']
    chainbreaks = []
    total_chainbreak_by_chain = [0.]*len(pose['chains'])

    for i, r1 in enumerate(resids[:-1]):
        r2 = resids[i+1]
        if r1[0] == r2[0]:
            if C in coords[r1] and N in coords[r2]:
                dis = np.sqrt(np.sum(np.square(coords[r1][C]-coords[r2][N])))
                if dis >= maxdis:
                    chainbreaks.append(i)
                    ichain = pose['chains'].index(r1[0])
                    total_chainbreak_by_chain[ichain] += dis-maxdis
                    if verbose:
                        print('found intra-chain chainbreak:', r1, r2,
                              dis,'>',maxdis)
            else:
                if C not in coords[r1]:
                    print('missing C atom', r1)
                if N not in coords[r2]:
                    print('missing N atom', r2)

    if return_total_chainbreak_by_chain:
        return chainbreaks, total_chainbreak_by_chain
    else:
        return chainbreaks

def pose_from_cif(
        fname,
        require_bb=True,
        require_CA=True,
        use_author_chains=False,
):
    #ATOM   1    N N   . GLU A 1 4   ? -62.685  20.483 -39.874 1.000 111.067 ? 3   GLU AAA N   1

    data = open(fname,'r')

    atom_fields = []

    coords = {}
    resids = []
    name1s = {}

    achain2chain = {} # debug

    for line in data:
        if line.startswith('_atom_site.'):
            atom_fields.append(line.strip()[11:])
            #print('atom_field:', atom_fields[-1])
        elif line.startswith('ATOM') or line.startswith('HETATM'):
            l = line.split()
            assert len(l) <= len(atom_fields)
            if len(l) != len(atom_fields):
                nmiss = len(atom_fields)-len(l)
                print(f'pdblite::pose_from_cif partial ATOM line missing {nmiss}',
                      'fields:', atom_fields[-nmiss:])
                # may lead to error below if the missing fields are needed!
            info = dict(zip(atom_fields, l))
            atom_name = info['label_atom_id'] # no whitespace!!
            altloc = info['label_alt_id']
            resname = info['label_comp_id']
            if atom_name == 'HOH' or resname not in long2short_plus:
                continue
            if len(resname) == 2 and resname[0] == 'D':
                atom_name = atom_name.replace('"','')
            chain = info['label_asym_id']
            if use_author_chains:
                achain = info['auth_asym_id']
                if achain in achain2chain:
                    assert achain2chain[achain] == chain
                else:
                    achain2chain[achain] = chain
                chain = achain[0]
            chain_number_maybe = info['label_entity_id']
            insert = info['pdbx_PDB_ins_code']
            xyz = np.array([float(info['Cartn_'+x]) for x in 'xyz'])

            resnum = int(info['label_seq_id'])

            if insert in ['.','?']:
                resid = f'{resnum:4d} '
            else:
                assert len(insert) == 1
                resid = f'{resnum:4d}{insert}'

            if len(atom_name)<4:
                if atom_name[0].isdigit(): #hydrogen?
                    assert atom_name[1] == 'H'
                    lpad = ''
                else:
                    lpad = ' '
                atom_name = lpad+atom_name+' '*(4-len(lpad+atom_name))
            assert len(atom_name) == 4

            assert len(chain) == 1

            resid = (chain, resid)
            if resid not in resids:
                resids.append(resid)
                coords[resid] = OrderedDict()
                name1s[resid] = long2short_plus[resname]

            if atom_name in coords[resid]:
                print('WARNING: take first xyz for atom, ignore others:',
                      chain, resid, atom_name, 'altloc:', altloc, fname)
            else:
                coords[resid][atom_name] = xyz
    data.close()


    N, CA, C  = ' N  ', ' CA ', ' C  '
    C1p = " C1'"
    require_atoms = [N,CA,C] if require_bb else [CA] if require_CA else []
    if require_atoms:
        bad_resids = [x for x,y in coords.items()
                      if ((any(a not in y for a in require_atoms) and
                           name1s[x] in protein_name1s) or
                          (C1p not in y and name1s[x] in dna_name1s))]

        if bad_resids:
            print('missing one of', require_atoms, bad_resids)
            for r in bad_resids:
                resids.remove(r)

    sequence = ''.join(name1s[x] for x in resids)

    pose = {'resids':resids, 'coords':coords, 'sequence':sequence}

    pose = update_derived_data(pose)

    return pose

def make_empty_pose():
    pose = dict(resids=[], coords={}, sequence='')
    pose = update_derived_data(pose)
    return pose

