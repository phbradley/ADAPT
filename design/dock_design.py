######################################################################################88

import argparse
parser = argparse.ArgumentParser(
    description="alphafold dock design",
    epilog='''
Flexible-dock tcr design using the "random loops" approach

input is a TSV file with peptide-MHC target info (allele name, peptide seq)

The script repeats the following steps '--num_designs' times:

STEP 1. pick a random pmhc target from list

STEP 2. pick a template tcr for that pmhc target. By default this will be a tcr that
binds to the same MHC allele. This template contributes the following information:
* va, ja, vb, jb (the V and J genes for the alpha and beta chains)
* the first 3 and last 2 residues of the CDR3 loops

    --tcr_pdbids <pdbid1> ... <pdbidN> will instead pick templates from the given pdbids

    --allow_mhc_mismatch will expand the set of potential templates to include all
tcrs that bind the same MHC class

STEP 3. pick CDR3 loops from a random paired TCR. Mutate the first 3 and last 2 aas
to match the template tcr from STEP 2.

STEP 4. Provide this information (peptide,MHC,V/J genes, CDR3 sequences) to a
modified alphafold TCR docking protocol

STEP 5. Re-design the CDR3 loops (excluding first 3 and last 2 residues) using MPNN

STEP 6. Re-dock the TCR to the pMHC using the same alphafold docking protocol used
in step 4.

STEP 7. Compute final stats like pmhc_tcr_pae, peptide_loop_pae, and dock-rmsd between
first and second alphafold models.



Example command line:

python dock_design.py --pmhc_targets my_pmhc_targets.tsv \\
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

parser.add_argument('--num_designs', type=int, required=True)
parser.add_argument('--pmhc_targets', required=True)
parser.add_argument('--resample', action='store_true')
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--tcr_pdbids', nargs='*')
parser.add_argument('--allow_mhc_mismatch', action='store_true')
parser.add_argument('--design_other_cdrs', action='store_true')
parser.add_argument('--design_cdrs', type=int, nargs='*')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--use_pmhc_pdbfile', action='store_true')
parser.add_argument('--use_native_dgeoms_in_round1', action='store_true')
parser.add_argument('--use_native_dgeoms_in_round2', action='store_true')
parser.add_argument('--extend_cdr2', action='store_true')
parser.add_argument('--num_mpnn_seqs', type=int)
parser.add_argument('--weak_mpnn', action='store_true')
parser.add_argument('--pepspec_mpnn', action='store_true')
parser.add_argument('--pepspec_opt_mpnn', action='store_true')
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--random_state', type=int)
parser.add_argument('--skip_rf_antibody', action='store_true',
                    help='dont run rf_antibody to evaluate the designs')

parser.add_argument('--model_name', default='model_2_ptm_ft_binder',
                    help='this doesnt really matter but it has to start with '
                    '"model_2_ptm_ft"')

parser.add_argument('--model_params_file',
                    help='The default is a binder-fine-tuned model that was trained '
                    'on structures and a new distillation set')
#parser.add_argument('--rundir')
#parser.add_argument('--loop_design_extra_args', default = '')
#parser.add_argument('--random_state',type=int, default=11)

args = parser.parse_args()


# other imports
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import pandas as pd
from os.path import exists
from os import mkdir, makedirs
import random
from collections import Counter
from timeit import default_timer as timer
import itertools as it

import design_stats
import wrapper_tools

wrapper_tools.show_hostname()

if args.model_params_file is None:
    args.model_params_file = design_paths.AF2_BINDER_FT_PARAMS

## hard-coded -- these control how much sequence is retained from tcr template cdr3s
nterm_seq_stem = 3
cterm_seq_stem = 2
## defaults ##########


# read the targets
pmhc_targets = pd.read_table(args.pmhc_targets)
required_cols = 'organism mhc_class mhc peptide'.split()
if args.resample:
    required_cols += 'cdr3a cdr3b tcr_template_pdbid'.split()

for col in required_cols:
    assert col in pmhc_targets.columns, f'Need {col} in --pmhc_targets'

if args.resample: # drop other design-related columns
    pmhc_targets = pmhc_targets[required_cols].copy()
    assert all(pmhc_targets.tcr_template_pdbid.str.len()==4) # pdbid
    pmhc_targets['dd_lineage'] = [f'{args.pmhc_targets}:{x}'
                                  for x in range(pmhc_targets.shape[0])]

if args.use_pmhc_pdbfile:
    # Need pdbfile column and also pdbid column (used for tcrdock internal mapping junk)
    # can't match any existing pdbids
    assert 'pdbfile' in pmhc_targets.columns and 'pdbid' in pmhc_targets.columns
    pdbids = set(pmhc_targets.pdbid)
    assert not any(x in td2.sequtil.ternary_info.index for x in pdbids)
    assert not any(x in td2.sequtil.new_ternary_info.index for x in pdbids)

# read the big paired tcr database, this provides the random cdr3a/cdr3b pairs
tcrs_file = design_paths.PAIRED_TCR_DB
print('reading:', tcrs_file)
big_tcrs_df = pd.read_table(tcrs_file)

# exclude extreme len cdr3s
badmask = ((big_tcrs_df.cdr3a.str.len()<9)|
           (big_tcrs_df.cdr3b.str.len()<9)|
           (big_tcrs_df.cdr3a.str.len()>17)|
           (big_tcrs_df.cdr3b.str.len()>17))
big_tcrs_df = big_tcrs_df[~badmask]

targets_dfl = []

# sample --num_designs pmhcs and cdr3a/b pairs
pmhcs = pmhc_targets.sample(n=args.num_designs, replace=True,
                            random_state=args.random_state)

# cdr3s is unused if args.resample:
cdr3s = big_tcrs_df.sample(n=args.num_designs, replace=True,
                           random_state=args.random_state)


outdir = f'{args.outfile_prefix}_tmp/'
if not exists(outdir):
    mkdir(outdir)

if args.use_pmhc_pdbfile: # add in some extra templates for af2 modeling
    extra_pmhc_templates = wrapper_tools.setup_extra_pmhc_templates(
        pmhcs, outdir+'pmhc_tmpl_')
else:
    extra_pmhc_templates = None

dfl = []
for (_,lpmhc), lcdr3 in zip(pmhcs.iterrows(), cdr3s.itertuples()):
    assert lpmhc.mhc_class==1 # for the time being...
    mhc = ':'.join(lpmhc.mhc.split(':')[:2]) # trim allele beyond 2-digit

    # look for tcrs with the same allele
    templates = pd.concat([
        td2.sequtil.ternary_info, td2.sequtil.new_ternary_info])

    # need templates that are also in tcr-info
    tcr_chains = pd.concat([td2.sequtil.tcr_info, td2.sequtil.new_tcr_info]).index
    a_pdbids = set(x[0] for x in tcr_chains if x[1] == 'A')
    b_pdbids = set(x[0] for x in tcr_chains if x[1] == 'B')
    goodmask = templates.pdbid.isin(a_pdbids&b_pdbids)
    print('subset to good tcr pdbids:', goodmask.sum(), templates.shape[0])
    templates = templates[goodmask]

    templates = templates[templates.organism == lpmhc.organism].copy()

    if args.resample: # NOTE: we will change lcdr3 inside here
        # use same tcr template pdbid
        # use cdr3a and cdr3b loops with the same length as resampled design
        cdr3alen = len(lpmhc.cdr3a)
        cdr3blen = len(lpmhc.cdr3b)
        mask = ((big_tcrs_df.cdr3a.str.len() == cdr3alen)&
                (big_tcrs_df.cdr3b.str.len() == cdr3blen))
        assert mask.any()
        lcdr3 = big_tcrs_df[mask].sample(n=1).iloc[0]
        ltcr = templates.loc[lpmhc.tcr_template_pdbid]
        print(cdr3alen, cdr3blen, lcdr3.cdr3a, lcdr3.cdr3b)
    else:
        if args.tcr_pdbids: # fix this
            templates_pdbids = set(templates.pdbid)
            resample_pdbids = [x for x in args.tcr_pdbids if x in templates_pdbids]
            bad_pdbids  = [x for x in args.tcr_pdbids if x not in templates_pdbids]
            if bad_pdbids:
                print(f'WARNING --tcr_pdbids contains bad pdbids: {bad_pdbids}!!!')
            assert resample_pdbids, 'no valid --tcr_pdbids'
            template_pdbid = random.choice(resample_pdbids)
            print(f'randomly chose {template_pdbid} from {resample_pdbids}')
            templates = templates[templates.pdbid==template_pdbid]
        elif args.allow_mhc_mismatch: # only enforce same mhc_class
            templates = templates[templates.mhc_class==lpmhc.mhc_class]
        else: # require same mhc class and same mhc allele
            templates = templates[(templates.mhc_class==lpmhc.mhc_class)&
                                  (templates.mhc_allele.str.startswith(mhc))]
        if templates.shape[0] == 0:
            print('ERROR no matching templates found for mhc:', mhc)
            exit(1)
        ltcr = templates.sample(n=1).iloc[0]

    outl = lpmhc.copy()
    cdr3a = lcdr3.cdr3a
    cdr3b = lcdr3.cdr3b
    # preserve 1st 3 and last 2 cdr3 rsds from template tcr (set by n/cterm_seq_stem)
    cdr3a = (ltcr.cdr3a[:nterm_seq_stem] +
             cdr3a[nterm_seq_stem:-cterm_seq_stem] +
             ltcr.cdr3a[-cterm_seq_stem:])
    cdr3b = (ltcr.cdr3b[:nterm_seq_stem] +
             cdr3b[nterm_seq_stem:-cterm_seq_stem] +
             ltcr.cdr3b[-cterm_seq_stem:])
    outl['va'] = ltcr.va
    outl['ja'] = ltcr.ja
    outl['cdr3a'] = cdr3a
    outl['vb'] = ltcr.vb
    outl['jb'] = ltcr.jb
    outl['cdr3b'] = cdr3b
    outl['tcr_template_pdbid'] = ltcr.pdbid
    if args.use_pmhc_pdbfile:
        outl['pmhc_template_pdbid'] = lpmhc.pdbid
    dfl.append(outl)

tcrs = pd.DataFrame(dfl)


targets = td2.sequtil.setup_for_alphafold(
    tcrs, outdir, num_runs=1, use_opt_dgeoms=True, clobber=True,
    force_tcr_pdbids_column='tcr_template_pdbid',
    force_pmhc_pdbids_column='pmhc_template_pdbid' if args.use_pmhc_pdbfile else None,
    extra_pmhc_templates=extra_pmhc_templates,
    use_new_templates=True,
    use_same_pmhc_dgeoms=args.use_native_dgeoms_in_round1,
)

if args.use_native_dgeoms_in_round1 and not args.use_native_dgeoms_in_round2:
    round2_outdir = f'{args.outfile_prefix}_tmp_rnd2/'
    makedirs(round2_outdir, exist_ok=True)
    round2_targets = td2.sequtil.setup_for_alphafold(
        tcrs, round2_outdir, num_runs=1, use_opt_dgeoms=True, clobber=True,
        force_tcr_pdbids_column='tcr_template_pdbid',
        force_pmhc_pdbids_column='pmhc_template_pdbid' if args.use_pmhc_pdbfile else None,
        extra_pmhc_templates=extra_pmhc_templates,
        use_new_templates=True,
    )
    round2_alignfiles = list(round2_targets.templates_alignfile)



if args.design_other_cdrs:
    which_cdrs = [0,1,3,4,5,7]
elif args.design_cdrs:
    which_cdrs = args.design_cdrs[:]
else:
    which_cdrs = [3,7]


targets.rename(columns={'target_chainseq':'chainseq',
                        'templates_alignfile':'alignfile'}, inplace=True)
dfl = []
for l in targets.itertuples():
    posl = []
    tdinfo = design_stats.get_row_tdinfo(l)
    for ii in which_cdrs:
        loop = tdinfo.tcr_cdrs[ii]
        npad, cpad = (nterm_seq_stem, cterm_seq_stem) if ii in [3,7] else \
                     (0,0)
        if args.extend_cdr2:
            if ii==1: # cdr2a
                npad, cpad = -1,  0 # negative makes loop longer
                print('extending cdr2a:', npad, cpad)
            elif ii==5: # cdr2b
                npad, cpad = -1, -2
                print('extending cdr2b:', npad, cpad)
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
nseq = (args.num_mpnn_seqs if args.num_mpnn_seqs is not None else
        30 if args.weak_mpnn else
        20 if args.pepspec_mpnn else
        5 if args.pepspec_opt_mpnn else
        3)
targets = wrapper_tools.run_mpnn(
    targets,
    outprefix,
    num_mpnn_seqs=nseq,
    extend_flex='barf',
    dry_run=args.debug,
    sort_by_strength=args.weak_mpnn,
    sort_by_peptide_probs=args.pepspec_mpnn or args.pepspec_opt_mpnn,
    pepspec_optimize=args.pepspec_opt_mpnn,
    pepspec_weight=20.0,
)
mpnn_time = timer()-start

# run alphafold again
outprefix = f'{outdir}afold2'
if args.use_native_dgeoms_in_round1 and not args.use_native_dgeoms_in_round2:
    targets['alignfile'] = round2_alignfiles

start = timer()
targets = wrapper_tools.run_alphafold(
    targets, outprefix,
    num_recycle=args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    dry_run = args.debug,
    ignore_identities = True, # since mpnn changed sequence...
)
af2_time += timer()-start

# compute stats again. this should compute docking rmsd to the original mpnn dock
targets = design_stats.compute_simple_stats(targets, extend_flex='barf')

# compute buried strength against mhc
dfl = []
for _,row in targets.iterrows():
    pose = td2.pdblite.pose_from_pdb(row.model_pdbfile)
    cbs = [0] + list(it.accumulate(len(x) for x in row.chainseq.split('/')))
    pose = td2.pdblite.set_chainbounds_and_renumber(pose, cbs)
    tdinfo = design_stats.get_row_tdinfo(row, trust_tcr_positions=True)
    des_posl = sorted([int(x) for x in row.designable_positions.split(',')])

    burials = design_stats.get_mhc_buried_strengths(pose, tdinfo, designable_positions=
                                                    des_posl)
    dfl.append(burials)
    burial_types = list(burials.keys())

for btype in burial_types:
    targets[f'mhc_{btype}_buried_strength'] = [x[btype] for x in dfl]

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
        targets, outprefix, delete_old_results=True) # in case of restart for partial
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

    targets['pepspec_combo'] = targets.combo_score_wtd - 2*targets.pepspec_delta

# write results
targets['af2_time'] = af2_time/args.num_designs
targets['mpnn_time'] = mpnn_time/args.num_designs

targets = design_stats.reorient_and_rechain_models(targets)

targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')
