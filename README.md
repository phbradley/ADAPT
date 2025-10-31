# ADAPT

## Antigen-receptor Design Against Peptide-MHC Targets

This repository contains Python code implementing the ADAPT antigen-receptor design
pipeline. This pipeline is described in the bioRxiv preprint

[insert link]

## here's the abstract

Class I major histocompatibility complexes (MHCs), expressed on the surface of all nucleated cells, present peptides derived from intracellular proteins for surveillance by T cells.
The precise recognition of foreign or mutated peptide–MHC (pMHC) complexes by T cell receptors (TCRs) is fundamental to immune defense against pathogens and tumors.
Although patient-derived TCRs specific for cancer-associated antigens have been used to engineer tumor-targeting therapies, their reactivity toward self or near-self antigens is often limited by negative selection in the thymus.
Here, we introduce a structure-based deep learning framework, ADAPT (Antigen-receptor Design Against Peptide-MHC Targets), for designing TCRs and antibodies that bind defined pMHC targets.
We validated the ADAPT pipeline by designing and characterizing TCRs and antibodies against a diverse panel of pMHCs.
Cryogenic electron microscopy structures of two designed antibodies bound to their respective pMHC targets demonstrate atomic-level accuracy at the recognition interface, supporting the robustness of our structure-based approach.
Computationally designed TCRs and antibodies targeting pMHC complexes could enable a broad range of therapeutic applications, from cancer immunotherapy to autoimmune disease treatment, while insights gained from TCR–pMHC design advance predictive understanding of TCR specificity with implications for basic immunology and clinical diagnostics.


# Installation

## the pipeline scripts

The ADAPT pipeline consists of a set of scripts that call the independent
neural network tools Alphafold2, ProteinMPNN, and (optionally for design
ranking) RFantibody. To run the scripts, you will need to create a Python environment
which satisfies the requirements in `requirements.txt`. We recommend that you
do this in a virtual environment. With the environment activated you could do
something like

```
pip install -r requirements.txt
```

## the NN models

We are working on a Docker file to make this process easier, but in the meantime
it will be necessary for you to have working installations of those 3
packages, which are available at the following locations:

https://github.com/google-deepmind/alphafold

https://github.com/dauparas/ProteinMPNN

https://github.com/RosettaCommons/RFantibody

Note that ADAPT ships with a slightly modifed version of Alphafold2, but you will
still need a compatible Python environment to run it. The installation process is
somewhat platform specific as it depends on your version of CUDA.


## telling ADAPT where to find things

Once you have the network tools installed and the parameter and database files
downloaded, you will need to edit `./config_paths.yaml` to point the ADAPT scripts
to the various files and environments.

# running TCR design

```

python design/dock_design.py \
    --pmhc_targets design_targets.tsv \
    --tcr_pdbids 1oga 5bs0 3gsn 3qdg \
    --design_cdrs 0 1 3 4 5 7 \
    --num_designs 10 \
    --outfile_prefix /path/to/output/run1_design_jobN


```

where `design_targets.tsv` specifies the pMHC targets and would look something like this:

```
organism	mhc_class	mhc	peptide
human	1	A*01:01	EVDPIGHLY
human	1	A*02:01	ALYDKTKRI
human	1	A*02:01	TLMSAMTNL
```


# running TCR refinement

```

python design/dock_refine.py \
    --poolfile /path/to/output/run1_refine_pool.tsv \
    --sort_tag combo_score_wtd \
    --pmhc_targets design_targets.tsv \
    --max_pool_size 200 \
    --max_per_lineage 10 \
    --num_parents 10 \
    --num_mutations 2 \
    --outfile_prefix /path/to/output/run1_refine_jobN


```
