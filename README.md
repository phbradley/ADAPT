# ADAPT
Antigen-receptor Design Against Peptide-MHC Targets

# Installation

## the pipeline scripts

The ADAPT pipeline consists of a set of scripts that call the independent
neural network tools Alphafold2, ProteinMPNN, and (optionally for design
ranking) RFantibody. To run the scripts, you will need to create a Python environment
which satisfies the requirements in `requirements.txt`. We recommend that you
do this in a virtual environment. With the environment activated you could do
something like

`pip install -r requirements.txt`


## the NN models

We are working on a Docker file to make this process easier, but in the meantime
it will be necessary for you to have working installations of those 3
packages, which are available at the following locations:

https://github.com/google-deepmind/alphafold

https://github.com/dauparas/ProteinMPNN

https://github.com/RosettaCommons/RFantibody

Note that ADAPT ships with a slightly modifed version of Alphafold2, but you will
still a compatible Python environment to run it. The installation process is somewhat
platform specific as it depends on your version of CUDA.


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
