# HGENII Module3 Project

### - HGEN report revised.pdf
This the write up which includes results and discussion of the results.

### - ./m3_proj_final.py
This is the main script for calculation allele frequencies, hwe, ld, PCA and inferring subpopulation membership. The usage of this script is listed below. Basically, it takes a given data table (in the exact format as suggested in the project description) with `-i` option and compute the answers to all questions of the project3 with `--all` option or selectively compute the answer of some of the questions as suggested below.

One caveat is that when calculating the LD, it only computes for polymorphic variants of the first 500 variants of each dataset due to the computation power.

Required packages: numpy, pandas, matplotlib, seaborn, sklearn, scipy

```
./m3_proj_final.py -h
usage: m3_proj_final.py [-h] -i INPUT [-f] [--hwe] [--ld] [--pca]
                        [--subpopmembership] [-k K]
                        [--nofvariants NOFVARIANTS] [--rounds ROUNDS] [--all]
                        -o OUT

HGEN module 3 homework

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input the data table which should be an N by M+2
                        table, with each row representing an individual, and
                        the 1st column is ID, 2nd affection status, remaining
                        columns genotypes
  -f, --frequency       Plot a histogram of the allele frequency
  --hwe                 Calculate the p value of HWE for each variant and plot
                        the distribution of the p values and QQ plot of the p
                        values (log scale)
  --ld                  Calculate and plot 3 heatmaps of pairwise LD among all
                        pairs of the variants: D, D', r2
  --pca                 Perform PCA and plot PC1 vs PC2, PC1 vs PC3, PC2 vs
                        PC3
  --subpopmembership    Calculate the membership of each individual with the
                        marginal probability of being in the assined
                        substructure based on the Gibbs sampler results,
                        default K=2, use the common variants in the first 1000
                        variants, 200 rounds
  -k K                  Specify the number of populations to infer from the
                        gibbs sampling method
  --nofvariants NOFVARIANTS
                        Specify the number of variants to perform the gibbs
                        sampling method
  --rounds ROUNDS       Specify the number of rounds for the gibbs sampling
                        method to reach a stationary distribution
  --all                 Do all the analysis, generate a pdf report
  -o OUT, --out OUT     The file name of the pdf report
```

### - HGEN341_module3_final_revised.ipynb/html
  This jupyter notebook have the code, answers and the plots. The html is a static version of the notebook.
