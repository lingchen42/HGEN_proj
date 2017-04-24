# HGENII Module3 Project

### - HGEN341_module3_final.ipynb/html
  This jupyter notebook have the code, answers and the plots. The html is a static version of the notebook.

### - m3_proj_new.py
This is the main script for calculation allele frequencies, hwe, ld and PCA. I am still working on implementing the subpopulation membership. The usage of this script is listed below. Basically, it takes a given data table (in the exact format as suggested in the project description) with `-i` option and compute the answers to all questions of the project3 with `--all` option or selectively compute the answer of some of the questions as suggested below.

One caveat is that when calculating the LD, it only computes for the first 500 polymorphic variants due to the computation power.
Another caveat is that I only use first 500 common variants in the data table for computing subpop membership due to the computation power.

Required packages: numpy, pandas, matplotlib, seaborn, sklearn, scipy

```
./m3_proj_new.py -h
usage: m3_proj.py [-h] -i INPUT [-f] [--hwe] [--ld] [--pca]
                  [--subpopmembership] [--all] [-o OUT]

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
                        substructure based on the Gibbs sampler results, default 
                        k=3
  --all                 Do all the analysis, generate a pdf report
  -o OUT, --out OUT     The file name of the pdf report 
```


### - m3_proj.py
Same as `m3_proj_new.py`, without subpopmembership method.

### - hgen_data#.500.pdf
  These are pdf files generated by `m3_proj.py`, which contain all the plots for data 1-6.
