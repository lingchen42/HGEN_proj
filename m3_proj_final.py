#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.stats import chisquare
from scipy.stats import probplot
from numpy.random import dirichlet
from numpy.random import dirichlet
from numpy.random import choice
from scipy.stats import mode

import warnings
warnings.filterwarnings('ignore')

sns.set_style('white')

parser = argparse.ArgumentParser(description='HGEN module 3 homework')
parser.add_argument("-i", "--input", required=True,
          help="input the data table which should be an N by M+2 table, with each row representing an individual, and the 1st column is ID, 2nd affection status, remaining columns genotypes")
parser.add_argument("-f", "--frequency", action="store_true", default=False,
					help="Plot a histogram of the allele frequency")
parser.add_argument("--hwe", action="store_true", default=False,
					help="Calculate the p value of HWE for each variant and plot the distribution of the p values and QQ plot of the p values (log scale)"),
parser.add_argument("--ld", action="store_true", default=False,
					help="Calculate and plot 3 heatmaps of pairwise LD among all pairs of the variants: D, D', r2")
parser.add_argument("--pca", action="store_true", default=False,
					help = "Perform PCA and plot PC1 vs PC2, PC1 vs PC3, PC2 vs PC3")
parser.add_argument("--subpopmembership", action="store_true",
					help = "Calculate the membership of each individual with the marginal probability of being in the assined substructure based on the Gibbs sampler results, default K=2, use the common variants in the first 1000 variants, 200 rounds")
parser.add_argument("-k", default=2, type=int,
                    help= "Specify the number of populations to infer from the gibbs sampling method")
parser.add_argument("--nofvariants", default=1000, type=int,
                    help= "Specify the number of variants to perform the gibbs sampling method")
parser.add_argument("--rounds", default=200, type=int,
                    help= "Specify the number of rounds for the gibbs sampling method to reach a stationary distribution")
parser.add_argument("--all", action="store_true", default=False,
					help = "Do all the analysis, generate a pdf report")
parser.add_argument("-o", "--out", required=True,
					help = "The file name of the pdf report")


args = parser.parse_args()


def cal_allelefreq(df):
    afs_0 = []
    afs_minor = []
    genotypes_00 = []
    genotypes_01 = []
    genotypes_11 = []
    
    variants = df.columns[2:]
    for variant in variants:
        df_genotypes = df[variant].value_counts()
        
        try:
            genotype_00 = df_genotypes['0/0']
        except KeyError:
            genotype_00 = 0
            
        try:
            genotype_01 = df_genotypes['0/1']
        except KeyError:
            genotype_01 = 0
            
        try:
            genotype_11 = df_genotypes['1/1']
        except KeyError:
            genotype_11 = 0
        
        total = float(2*(genotype_00+genotype_01+genotype_11))
        af_0 = (2*genotype_00 + genotype_01)/total
        af_1 = 1 - af_0
        
        afs_0.append(af_0)
        afs_minor.append(min(af_0,af_1))
        genotypes_00.append(genotype_00)
        genotypes_01.append(genotype_01)
        genotypes_11.append(genotype_11)
        
    allele_summary = pd.DataFrame(data={'Variants': variants, 'Frequency of 0 allele': afs_0, 'Frequency of minor allele': afs_minor, 'Frequency of genotype 0/0': genotypes_00, 'Frequency of genotype 0/1': genotypes_01,'Frequency of genotype 1/1':genotypes_11})
        
    return allele_summary


def af_hist(allele_summary):
    
    include_fixed = allele_summary['Frequency of 0 allele']
    exclude_fixed = [i for i in allele_summary['Frequency of 0 allele'] if (i!=1)&(i!=0)]

    af_fig = plt.figure(figsize=(10,4))
    ax1 = af_fig.add_subplot(121) 
    sns.distplot(exclude_fixed, kde=False, ax=ax1)
    ax1.set_xlim(0,1)
    ax1.set_title('Allele Frequency Distribution')
    ax1.set_xlabel('The Frequency of Polymorphic Alleles')

    ax2 = af_fig.add_subplot(122) 
    sns.distplot(include_fixed, kde=False, ax=ax2)
    ax2.set_xlim(0,1)
    ax2.set_title('Allele Frequency Distribution')
    ax2.set_xlabel('The Frequency of All Alleles')

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    

def hwe_p(row):
    observed_genotypes = [row['Frequency of genotype 0/0'], row['Frequency of genotype 0/1'], row['Frequency of genotype 1/1']]
    total = sum(observed_genotypes)
    af_0 = row['Frequency of 0 allele']
    expected_genotypes = [total*af_0**2, total*2*af_0*(1-af_0), total*(1-af_0)**2]
    
    statistic , p = chisquare(observed_genotypes, expected_genotypes, ddof=1)

    return p


def hwe(allele_summary):

    hwe_fig = plt.figure(figsize=(8,8)) 
    
    # Histogram of observed p-values exluding those have allele frequencies ~ (0.02,0.98)
    allele_summary_common = allele_summary[(allele_summary['Frequency of 0 allele'] > 0.02) & (allele_summary['Frequency of 0 allele'] < 0.98)] 
    allele_summary_common['HWE p value'] = allele_summary_common.apply(hwe_p, axis=1)
    observed_data_common = list(allele_summary_common['HWE p value'].dropna())
    log_observed_data_common = -np.log(observed_data_common)
    
    ax3 = hwe_fig.add_subplot(221)
    sns.distplot(observed_data_common, kde=False, ax=ax3)
    ax3.set_title('HWE p value distribution of common alleles')
    ax3.set_xlabel('The p-value of common Polymorphic Alleles')
    
    # QQ-plot excluding extremly rare alleles (possibly genotying error?)
    ax4 = hwe_fig.add_subplot(222)
    probplot(log_observed_data_common, dist='expon',plot=ax4)
    ax4.set_xlabel('Expected -ln(p)')
    ax4.set_ylabel('Observed -ln(p)')
    ax4.set_title('QQ plot (common variants)')
    ax4.set_ylim(0,)
    ax4.set_aspect('equal')
    lims = [np.min([ax4.get_xlim(), ax4.get_ylim()]),
    np.max([ax4.get_xlim(), ax4.get_ylim()])]
    ax4.plot(lims, lims, '--', color='grey')
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    return allele_summary, allele_summary_common, hwe_fig


def em(total, p_h00, p_h01, p_h10, p_h11, n_unphased, n_h00_h00, n_h01_h01, n_h10_h10, n_h11_h11, n_h00_h01, n_h00_h10, 
       n_h01_h00, n_h01_h11, n_h10_h00, n_h10_h11, n_h11_h01, n_h11_h10):
         
    # E-step, calculate expectation
    p_unphased = p_h00*p_h11+p_h01*p_h10
    if p_unphased:
        n_h00 = 2*n_h00_h00 + n_h00_h01 + n_h00_h10 + n_unphased*(p_h00*p_h11/p_unphased)  
        n_h01 = 2*n_h01_h01 + n_h01_h00 + n_h01_h11 + n_unphased*(p_h01*p_h10/p_unphased)
        n_h10 = 2*n_h10_h10 + n_h10_h00 + n_h10_h11 + n_unphased*(p_h01*p_h10/p_unphased)
        n_h11 = 2*n_h11_h11 + n_h11_h01 + n_h11_h10 + n_unphased*(p_h00*p_h11/p_unphased)
    else:
        n_h00 = 2*n_h00_h00 + n_h00_h01 + n_h00_h10
        n_h01 = 2*n_h01_h01 + n_h01_h00 + n_h01_h11
        n_h10 = 2*n_h10_h10 + n_h10_h00 + n_h10_h11
        n_h11 = 2*n_h11_h11 + n_h11_h01 + n_h11_h10
        
    # M-step, gene counting
    p_h00_new = n_h00/total
    p_h01_new = n_h01/total
    p_h10_new = n_h10/total
    p_h11_new = n_h11/total
    
    return p_h00_new, p_h01_new, p_h10_new, p_h11_new
    
    
def run_em(df, a1, a2):
    
    # initialize the prob
    p_h00 = 0.25
    p_h01 = 0.25
    p_h10 = 0.25
    p_h11 = 0.25
    
    # haplotype counts
    total = 2*float(len(df))
    n_unphased = len(df[(df[a1]=='0/1')&(df[a2]=='0/1')])
    n_h00_h00 = len(df[(df[a1]=='0/0')&(df[a2]=='0/0')]) 
    n_h01_h01 = len(df[(df[a1]=='0/0')&(df[a2]=='1/1')]) 
    n_h10_h10 = len(df[(df[a1]=='1/1')&(df[a2]=='0/0')]) 
    n_h11_h11 = len(df[(df[a1]=='1/1')&(df[a2]=='1/1')]) 
    n_h00_h01 = len(df[(df[a1]=='0/0')&(df[a2]=='0/1')])                            
    n_h00_h10 = len(df[(df[a1]=='0/1')&(df[a2]=='0/0')])
    n_h01_h00 = n_h00_h01
    n_h01_h11 = len(df[(df[a1]=='0/1')&(df[a2]=='1/1')])
    n_h10_h00 = n_h00_h10
    n_h10_h11 = len(df[(df[a1]=='1/1')&(df[a2]=='0/1')])
    n_h11_h01 = n_h01_h11
    n_h11_h10 = n_h10_h11
    
    # Convergence criteria
    crit = 1e-4
    converged = False
    
    while not converged:
        p_h00_new, p_h01_new, p_h10_new, p_h11_new = em(total, p_h00, p_h01, p_h10, p_h11, n_unphased, 
                                                        n_h00_h00, n_h01_h01, n_h10_h10, n_h11_h11, 
                                                        n_h00_h01, n_h00_h10, n_h01_h00, n_h01_h11, 
                                                        n_h10_h00, n_h10_h11, n_h11_h01, n_h11_h10)
        converged = (np.abs(p_h00_new-p_h00)<crit)&(np.abs(p_h01_new-p_h01)<crit)&(np.abs(p_h10_new-p_h10)<crit)&(np.abs(p_h11_new-p_h11)<crit)
        p_h00, p_h01, p_h10, p_h11 = p_h00_new, p_h01_new, p_h10_new, p_h11_new      
    
    return p_h00, p_h01, p_h10, p_h11


def cal_ld(p_A, p_B, p_hAB):
    
    # D
    d = p_hAB - p_A*p_B
    # D'
    if d:
        if d < 0:
            dprime = d/max(-p_A*p_B, -(1-p_A)*(1-p_B))
        elif d > 0:
            dprime = d/min(p_A*(1-p_B), (1-p_A)*p_B)
        # r2
        r2 = d**2/(p_A*(1-p_A)*p_B*(1-p_B))
    else:
        dprime = r2 = 0 
    
    return {'d':d, 'dprime':dprime, 'r2':r2}


def ld_table(df, allele_summary):
            
    polymorphic_alleles = list(allele_summary[(allele_summary['Frequency of 0 allele']<=0.98)&(allele_summary['Frequency of 0 allele']>0.02)]['Variants'])  
    ld_df_d = pd.DataFrame(index=polymorphic_alleles,columns=polymorphic_alleles, dtype=np.float64)
    ld_df_dprime = pd.DataFrame(index=polymorphic_alleles,columns=polymorphic_alleles, dtype=np.float64)
    ld_df_r2 = pd.DataFrame(index=polymorphic_alleles,columns=polymorphic_alleles, dtype=np.float64)

    for a1 in polymorphic_alleles:
        for a2 in polymorphic_alleles:
            if a1!=a2:
                p_hAB, p_hAb, p_haB, p_hab = run_em(df, a1, a2)
                p_A = p_hAB + p_hAb
                p_B = p_hAB + p_haB
            
                ld_dict = cal_ld(p_A, p_B, p_hAB)
                ld_df_d[a1][a2] = ld_df_d[a2][a1] = ld_dict['d']
                ld_df_dprime[a1][a2] = ld_df_dprime[a2][a1] = ld_dict['dprime']
                ld_df_r2[a1][a2] = ld_df_r2[a2][a1] = ld_dict['r2']
            else:
                ld_df_r2[a1][a2] = 1

    return ld_df_d, ld_df_dprime, ld_df_r2


def ld_heatmap(ld_df, metric='D'):
    
    def triu_mask(df):
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        return mask
    
    cmap = sns.light_palette('red', as_cmap=True) 
    
    sns.set(font_scale=0.5)
    sns.set_style('white')
    fig = plt.figure(figsize=(11,9))
    mask = triu_mask(ld_df)
    plt.title(metric)
    sns.heatmap(ld_df, mask=mask, cmap=cmap, square=True, 
                linewidths=.5)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def code_genotype(df, mode='additive'):
    
    if mode=='additive':
        df = df.replace('0/0', 0)
        df = df.replace('0/1', 1)
        df = df.replace('1/1', 2)
    elif mode=='dominant':
        df = df.replace('0/0', 0)
        df = df.replace('0/1', 1)
        df = df.replace('1/1', 1)
    else: #recessive
        df = df.replace('0/0', 0)
        df = df.replace('0/1', 0)
        df = df.replace('1/1', 1)
        
    return df

def preprocess_df(df):
     
    labels = df['status']
    df = df.drop('status',axis=1).set_index('ID')
    df = df.astype(np.float64)
    df_scaled = preprocessing.scale(df, axis=0)
    
    return df_scaled, labels

def run_pca(df):
    
    # code the genotype
    df = code_genotype(df, mode='additive')
    
    # standardize the df
    df, labels = preprocess_df(df)
    
    # PCA
    pca = PCA(n_components=3)
    df_r = pca.fit(df).transform(df)
    explained_variance_ratio = pca.explained_variance_ratio_
    print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))
    pcs = pca.components_
    
    # Plot PCs
    fig = plt.figure(figsize=(12,4))
    
    color = ['navy','darkorange']
    lw=2
    
    controls = np.array(labels == 'control')
    cases = np.array(labels == 'case')
    
    # plot PC1 against PC2
    ax1 = fig.add_subplot(131) 
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.scatter(df_r[controls, 0], df_r[controls, 1], color=color[0], alpha=.8, lw=lw,
                label='control')
    ax1.scatter(df_r[cases, 0], df_r[cases, 1], color=color[1], alpha=.8, lw=lw,
                label='case')
    ax1.legend()
    
    # plot PC1 against PC3
    ax2 = fig.add_subplot(132) 
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC3')
    ax2.scatter(df_r[controls, 0], df_r[controls, 2], color=color[0], alpha=.8, lw=lw,
                label='control')
    ax2.scatter(df_r[cases, 0], df_r[cases, 2], color=color[1], alpha=.8, lw=lw,
                label='case')
    ax2.legend()
    
    
    # plot PC2 against PC3
    ax3 = fig.add_subplot(133) 
    ax3.set_xlabel('PC2')
    ax3.set_ylabel('PC3')
    ax3.scatter(df_r[controls, 1], df_r[controls, 2], color=color[0], alpha=.8, lw=lw,
                label='control')
    ax3.scatter(df_r[cases, 1], df_r[cases, 2], color=color[1], alpha=.8, lw=lw,
                label='case')
    ax3.legend()

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    return df_r

def init_gibbs(df, allele_summary, K=3):  
    
    # calculate the Pkl based on the subpopulation membership
    # Only use common polymorphic alleles
    polymorphic_alleles = list(allele_summary[(allele_summary['Frequency of 0 allele']>=0.02)&(allele_summary['Frequency of 0 allele']<=0.98)]['Variants'])  
    df = df.ix[:, ['ID']+polymorphic_alleles] #only keep polymorphic snps
    df =code_genotype(df)
    
    # initiate the subpopulation membership equally
    z=[]
    for i in range(K):
        z+=((len(df)/K)*[i])
    z+=(len(df)-len(z))*[K]
    
    df['subpop'] = z
    
    grped = df.groupby('subpop')
    
    return polymorphic_alleles, df

def update_z(K, polymorphic_alleles, subpop_dict, df):
    
    new_zs = []
    pr_ks = {}
   
    for df_index, row in df.iterrows():

        # prx_ks = [prx_pop0, prx_pop1]
        prx_ks = [1]*K

        for a in polymorphic_alleles:
            genotype = row[a]
                
            # prx_ks = [1:prx_pop0, 2:prx_pop1]
            # prx_pop0 *= prx_pop0_l
            # prx_pop0_1
                       
            for k in range(K):
                p_0 = subpop_dict[k][a][0]
                p_1 = subpop_dict[k][a][1]
                
                if genotype == 0: 
                    prx_popk_a = p_0**2
                elif genotype == 1:      
                    prx_popk_a = 2*p_0*p_1
                elif genotype == 2: 
                    prx_popk_a = p_1**2
                
                # update prx for allele a based on gentype of allele a
                prx_popk = prx_ks[k]
                prx_popk *= prx_popk_a
                prx_ks[k] = prx_popk 
       
        prx_sum = sum(prx_ks)
        pr_z = [i/prx_sum for i in prx_ks]
        # According to the probability, sample the subpop membership
        new_k = choice(K, 1, p=pr_z)[0]
        
        new_zs.append(new_k)
        pr_ks[row['ID']] = pr_z[new_k]
    
    new_df = df
    new_df['subpop'] = new_zs
    
    return new_df, pr_ks

def update_subpop_dict(new_df, K, polymorphic_alleles):
    
    # update subpop_dict = {pop1:{Psnp1=[p_0,p_1], ...}, pop2: ...}
    subpop_dict={}
    grped = new_df.groupby('subpop')
    
    for k in range(K):
        Pkl_dict = {}
        
        try:
            grp = grped.get_group(k)       
            for a in polymorphic_alleles:
                t = grp[a].value_counts().to_frame()
                try:
                    zeros = t[a][0]
                except:
                    zeros = 0
                try:
                    ones = t[a][1]
                except:
                    ones = 0
                try:
                    twos = t[a][2]
                except:
                    twos = 0
                    
                n_zero = 2*zeros + ones
                n_one = ones + 2*twos
            
                #sample prior probility of pkl from dirichlet
                Pkl_dict[a] = dirichlet([1+n_zero, 1+n_one], size=1)[0]
        
        except KeyError:
            # random ps for not-seen populations
            for a in polymorphic_alleles:
                Pkl_dict[a] = dirichlet([1, 1], size=1)[0]
                        
        subpop_dict[k] = Pkl_dict
    
    return subpop_dict

def run_gibbs(df, allele_summary, K=3, rounds=5000, plot_p_history=False, plot_zlikelihood_history=False):
   
    polymorphic_alleles, df = init_gibbs(df, allele_summary, K)
    zs = list(df['subpop'])
    zs_all =[zs]
    
    stable = False
    n = 0

    colors = ['darkorange','navy']
    
    if plot_p_history:
        # plot 0 allele frequency history during sampling of a random set of alleles (5) of subpopulation 1
        alleles_to_plt = choice(polymorphic_alleles, 5)       
        fig = plt.figure(figsize=(20,4))
        ax1 = fig.add_subplot(151)
        ax2 = fig.add_subplot(152)
        ax3 = fig.add_subplot(153)
        ax4 = fig.add_subplot(154)
        ax5 = fig.add_subplot(155)

    if plot_zlikelihood_history:

        individuals_to_plt = choice(list(df['ID']), 16)
        nrows = 4
        ncols = 4
        fig, axes = plt.subplots(nrows, ncols)
    
    subpop_dict = update_subpop_dict(new_df=df, K=K, polymorphic_alleles=polymorphic_alleles)
    while not stable:
               
        new_subpop_dict = update_subpop_dict(new_df=df, K=K, polymorphic_alleles=polymorphic_alleles)
        new_df, pr_ks = update_z(K=K, polymorphic_alleles=polymorphic_alleles, subpop_dict=new_subpop_dict, df=df)
        new_zs = list(new_df['subpop'])     
       
        df=new_df
        subpop_dict=new_subpop_dict
        zs = new_zs
        zs_all.append(zs)
        
        if plot_p_history:
            ax1.scatter(n, subpop_dict[1][alleles_to_plt[0]][0], c=colors[0], s=12)
            ax2.scatter(n, subpop_dict[1][alleles_to_plt[1]][0], c=colors[0], s=12) 
            ax3.scatter(n, subpop_dict[1][alleles_to_plt[2]][0], c=colors[0], s=12)
            ax4.scatter(n, subpop_dict[1][alleles_to_plt[3]][0], c=colors[0], s=12) 
            ax5.scatter(n, subpop_dict[1][alleles_to_plt[4]][0], c=colors[0], s=12) 

        if plot_zlikelihood_history:
            for x in range(nrows):
                for y in range(ncols):
                    individual = individuals_to_plt[nrows*(x-1)+y]
                    axes[x][y].scatter(n, pr_ks[individual], c=colors[1], s=10)
                    if n ==0:
                        axes[x][y].set_title(individual, fontsize='xx-small')
                
        n+=1
        
        if n%100 ==0:
            print 'Round %s'%(n)
        if n > rounds :
            stable = True
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()

            
    return zs_all, subpop_dict, df

def compute_pr_z(zs_all, df, final_frac=0.5):
    
    after_burn_in = int(-len(zs_all)*final_frac)
    final_zs = np.array(zs_all[after_burn_in:]).T

    pr_zs =[]
    memberships = []
    for z in final_zs: 
        membership = mode(z)[0][0]
        counts = (z == membership).sum()
        pr_z = float(counts)/len(z)
        pr_zs.append(pr_z)
        memberships.append(membership)
    
    df['final membership'] = memberships
    df['Probability of membership'] = pr_zs
    
    return df

def validate_gibbs(df_r, df_gibbs, K=2):
    memberships = df_gibbs['final membership']
    
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    lw=1
    colors = ['orange', 'blue']
    
    for k, color in zip(range(K), colors[:K]):
        members_in_k = np.array(memberships==(k))
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        
        ax1.scatter(df_r[members_in_k, 0], df_r[members_in_k, 1], color=color, alpha=.7, lw=lw,
                label='%s'%(k))
        ax1.legend()
        
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC3')
        ax2.scatter(df_r[members_in_k, 0], df_r[members_in_k, 2], color=color, alpha=.7, lw=lw,
                label='%s'%(k))
        ax2.legend()
        
        ax3.set_xlabel('PC2')
        ax3.set_ylabel('PC3')
        ax3.scatter(df_r[members_in_k, 1], df_r[members_in_k, 2], color=color, alpha=.7, lw=lw,
                label='%s'%(k))
        
        ax3.legend()
        
    plt.tight_layout()      

    pdf.savefig()
    plt.close()


if __name__ == '__main__':

	# Out pdf
    pdf_fn = args.out
    print 'Writing output pdf report to %s'%(pdf_fn)
  

    pdf = PdfPages(pdf_fn)

	# Read the input table
    df = pd.read_table(args.input)

    if args.frequency or args.hwe or args.ld or args.subpopmembership or args.all: 
        allele_summary = cal_allelefreq(df)
	
	# Plot allele frequency distribution
    if args.frequency or args.all:
        af_hist(allele_summary)

	# Plot hardy weinberg p value distribution and QQ plot
    if args.hwe or args.subpopmembership or args.all:
        print 'Plot histrogram of allele frequency distribution...'
        allele_summary, allele_summary_common, hwe_fig=hwe(allele_summary)

	# Haplotype estimation and plot LD heatmap
    if args.ld or args.all:
        print 'Calculating LD table... for the first 500 polymorphic alleles'
        ld_df_d, ld_df_dprime, ld_df_r2 = ld_table(df, allele_summary[:500])

        ld_d_out = pdf_fn.split('.')[0]+'.ld%s_common_500.csv'%('.d')
        ld_df_d.to_csv(ld_d_out, header=True, index=True)

        ld_dprime_out = pdf_fn.split('.')[0]+'.ld%s_common_500.csv'%('.dprime')
        ld_df_dprime.to_csv(ld_dprime_out, header=True, index=True)

        ld_r2_out = pdf_fn.split('.')[0]+'.ld%s_common_500.csv'%('.r2')
        ld_df_r2.to_csv(ld_r2_out, header=True, index=True)
        print 'Saving ld matrix to %s, %s, %s'%(ld_d_out,ld_dprime_out, ld_r2_out)	

        ld_heatmap(ld_df_d)
        ld_heatmap(ld_df_dprime, metric="D'")
        ld_heatmap(ld_df_r2, metric="R-squared")

	# Run PCA on the data
    if args.pca or args.subpopmembership or args.all:
        print 'Plotting PCA...'
        df_r = run_pca(df)

    if args.subpopmembership:

        K = args.k
        n_variants = args.nofvariants
        rounds = args.rounds

        print 'Perform gibbs sampling methods for inferring subpopulation membership based on common variants in first %s variants, ignoring LD, K=%s, rounds=%s'%(n_variants, K, rounds)
        zs_all, subpop_dict, df_gibbs = run_gibbs(df, allele_summary[:n_variants], K=K, rounds=rounds, plot_zlikelihood_history=True)
    
        df_gibbs = compute_pr_z(zs_all, df, final_frac=0.5)
        print df_gibbs['final membership'].value_counts()
        validate_gibbs(df_r, df_gibbs, K=K)

        membership_out = pdf_fn.split('.')[0]+'.membership_500.csv'
        df_gibbs.to_csv(membership_out, header=True, index=True)

    pdf.close()
