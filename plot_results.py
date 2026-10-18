from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path(__file__).resolve().parent/'outputs';d=pd.read_csv(b/'metrics.csv');pairs=pd.read_csv(b/'paired_intervals.csv')
ref=d[(d.scenario=='reference')&(d.split=='test')].groupby('method')[['accuracy','tpr_gap']].agg(['mean','std'])
fig,ax=plt.subplots(figsize=(8,5))
for name,row in ref.iterrows():
 ax.errorbar(row[('accuracy','mean')],row[('tpr_gap','mean')],xerr=row[('accuracy','std')],yerr=row[('tpr_gap','std')],fmt='o',capsize=2,label=name)
ax.set(xlabel='Accuracy against qualification',ylabel='Absolute qualification TPR gap',title='Reference scenario across 30 seeds');ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(b/'reference_comparison.png',dpi=220,bbox_inches='tight');plt.close(fig)
z=pairs[(pairs.comparison=='review_tpr_0.05')&(pairs.reference=='review')&(pairs.metric=='tpr_gap')].copy().sort_values('mean_difference')
fig,ax=plt.subplots(figsize=(9,6));y=np.arange(len(z));ax.errorbar(z.mean_difference,y,xerr=[z.mean_difference-z.ci_low,z.ci_high-z.mean_difference],fmt='o',capsize=3);ax.set_yticks(y,z.scenario);ax.axvline(0,color='grey',linestyle='--');ax.set(xlabel='Paired mean change in TPR gap versus review only\n95% seed-bootstrap intervals; positive means a larger gap',title='Incremental constraint effect across sensitivity scenarios');fig.tight_layout();fig.savefig(b/'paired_sensitivity.png',dpi=220);plt.close(fig)
