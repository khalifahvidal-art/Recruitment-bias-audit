import pandas as pd,numpy as np,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_experiment import generate,metrics
b=ROOT/'outputs';df=pd.read_csv(b/'metrics.csv');g=pd.read_csv(b/'group_metrics.csv')
assert len(df)==5190
assert df[df.split=='test'].groupby(['scenario','method']).size().eq(30).all()
assert g.groupby(['scenario','seed','method','split'])['n'].sum().gt(0).all()
# Independently recalculate accuracy and group gaps from integer counts.
a=g.groupby(['scenario','seed','method','split']).apply(lambda z:(z.tp.sum()+z.negative.sum()-z.fp.sum())/z.n.sum(),include_groups=False)
m=df[df.method!='oracle'].set_index(['scenario','seed','method','split'])
assert np.allclose(m.accuracy.sort_index(),a.sort_index())
for name,num,den in [('tpr','tp','positive'),('fpr','fp','negative'),('sr','selected','n')]:
 assert np.allclose(g[name],g[num]/g[den])
 gap=g.pivot(index=['scenario','seed','method','split'],columns='group',values=name).diff(axis=1)[1].abs()
 assert np.allclose(m[name+'_gap'].sort_index(),gap.sort_index())
# Exact generator continuity against original saved seed 0.
old=pd.read_csv(ROOT/'original_reference/synthetic_seed0.csv');new=pd.read_csv(b/'synthetic_seed0.csv')
for x,y in [('skill','skill'),('experience','experience'),('proxy','proxy'),('a','group'),('qualification','qualification'),('historical_label','historical'),('split','split')]:
 if old[x].dtype=='object':assert (old[x]==new[y]).all()
 else:assert np.allclose(old[x],new[y])
# Check initial ten reference metrics against original outputs.
o=pd.read_csv(ROOT/'original_reference/metrics_by_seed.csv')
key={'baseline':'historical','reweighing_historical':'reweighing','TPR_on_biased_labels':'historical_tpr_0.05','label_review_only':'review','TPR_historical_0.02':'historical_tpr_0.02','TPR_historical_0.10':'historical_tpr_0.10','label_review_and_TPR_0.02':'review_tpr_0.02','label_review_and_TPR_0.05':'review_tpr_0.05','label_review_and_TPR_0.10':'review_tpr_0.10'}
o['method']=o.method.map(key);v=df[(df.scenario=='reference')&(df.split=='test')&(df.seed<10)&(df.method!='oracle')]
merged=o.merge(v,on=['seed','method']);maxdiff={k:float(abs(merged[k+'_x']-merged[k+'_y']).max()) for k in ['accuracy','tpr_gap','fpr_gap']}
assert max(maxdiff.values())<1e-12,maxdiff
res={'version':'1.0.0','scenarios':15,'seeds_per_scenario':30,'metric_rows':len(df),'group_rows':len(g),'independent_count_checks':'passed','original_seed0_generator':'passed','first_ten_reference_metric_max_difference':maxdiff}
(ROOT/'verification.json').write_text(json.dumps(res,indent=2));print(res)
