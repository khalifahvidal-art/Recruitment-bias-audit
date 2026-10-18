"""Controlled synthetic recruitment audit. No real personal data. Version 1.0.0."""
from pathlib import Path
import json, platform, hashlib, argparse
import numpy as np
import pandas as pd
import sklearn, fairlearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def metrics(y,p,a):
    rates={}; out={'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p)}
    for g in (0,1):
        m=a==g; pos=m&(y==1); neg=m&(y==0)
        rates[g]={'n':int(m.sum()),'positive':int(pos.sum()),'negative':int(neg.sum()),'selected':int(p[m].sum()),'tp':int(p[pos].sum()),'fp':int(p[neg].sum()),'sr':p[m].mean(),'tpr':p[pos].mean() if pos.any() else np.nan,'fpr':p[neg].mean() if neg.any() else np.nan,'ppv':y[m&(p==1)].mean() if (m&(p==1)).any() else np.nan}
    for k in ['sr','tpr','fpr','ppv']:out[k+'_gap']=abs(rates[1][k]-rates[0][k])
    out['selection_ratio']=min(rates[g]['sr'] for g in (0,1))/max(rates[g]['sr'] for g in (0,1)) if max(rates[g]['sr'] for g in (0,1)) else np.nan
    return out,rates

def generate(seed,c):
    rng=np.random.default_rng(seed); n=c.get('n',4000)
    a=rng.binomial(1,c.get('group_p',.5),n)
    skill=rng.normal(size=n); experience=rng.normal(size=n)
    proxy=rng.normal(c.get('proxy',1.8)*(.5-a),.7,n)
    prob=1/(1+np.exp(-(1.4*skill+.8*experience+c.get('base_shift',0)*a)))
    y=rng.binomial(1,prob)
    h=y.copy();h[(a==1)&(y==1)&(rng.random(n)<c.get('suppression',.45))]=0
    # Draw review noise after historical labels, preserving original default generator.
    reviewed=y.copy();flip=rng.random(n)<c.get('review_error',0);reviewed[flip]=1-reviewed[flip]
    tr,te=train_test_split(np.arange(n),test_size=.3,random_state=seed,stratify=2*a+y)
    X=np.column_stack([skill,experience,proxy]);sc=StandardScaler().fit(X[tr]);return X,sc.transform(X[tr]),sc.transform(X[te]),a,y,h,reviewed,prob,tr,te

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=30);ap.add_argument('--out',default='outputs');args=ap.parse_args()
    out=Path(__file__).resolve().parent/args.out;out.mkdir(exist_ok=True,parents=True)
    scenarios={'reference':{},'suppression_0':{'suppression':0},'suppression_025':{'suppression':.25},'suppression_065':{'suppression':.65},'proxy_0':{'proxy':0},'proxy_09':{'proxy':.9},'proxy_27':{'proxy':2.7},'sample_1000':{'n':1000},'sample_8000':{'n':8000},'group_02':{'group_p':.2},'group_08':{'group_p':.8},'base_minus_08':{'base_shift':-.8},'base_plus_08':{'base_shift':.8},'review_error_01':{'review_error':.1},'review_error_02':{'review_error':.2}}
    rows=[];group_rows=[];diag=[]
    for sn,c in scenarios.items():
      for seed in range(args.seeds):
        X,xt,xe,a,y,h,r,prob,tr,te=generate(seed,c)
        def lr(target,weights=None):return LogisticRegression(C=1,max_iter=1000).fit(xt,target,sample_weight=weights)
        models={'historical':lr(h[tr]),'review':lr(r[tr])}
        w=np.ones(len(tr))
        for g in (0,1):
          for v in (0,1):
            cell=(a[tr]==g)&(h[tr]==v)
            if not cell.any():raise ValueError('Empty reweighing cell')
            w[cell]=np.mean(a[tr]==g)*np.mean(h[tr]==v)/np.mean(cell)
        models['reweighing']=lr(h[tr],w)
        bounds=[.02,.05,.10] if sn=='reference' else [.05]
        for label,yt in [('historical',h),('review',r)]:
          for b in bounds:
            m=ExponentiatedGradient(LogisticRegression(C=1,max_iter=1000),TruePositiveRateParity(difference_bound=b),eps=.01,max_iter=50)
            m.fit(xt,yt[tr],sensitive_features=a[tr]);models[f'{label}_tpr_{b:.2f}']=m
        for name,m in models.items():
          for split,xx,ix in [('train',xt,tr),('test',xe,te)]:
            p=m.predict(xx,random_state=seed) if isinstance(m,ExponentiatedGradient) else m.predict(xx)
            vals,gr=metrics(y[ix],p,a[ix]);rows.append(dict(scenario=sn,seed=seed,method=name,split=split,**vals))
            for g in (0,1):group_rows.append(dict(scenario=sn,seed=seed,method=name,split=split,group=g,**gr[g]))
            if sn=='reference':
              target=r if name.startswith('review') else h
              for g in (0,1):
                mask=(a[ix]==g)&(target[ix]==1)
                diag.append(dict(seed=seed,method=name,split=split,group=g,label_tpr=p[mask].mean(),pooled_tpr=p[target[ix]==1].mean()))
          if sn=='reference' and name.startswith('review_tpr'):
            # Expected mixture predictions separate training constraint from one randomised draw.
            pp=m._pmf_predict(xt)[:,1]; pos=r[tr]==1
            group_devs=[abs(pp[pos&(a[tr]==g)].mean()-pp[pos].mean()) for g in (0,1)]
            diag.append(dict(seed=seed,method=name,split='expected_train',group=-1,label_tpr=max(group_devs),pooled_tpr=np.nan))
        p=(prob[te]>=.5).astype(int);vals,gr=metrics(y[te],p,a[te]);rows.append(dict(scenario=sn,seed=seed,method='oracle',split='test',**vals,bayes_expected_accuracy=np.maximum(prob[te],1-prob[te]).mean()))
        if sn=='reference' and seed==0:
          pd.DataFrame(dict(skill=X[:,0],experience=X[:,1],proxy=X[:,2],group=a,qualification=y,historical=h,reviewed=r,split=np.where(np.isin(np.arange(len(a)),tr),'train','test'))).to_csv(out/'synthetic_seed0.csv',index=False)
      print(sn,'complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False);pd.DataFrame(group_rows).to_csv(out/'group_metrics.csv',index=False);pd.DataFrame(diag).to_csv(out/'constraint_diagnostics.csv',index=False)
    df.groupby(['scenario','method','split']).agg({k:['mean','std'] for k in ['accuracy','sr_gap','tpr_gap','fpr_gap','ppv_gap']}).to_csv(out/'summary.csv')
    rng=np.random.default_rng(20261016); pairs=[]
    for sn in scenarios:
      dd=df[(df.scenario==sn)&(df.split=='test')]
      for comparison,ref in [('review','historical'),('reweighing','historical'),('historical_tpr_0.05','historical'),('review_tpr_0.02','review'),('review_tpr_0.05','review'),('review_tpr_0.10','review')]:
        for metric in ['accuracy','tpr_gap','sr_gap','fpr_gap']:
          pivot=dd.pivot(index='seed',columns='method',values=metric)
          if comparison not in pivot:continue
          d=(pivot[comparison]-pivot[ref]).to_numpy();boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
          pairs.append(dict(scenario=sn,comparison=comparison,reference=ref,metric=metric,n=len(d),mean_difference=d.mean(),sd_difference=d.std(ddof=1),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
    pd.DataFrame(pairs).to_csv(out/'paired_intervals.csv',index=False)
    (out/'environment.json').write_text(json.dumps(dict(version='1.0.0',seeds=args.seeds,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,fairlearn=fairlearn.__version__,scenarios=scenarios,bootstrap_replicates=10000,bootstrap_seed=20261016),indent=2))
    (out/'checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='checksums.json'},indent=2))
if __name__=='__main__':main()
