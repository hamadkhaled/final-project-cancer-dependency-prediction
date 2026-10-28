"""Reanalyse saved CSVs only. No model fitting, imputation, or synthetic outcomes.

Usage: python3 src/verify_saved_results.py [--source /path/to/project]
Writes verified_summary.json and machine-readable paired gene tables beside script.
Bootstrap intervals resample genes conditional on the saved cohort, folds and fits;
they are descriptive, not population or independent replication confidence limits.
"""
from pathlib import Path
import argparse, ast, hashlib, json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent
ARM_LABEL = {'A_lineage':'Lineage only','B_expression':'Expression',
             'C_copynumber':'Copy number','D_mutation':'Mutation',
             'E_multiomics':'Fusion','F_multi_plus_lin':'Fusion + lineage'}

def estimate(values, n_boot=2000):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {'n':0,'median':None,'ci_low':None,'ci_high':None}
    rng = np.random.default_rng(0)
    draws = np.median(v[rng.integers(0,len(v),size=(n_boot,len(v)))],axis=1)
    lo,hi = np.percentile(draws,[2.5,97.5])
    return {'n':len(v),'median':float(np.median(v)),
            'ci_low':float(lo),'ci_high':float(hi)}

def stats(values):
    v = np.asarray(values,dtype=float)
    v = v[np.isfinite(v)]
    out = estimate(v)
    if len(v):
        out.update(min=float(v.min()),max=float(v.max()),mean=float(v.mean()),
                   q10=float(np.quantile(v,.1)),q25=float(np.quantile(v,.25)),
                   q75=float(np.quantile(v,.75)),q90=float(np.quantile(v,.9)),
                   n_positive=int((v>0).sum()),n_negative=int((v<0).sum()))
    return out

def series(df,scheme,arm,stratum='selective',metric='r2'):
    d=df[df.scheme.eq(scheme)&df.arm.eq(arm)]
    if stratum!='all': d=d[d.stratum.eq(stratum)]
    return d.set_index('gene')[metric]

def paired(a,b):
    j=pd.concat([a.rename('a'),b.rename('b')],axis=1,join='inner').dropna()
    out=stats(j.a-j.b)
    out.update(n_better=int((j.a>j.b).sum()),n_worse=int((j.a<j.b).sum()))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    source=ap.parse_args().source.resolve();res=source/'results'
    df=pd.read_csv(res/'main_pergene.csv');meta=pd.read_csv(source/'data/meta.csv')
    assert not df.duplicated(['scheme','arm','gene']).any()
    assert len(df)==11772 and df.gene.nunique()==981
    assert df.groupby(['scheme','arm']).size().eq(981).all()
    out={'provenance':{'source':source.name,'analysis':'saved-result reanalysis only',
         'bootstrap':{'unit':'gene','n_resamples':2000,'seed':0,'interval':'95% percentile',
           'interpretation':'Conditional on saved cohort, folds, selected genes and trained models; genes may be dependent. Not uncertainty over new cohorts or model retraining.'},
         'missing_raw_inputs':[f for f in ['Y.npy','expr.npy','cn.npy','mut.npy','names.json'] if not (source/'data'/f).exists()],
         'prediction_arrays_present':[p.name for p in sorted(res.glob('*preds*.npy'))],
         'file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(res.glob('*.csv'))}},
         'integrity':{'rows':len(df),'genes':df.gene.nunique(),'duplicate_scheme_arm_gene_rows':0,
         'gene_counts_by_stratum':df.drop_duplicates('gene').stratum.value_counts().to_dict()},
         'arm_labels':ARM_LABEL,'main':[],'paired':[],'fusion_vs_expression':{},
         'ablation_comparisons':[],'rf_matched':[],'cases':{},'spearman_by_gene_set':[],
         'metric_discordance':[], 'threshold_sensitivity':[]}
    for (sc,arm,st),d in df.groupby(['scheme','arm','stratum'],sort=False):
        row={'scheme':sc,'arm':arm,'stratum':st,'skill':stats(d.r2),'spearman':stats(d.spearman),
             'alpha_median_across_genes':float(d.alpha_median.median()),
             'n_gene_median_alpha_at_grid_max':int(d.alpha_median.eq(1e5).sum())}
        out['main'].append(row)
        out['metric_discordance'].append({'scheme':sc,'arm':arm,'stratum':st,
              'n_genes':len(d),'n_negative_skill':int(d.r2.lt(0).sum()),
              'n_negative_skill_positive_spearman':int((d.r2.lt(0)&d.spearman.gt(0)).sum())})
    for sc in df.scheme.unique():
        for st in ['selective','common_essential','flat','all']:
            for a,b in [(a,'A_lineage') for a in list(ARM_LABEL)[1:]]+[
                    ('E_multiomics','B_expression'),('F_multi_plus_lin','E_multiomics')]:
                out['paired'].append({'scheme':sc,'stratum':st,'a':a,'b':b,
                         'delta_skill':paired(series(df,sc,a,st),series(df,sc,b,st))})
        d=series(df,sc,'E_multiomics')-series(df,sc,'B_expression')
        out['fusion_vs_expression'][sc]={'delta_skill':stats(d),
            'counts':{str(t):{'benefit':int((d>t).sum()),'loss':int((d<-t).sum()),
                          'within_inclusive':int((d.abs()<=t).sum())} for t in [.01,.05]},
            'n_exact_zero':int((d==0).sum()),
            'top_benefits':d.nlargest(10).rename('delta').reset_index().to_dict('records'),
            'top_losses':d.nsmallest(10).rename('delta').reset_index().to_dict('records')}
        gene_table=df[df.scheme.eq(sc)&df.stratum.eq('selective')].pivot(
            index=['gene','symbol','gene_std'],columns='arm',values='r2').reset_index()
        gene_table['delta_fusion_expression']=gene_table.E_multiomics-gene_table.B_expression
        gene_table.sort_values('delta_fusion_expression',ascending=False).to_csv(HERE/f'fusion_gene_deltas_{sc}.csv',index=False)
        for arm in ['A_lineage','B_expression','E_multiomics']:
            for st in ['all','selective','common_essential','flat']:
                out['spearman_by_gene_set'].append({'scheme':sc,'arm':arm,'gene_set':st,
                    'spearman':stats(series(df,sc,arm,st,'spearman'))})
    definitions=[('abl_self','E_multiomics','E_multiomics','Self-feature exclusion'),
        ('abl_permlin','A_lineage','A_lineage','Permuted lineage'),
        ('abl_k100','E_multiomics','E_multiomics','Fusion k=100'),
        ('abl_k600','E_multiomics','E_multiomics','Fusion k=600'),
        ('abl_logcn','C_copynumber','C_copynumber','Log-transformed copy number'),
        ('abl_logcn','E_multiomics','E_multiomics','Fusion with log copy number'),
        ('abl_k20_lowdim','D_mutation','D_mutation','Mutation k=20'),
        ('abl_k50_lowdim','D_mutation','D_mutation','Mutation k=50'),
        ('abl_k100_lowdim','D_mutation','D_mutation','Mutation k=100')]
    for tag,a,b,label in definitions:
        abl=pd.read_csv(res/f'{tag}_pergene.csv')
        for sc in abl.scheme.unique():
            av=series(abl,sc,a);bv=series(df,sc,b)
            out['ablation_comparisons'].append({'tag':tag,'comparison':label,'scheme':sc,'arm':a,
                'delta_skill':paired(av,bv),'ablation_skill':stats(av)})
    rf=pd.read_csv(res/'abl_rf_pergene.csv')
    out['rf_arms_available']=rf.arm.unique().tolist()
    for sc in rf.scheme.unique():
        for a in rf.arm.unique():
            rv=series(rf,sc,a);bv=series(df,sc,a).loc[rv.index]
            out['rf_matched'].append({'scheme':sc,'arm':a,'n_genes':len(rv),
               'rf_skill':stats(rv),'ridge_skill_matched':stats(bv),'delta_skill_rf_minus_ridge':paired(rv,bv)})
    sub=df[df.scheme.eq('lineage_blocked')&df.stratum.eq('selective')]
    gene_sd=sub.drop_duplicates('gene').set_index('gene').gene_std
    for t in [.25,.30,.35,.45]:
        keep=gene_sd[gene_sd>=t].index
        for a,b in [('E_multiomics','A_lineage'),('E_multiomics','B_expression')]:
            out['threshold_sensitivity'].append({'scheme':'lineage_blocked','min_gene_std':t,
                'a':a,'b':b,'delta_skill':paired(series(df,'lineage_blocked',a).loc[keep],
                                                    series(df,'lineage_blocked',b).loc[keep])})
    for symbol in ['WRN','NDE1','EPC1','IGF1R','VPS4A']:
        d=df[df.symbol.eq(symbol)]
        out['cases'][symbol]=d[['scheme','arm','r2','spearman','stratum']].to_dict('records')
    out['legacy_pooled_spearman']=pd.read_csv(res/'fig1_metric_inflation.csv').to_dict('records')
    out['legacy_pooled_caveat']='Pooled correlation uses all 981 genes while legacy pergene column uses 581 selective genes. The matched-set pooled correlation cannot be recovered without labels/predictions.'
    out['feature_composition_first_S2_training_fold']=json.loads((res/'feature_composition.json').read_text())
    pc=pd.read_csv(res/'paralog_check.csv')
    pc['decile']=pd.qcut(pc.r2.rank(method='first'),10,labels=False)
    out['symbol_root_heuristic']={'n':len(pc),'n_flagged':int(pc.is_paralog.sum()),
        'n_self':int(pc.is_self.sum()),'top_decile_n':int(pc.decile.eq(9).sum()),
        'top_decile_flagged':int(pc[pc.decile.eq(9)].is_paralog.sum()),
        'high_skill_n':int(pc.r2.gt(.3).sum()),'high_skill_flagged':int(pc[pc.r2.gt(.3)].is_paralog.sum()),
        'caveat':'Unvalidated gene-symbol-root heuristic; feature correlations selected on full cohort; descriptive hypothesis generation only.'}
    env={'np':np,'pd':pd,'SEED':0,'N_FOLDS':5,'MIN_LINEAGE_SIZE':10,'Dataset':object}
    tree=ast.parse((source/'src/common.py').read_text())
    names={'_group_kfold','splits_patient_grouped','splits_lineage_blocked'}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'common.py','exec'),env)
    class DS: pass
    ds=DS();ds.meta=meta;ds.lineage=meta.OncotreeLineage.fillna('Unknown').to_numpy()
    out['cohort']={'n_cell_lines':len(meta),'n_lineages':int(meta.OncotreeLineage.nunique()),
        'n_patients':int(meta.PatientID.nunique()),'missing_patient_ids':int(meta.PatientID.isna().sum()),
        'patient_groups_with_multiple_lines':int(meta.PatientID.value_counts().gt(1).sum()),
        'cell_lines_in_repeated_patient_groups':int(meta.PatientID.duplicated(keep=False).sum()),
        'lineage_composition':meta.OncotreeLineage.fillna('Unknown').value_counts().to_dict()}
    out['folds']={}
    archived=pd.read_csv(res/'fold_assignments.csv')
    out['fold_source']='results/fold_assignments.csv (archived partitions used by the original experiment)'
    out['regenerated_splitter_matches_archive']={}
    for sc,fn in [('patient_grouped',env['splits_patient_grouped']),('lineage_blocked',env['splits_lineage_blocked'])]:
        block=archived[archived.scheme.eq(sc)].sort_values('row_index')
        assert np.array_equal(block.row_index.to_numpy(),np.arange(len(meta))) and np.array_equal(block.ModelID.astype(str).to_numpy(),meta.ModelID.astype(str).to_numpy())
        labels=block.fold.astype(int).to_numpy()
        folds=[(np.flatnonzero(labels!=f),np.flatnonzero(labels==f)) for f in range(5)]
        regen=np.full(len(meta),-1)
        for f,(tr,te) in enumerate(fn(ds)): regen[te]=f
        out['regenerated_splitter_matches_archive'][sc]=bool(np.array_equal(regen,labels))
        out['folds'][sc]=[]
        all_test=np.concatenate([te for tr,te in folds]);assert np.array_equal(np.sort(all_test),np.arange(len(meta)))
        for i,(tr,te) in enumerate(folds,1):
            pats_tr=set(meta.iloc[tr].PatientID.fillna('NA'));pats_te=set(meta.iloc[te].PatientID.fillna('NA'))
            overlap=pats_tr&pats_te
            out['folds'][sc].append({'fold':i,'n_train':len(tr),'n_test':len(te),
              'n_patient_groups_overlap':len(overlap),'patient_groups_overlap':sorted(overlap),
              'n_lineages_overlap':len(set(ds.lineage[tr])&set(ds.lineage[te])),
              'test_lineage_counts':meta.iloc[te].OncotreeLineage.fillna('Unknown').value_counts().to_dict()})
    legacy_main=pd.read_csv(res/'main_table_main.csv')
    diffs=[]
    for row in out['main']:
        old=legacy_main[legacy_main.scheme.eq(row['scheme'])&legacy_main.arm.eq(row['arm'])&legacy_main.stratum.eq(row['stratum'])].iloc[0]
        diffs += [abs(float(old[a])-row['skill'][b]) for a,b in [('median_r2','median'),('ci_lo','ci_low'),('ci_hi','ci_high')]]
    assert max(diffs)<1e-12
    out['integrity']['independent_summary_max_absolute_difference_from_archived_main_table']=max(diffs)
    (HERE/'verified_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print('Verified',len(df),'rows;',out['integrity']['gene_counts_by_stratum'])
    for sc,item in out['fusion_vs_expression'].items(): print(sc,item['delta_skill'],item['counts'])
    for row in out['rf_matched']: print('RF',row['scheme'],row['arm'],row['delta_skill_rf_minus_ridge'])
    for sc,folds in out['folds'].items(): print(sc,[(f['n_train'],f['n_test'],f['n_patient_groups_overlap']) for f in folds])

if __name__=='__main__': main()
