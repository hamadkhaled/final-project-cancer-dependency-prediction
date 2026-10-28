"""Publication-style static scientific charts from verified saved-result analysis.

Uses the installed ReportLab graphics plotting/drawing library; renderSVG exports
vector source. PNGs are generated separately with installed Node sharp.
No figure data are estimated from pixels or fabricated.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing,Line,Rect,Circle,String,Polygon
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=HERE.parent, help='Project directory with results/ and audited src/ summaries')
parser.add_argument('--output-dir', type=Path, default=HERE/'figures', help='Destination for four vector SVG figures')
args=parser.parse_args()
SRC=args.source.resolve(); AUDIT=SRC/'src'
FIG=args.output_dir.resolve(); FIG.mkdir(parents=True,exist_ok=True)
S=json.loads((AUDIT/'verified_summary.json').read_text())
DF=pd.read_csv(SRC/'results/main_pergene.csv')
NAVY='#162b46';BLUE='#2d6fa3';TEAL='#087f83';ORANGE='#be6330';GRAY='#728294'
LIGHT='#dde5ec';BG='#ffffff';PURPLE='#7464a0'
W=1400;H=850

def col(c): return HexColor(c)
def line(d,x1,y1,x2,y2,c=LIGHT,width=1,dash=None):
    d.add(Line(x1,y1,x2,y2,strokeColor=col(c),strokeWidth=width,strokeDashArray=dash))
def text(d,x,y,s,size=20,c=NAVY,anchor='start',bold=False):
    d.add(String(x,y,str(s),fontName='Helvetica-Bold' if bold else 'Helvetica',
                 fontSize=size,fillColor=col(c),textAnchor=anchor))
def rect(d,x,y,w,h,c,stroke=None):
    d.add(Rect(x,y,w,h,fillColor=col(c),strokeColor=col(stroke) if stroke else None))
def circle(d,x,y,r,c,stroke=None):
    d.add(Circle(x,y,r,fillColor=col(c),strokeColor=col(stroke) if stroke else None))
def base(title,subtitle):
    d=Drawing(W,H);rect(d,0,0,W,H,BG)
    text(d,65,H-65,title,31,bold=True)
    text(d,65,H-101,subtitle,20,GRAY)
    return d
def footer(d,t1,t2=''):
    line(d,65,87,W-65,87)
    text(d,65,59,t1,16,GRAY)
    if t2: text(d,65,33,t2,16,GRAY)
def save(d,name):
    renderSVG.drawToFile(d,str(FIG/f'{name}.svg'))
def xmap(v,x,w,lo,hi):return x+(v-lo)/(hi-lo)*w
def ymap(v,y,h,lo,hi):return y+(v-lo)/(hi-lo)*h
def fmt(v):return f'{v:.3f}' if abs(v)>.0005 else '0.000'

def fig_main():
    d=base('Expression and fusion retain predictive skill across unseen lineages',
           'Median training-mean skill (S_mean) across 581 selective dependency genes')
    arms=list(S['arm_labels']);colors=[GRAY,BLUE,GRAY,ORANGE,TEAL,PURPLE]
    ys=[620,540,460,380,300,220]
    lo,hi=-.14,.175
    for pi,sc in enumerate(['patient_grouped','lineage_blocked']):
        x=260+pi*590;w=440
        text(d,x+w/2,688,'S1 · Patient grouped' if pi==0 else 'S2 · Lineage blocked',23,bold=True,anchor='middle')
        for tick in [-.1,-.05,0,.05,.1,.15]:
            xx=xmap(tick,x,w,lo,hi)
            line(d,xx,175,xx,651,GRAY if tick==0 else LIGHT,1.5 if tick==0 else 1,dash=[5,4] if tick==0 else None)
            text(d,xx,148,f'{tick:.2f}',17,GRAY,anchor='middle')
        for yi,(a,c) in enumerate(zip(arms,colors)):
            if pi==0:text(d,65,ys[yi]-7,S['arm_labels'][a],21,bold=a in ['B_expression','E_multiomics'])
            z=next(r['skill'] for r in S['main'] if r['scheme']==sc and r['stratum']=='selective' and r['arm']==a)
            a0,a1,am=[xmap(z[k],x,w,lo,hi) for k in ['ci_low','ci_high','median']]
            line(d,a0,ys[yi],a1,ys[yi],c,4)
            line(d,a0,ys[yi]-7,a0,ys[yi]+7,c,2);line(d,a1,ys[yi]-7,a1,ys[yi]+7,c,2)
            circle(d,am,ys[yi],7,c)
            text(d,x+w+18,ys[yi]-6,fmt(z['median']),19,c,bold=a in ['B_expression','E_multiomics'])
    footer(d,'Points: median gene score. Whiskers: descriptive 95% percentile bootstrap over genes (2,000 resamples).',
             'Intervals are conditional on the saved cohort, folds and fitted models; they do not measure new-cohort uncertainty.')
    save(d,'fig1_selective_skill')

def fig_distribution():
    d=base('Fusion has a near-zero typical effect, with gains and losses for individual genes',
           'S2 lineage-blocked CV · paired ΔS_mean = fusion − expression · 581 selective genes')
    q=pd.read_csv(AUDIT/'fusion_gene_deltas_lineage_blocked.csv').sort_values('delta_fusion_expression')
    v=q.delta_fusion_expression.to_numpy()
    bins=[int((v<-.05).sum()),int(((v>=-.05)&(v<-.01)).sum()),int((abs(v)<=.01).sum()),
          int(((v>.01)&(v<=.05)).sum()),int((v>.05).sum())]
    labels=['< −0.05','−0.05 to −0.01','±0.01','0.01 to 0.05','> 0.05']
    colors=[ORANGE,'#d99a78',GRAY,'#63a8aa',TEAL]
    x=135;y=212;w=540;h=395
    text(d,x+w/2,658,'Most effects are small',24,anchor='middle',bold=True)
    for tick in [0,100,200,300,400]:
        yy=ymap(tick,y,h,0,400);line(d,x,yy,x+w,yy)
        text(d,x-18,yy-6,tick,17,GRAY,anchor='end')
    for i,(n,c,l) in enumerate(zip(bins,colors,labels)):
        bx=x+18+i*108;bw=72;hh=h*n/400
        rect(d,bx,y,bw,hh,c)
        text(d,bx+bw/2,y+hh+13,n,23,c,anchor='middle',bold=True)
        parts=l.split(' to ')
        for k,part in enumerate(parts):text(d,bx+bw/2,181-k*23,part if k==0 else 'to '+part,16,GRAY,anchor='middle')
    text(d,x+w/2,117,'Exploratory ΔS_mean thresholds',17,GRAY,anchor='middle')
    text(d,80,625,'Genes',17,GRAY)
    x=790;y=212;w=510;h=395;lo,hi=-.4,1.4
    text(d,x+w/2,658,'The full paired distribution has asymmetric tails',22,anchor='middle',bold=True)
    rect(d,x,ymap(-.05,y,h,lo,hi),w,h*.1/(hi-lo),'#eff4f7')
    for tick in [-.4,0,.4,.8,1.2]:
        yy=ymap(tick,y,h,lo,hi);line(d,x,yy,x+w,yy,GRAY if tick==0 else LIGHT,width=1.5 if tick==0 else 1)
        text(d,x-15,yy-6,f'{tick:.1f}',17,GRAY,anchor='end')
    for i,r in enumerate(q.itertuples(),1):
        vv=r.delta_fusion_expression;circle(d,xmap(i,x,w,1,581),ymap(vv,y,h,lo,hi),2.7,TEAL if vv>.05 else ORANGE if vv<-.05 else GRAY)
    for rank in [1,150,300,450,581]:text(d,xmap(rank,x,w,1,581),181,rank,17,GRAY,anchor='middle')
    text(d,x+w/2,145,'Gene rank (ordered by paired difference)',17,GRAY,anchor='middle')
    for sym,dy in [('EPC1',0),('WRN',0),('NDE1',10),('IGF1R',-9),('VPS4A',14)]:
        rr=q[q.symbol.eq(sym)].iloc[0];ii=int(np.where(q.symbol.eq(sym))[0][0])+1
        xx=xmap(ii,x,w,1,581);yy=ymap(rr.delta_fusion_expression,y,h,lo,hi)
        text(d,xx-12 if ii>400 else xx+12,yy+dy,sym,16,TEAL if rr.delta_fusion_expression>0 else ORANGE,
             anchor='end' if ii>400 else 'start',bold=True)
    footer(d,'Within ±0.01: 353/581 genes (60.8%). Gains >0.05: 42 genes. Losses >0.05: 24 genes.',
             'Thresholds describe the saved results; no per-gene significance test or prospective validation is claimed.')
    save(d,'fig2_fusion_distribution')

def fig_cases():
    d=base('Fusion can reduce error—or worsen it while rank correlation rises',
           'S2 lineage-blocked CV · five retrospectively selected examples · expression vs fusion')
    x=230;y=178;w=800;lo,hi=-1.4,.3
    for tick in [-1.25,-1,-.75,-.5,-.25,0,.25]:
        xx=xmap(tick,x,w,lo,hi);line(d,xx,y,xx,647,GRAY if tick==0 else LIGHT,1.5 if tick==0 else 1,dash=[5,4] if tick==0 else None)
        text(d,xx,146,f'{tick:.2f}',17,GRAY,anchor='middle')
    circle(d,245,681,7,BLUE);text(d,262,675,'Expression',19,BLUE)
    circle(d,414,681,7,TEAL);text(d,431,675,'Fusion',19,TEAL)
    text(d,1120,681,'ΔS_mean',18,GRAY,anchor='middle')
    text(d,1265,681,'Spearman',18,GRAY,anchor='middle')
    text(d,1265,657,'B → E',17,GRAY,anchor='middle')
    for i,sym in enumerate(['EPC1','WRN','NDE1','IGF1R','VPS4A']):
        yy=609-i*89
        b=next(r for r in S['cases'][sym] if r['scheme']=='lineage_blocked' and r['arm']=='B_expression')
        e=next(r for r in S['cases'][sym] if r['scheme']=='lineage_blocked' and r['arm']=='E_multiomics')
        xx1=xmap(b['r2'],x,w,lo,hi);xx2=xmap(e['r2'],x,w,lo,hi)
        delta=e['r2']-b['r2'];cc=TEAL if delta>0 else ORANGE
        text(d,65,yy-7,sym,24,bold=True)
        line(d,xx1,yy,xx2,yy,cc,5);circle(d,xx1,yy,8,BLUE);circle(d,xx2,yy,8,TEAL)
        text(d,xx1,yy-29,fmt(b['r2']),16,BLUE,anchor='middle')
        text(d,xx2,yy+16,fmt(e['r2']),16,TEAL,anchor='middle')
        text(d,1120,yy-6,f'{delta:+.3f}',20,cc,anchor='middle',bold=True)
        text(d,1265,yy-6,f'{b["spearman"]:.3f} → {e["spearman"]:.3f}',18,GRAY,anchor='middle')
    text(d,x+w/2,112,'Training-mean skill (S_mean)',19,GRAY,anchor='middle')
    footer(d,'EPC1 has the largest paired gain, yet fusion remains below the training-mean baseline.',
             'IGF1R and VPS4A rank slightly better under fusion, but their squared-error skill becomes negative.')
    save(d,'fig3_selected_cases')

def fig_metrics():
    d=base('Positive rank correlation does not guarantee useful squared-error skill',
           'S2 lineage-blocked fusion · each point is one of 581 selective dependency genes')
    q=DF[DF.scheme.eq('lineage_blocked')&DF.arm.eq('E_multiomics')&DF.stratum.eq('selective')]
    x=160;y=178;w=870;h=470;xlo,xhi=-.1,.8;ylo,yhi=-.75,.75
    for tick in [-.1,0,.2,.4,.6,.8]:
        xx=xmap(tick,x,w,xlo,xhi);line(d,xx,y,xx,y+h,GRAY if tick==0 else LIGHT)
        text(d,xx,147,f'{tick:.1f}',18,GRAY,anchor='middle')
    for tick in [-.6,-.3,0,.3,.6]:
        yy=ymap(tick,y,h,ylo,yhi);line(d,x,yy,x+w,yy,GRAY if tick==0 else LIGHT,width=1.5 if tick==0 else 1)
        text(d,x-18,yy-6,f'{tick:.1f}',18,GRAY,anchor='end')
    for r in q.itertuples():circle(d,xmap(r.spearman,x,w,xlo,xhi),ymap(r.r2,y,h,ylo,yhi),3.7,ORANGE if r.r2<0 else '#7197b3')
    for sym in ['IGF1R','VPS4A','WRN']:
        rr=q[q.symbol.eq(sym)].iloc[0];xx=xmap(rr.spearman,x,w,xlo,xhi);yy=ymap(rr.r2,y,h,ylo,yhi)
        circle(d,xx,yy,6,ORANGE if rr.r2<0 else TEAL)
        text(d,xx+12,yy-7,sym,19,ORANGE if rr.r2<0 else TEAL,bold=True)
    text(d,x+w/2,110,'Per-gene Spearman correlation (out-of-fold)',19,GRAY,anchor='middle')
    text(d,65,678,'S_mean',20,GRAY)
    rect(d,1071,368,260,229,'#f2f6f9')
    text(d,1094,555,'113 genes',28,ORANGE,bold=True)
    text(d,1094,520,'have negative skill.',20,GRAY)
    text(d,1094,478,'108 of those 113',24,NAVY,bold=True)
    text(d,1094,445,'still have positive',20,GRAY)
    text(d,1094,413,'rank correlation.',20,GRAY)
    footer(d,'S_mean compares squared prediction error against each fold’s training-mean predictor.',
             'Spearman measures ordering. Fold-specific calibration, offset or variance errors can separate these metrics.')
    save(d,'fig4_rank_vs_skill')

if __name__=='__main__':
    fig_main();fig_distribution();fig_cases();fig_metrics()
    print('Wrote',len(list(FIG.glob('fig*.svg'))),'verified scientific chart SVGs to',FIG)
