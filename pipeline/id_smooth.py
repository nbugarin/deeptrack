"""
Temporal smoothing of IDs between nvtracker and the OSD.

Flicker: the same physical track alternately receives two IDs from frame to frame.
The correction works in two steps:
  1. A chain of detections is built in space and time (consecutive frames,
     high IoU); it is one physical object regardless of the assigned ID.
  2. Each chain is given one ID: the one that occurs most often.

It does not use the ground truth in any way; it works only on the tracker output,
so it can be applied in real time.

The experiments of the article use only the loaders load_gt, load_tr, pick and iou of this file;
the smoothing itself is not applied to any result of the article.
"""
import json, numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter

LINK_IOU = 0.5   # minimum overlap for linking detections in consecutive frames
MAXGAP   = 3     # how many frames an object may be missing before the chain is broken

def load_gt(p,s):
    g={}
    for l in open(p):
        l=l.strip()
        if not l: continue
        f=l.split(',')
        g.setdefault(int(f[0]),[]).append({'id':int(f[1]),
          'box':[float(f[2])*s,float(f[3])*s,float(f[4])*s,float(f[5])*s]})
    return g

def load_tr(p):
    d={}
    for fd in json.load(open(p)):
        if fd['detections']:
            d[fd['frame']]=[{'id':x['id'],'box':list(x['box'])} for x in fd['detections']]
    return d

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3])
    i=max(0,x2-x1)*max(0,y2-y1); u=a[2]*a[3]+b[2]*b[3]-i
    return i/u if u>0 else 0

def pick(gp,tr):
    best=(1.0,-1)
    for s in [1.0,1.5]:
        g=load_gt(gp,s); fr=sorted(set(g)&set(tr))[:400]
        h=sum(1 for f in fr for o in g[f] if any(iou(o['box'],t['box'])>0.5 for t in tr[f]))
        if h>best[1]: best=(s,h)
    return best[0]

def smooth(tr):
    frames=sorted(tr)
    # every detection is a node (frame, index)
    chain={}          # (f,i) -> chain id
    nxt=0
    active=[]         # list of (last_frame, box, chain)
    for f in frames:
        dets=tr[f]
        free=list(range(len(dets)))
        act=[a for a in active if f-a[0]<=MAXGAP]
        if act and free:
            C=np.zeros((len(act),len(free)))
            for x,a in enumerate(act):
                for y,d in enumerate(free):
                    C[x,y]=1-iou(a[1],dets[d]['box'])
            r,c=linear_sum_assignment(C)
            taken=set(); newact=[]
            for x,y in zip(r,c):
                if 1-C[x,y]>=LINK_IOU:
                    d=free[y]; chain[(f,d)]=act[x][2]; taken.add(d)
                    newact.append((f,dets[d]['box'],act[x][2]))
            keep=[a for k,a in enumerate(act) if k not in set(r[i] for i in range(len(r)) if 1-C[r[i],c[i]]>=LINK_IOU)]
            active=[a for a in keep if f-a[0]<=MAXGAP]+newact
            free=[d for d in free if d not in taken]
        else:
            active=[a for a in active if f-a[0]<=MAXGAP]
        for d in free:
            chain[(f,d)]=nxt; active.append((f,dets[d]['box'],nxt)); nxt+=1

    # majority ID per chain
    votes=defaultdict(Counter)
    for f in frames:
        for i,d in enumerate(tr[f]):
            votes[chain[(f,i)]][d['id']]+=1
    best={ch:v.most_common(1)[0][0] for ch,v in votes.items()}

    out={}; changed=0
    for f in frames:
        row=[]
        for i,d in enumerate(tr[f]):
            nid=best[chain[(f,i)]]
            if nid!=d['id']: changed+=1
            row.append({'id':nid,'box':d['box']})
        out[f]=row
    nchains=len([c for c in votes if len(votes[c])>1])
    return out, changed, nchains

def evaluate(gt,tr):
    frames=sorted(set(gt)|set(tr))
    TP=FP=FN=SW=0; prev={}
    gc=defaultdict(int); tc=defaultdict(int); M=defaultdict(int)
    for f in frames:
        for o in gt.get(f,[]): gc[o['id']]+=1
        for o in tr.get(f,[]): tc[o['id']]+=1
    for f in frames:
        G=gt.get(f,[]); T=tr.get(f,[]); pr=[]
        if G and T:
            C=np.array([[1-iou(a['box'],b['box']) for b in T] for a in G])
            r,c=linear_sum_assignment(C)
            pr=[(i,j) for i,j in zip(r,c) if 1-C[i,j]>=0.5]
        for i,j in pr:
            TP+=1; M[(G[i]['id'],T[j]['id'])]+=1
            if G[i]['id'] in prev and prev[G[i]['id']]!=T[j]['id']: SW+=1
            prev[G[i]['id']]=T[j]['id']
        FN+=len(G)-len(pr); FP+=len(T)-len(pr)
    n=sum(gc.values())
    MOTA=1-(FP+FN+SW)/max(n,1)
    gi={v:k for k,v in enumerate(sorted(gc))}; ti={v:k for k,v in enumerate(sorted(tc))}
    N=max(len(gi),len(ti)); cost=np.zeros((N,N))
    for (a,b),v in M.items(): cost[gi[a],ti[b]]=-v
    for a in gc: cost[gi[a],:]+=gc[a]/2
    for b in tc: cost[:,ti[b]]+=tc[b]/2
    r,c=linear_sum_assignment(cost); S=set(zip(r,c))
    IDTP=sum(M.get((s,t),0) for s in gc for t in tc if (gi[s],ti[t]) in S)
    IDF1=2*IDTP/max(sum(gc.values())+sum(tc.values()),1)
    hs=[]
    for al in np.arange(0.05,1.0,0.05):
        tp=fp=fn=0; mm=defaultdict(int)
        for f in frames:
            G=gt.get(f,[]); T=tr.get(f,[]); pr=[]
            if G and T:
                C=np.array([[1-iou(a['box'],b['box']) for b in T] for a in G])
                rr,cc=linear_sum_assignment(C)
                pr=[(i,j) for i,j in zip(rr,cc) if 1-C[i,j]>=al]
            for i,j in pr: mm[(G[i]['id'],T[j]['id'])]+=1
            tp+=len(pr); fn+=len(G)-len(pr); fp+=len(T)-len(pr)
        D=tp/max(tp+fn+fp,1)
        A=sum(v*(v/(gc[g]+tc[t]-v)) for (g,t),v in mm.items())/tp if tp else 0
        hs.append(np.sqrt(D*A))
    return dict(MOTA=round(MOTA,3),IDF1=round(IDF1,3),
                HOTA=round(float(np.mean(hs)),3),IDSW=SW,nIDs=len(tc))

CFG={1:('/workspace/log_nvsort_1.json','/workspace/log_deep_1.json'),
     2:('/workspace/tracking_log_nvsort_2.json','/workspace/tracking_log_2.json'),
     3:('/workspace/log_nvsort_3.json','/workspace/log_deep_3.json'),
     4:('/workspace/log_nvsort_4.json','/workspace/log_deep_4.json'),
     5:('/workspace/log_nvsort_5.json','/workspace/log_deep_5.json')}

agg=defaultdict(lambda: defaultdict(list))
for v in sorted(CFG):
    gtp='/workspace/gt_%d.txt'%v
    print('--- Video %d ---'%v)
    for name,path in zip(['NvSORT','NvDeepSORT'],CFG[v]):
        tr=load_tr(path); s=pick(gtp,tr); gt=load_gt(gtp,s)
        b=evaluate(gt,tr)
        sm,ch,nc=smooth(tr)
        a=evaluate(gt,sm)
        print('  %-11s before '%name,b)
        print('  %-11s after  '%'',a,'  (%d detections corrected, %d unstable chains)'%(ch,nc))
        for k in ['MOTA','IDF1','HOTA','IDSW']:
            agg[name]['b'+k].append(b[k]); agg[name]['a'+k].append(a[k])

print()
print('='*62)
print('MEAN over 5 sequences')
for name in ['NvSORT','NvDeepSORT']:
    print(' ',name)
    for k in ['MOTA','IDF1','HOTA','IDSW']:
        b=np.mean(agg[name]['b'+k]); a=np.mean(agg[name]['a'+k])
        f='%.3f' if k!='IDSW' else '%.1f'
        print(('    %-5s  before'+f+'   after   '+f+'   change  '+f)%(k,b,a,a-b))
