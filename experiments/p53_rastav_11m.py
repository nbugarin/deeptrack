#!/usr/bin/env python3
"""p53: DECOMPOSITION AND SENSITIVITY OF THE FINAL VARIANT OF OUR TRACKER -- YOLO11m, 16 SSMOT sequences (for IEEE, SSMOT only)

Plan written BEFORE the run (project: claude/p53_rastav_11m_plan.md). Everything is descriptive; nothing is selected on these results.
Variants (all with YOLO11m, config_infer_11m.txt from p44):
  cls    class penalty only   = baseline + classGate 0, classPenalty 0.5
  prob   probation only       = baseline + confHigh 0.8, maxAgeSec 3.3333, minHitsSec 0.6667 (the hard class condition stays)
         (cls + prob = exactly the final variant; checked in the code)
  l02    final with classPenalty 0.2       l03  final with classPenalty 0.3       c07  final with confHigh 0.7
  base_t, final_t   baseline and final AGAIN, with timing -> must be identical to log_p52_* (reproducibility check) and give the speed
Protections as in p52: track.h -> track.h.bak_p53 (always restored and compiled), engines -> .p53bak (checked after every run).
Run (GPU free):           nohup python3 /workspace/p53_rastav_11m.py > /workspace/p53_out.txt 2>&1 &
Monitoring:               tail -3 /workspace/p53_out.txt
Evaluation only:          python3 /workspace/p53_rastav_11m.py --eval
If interrupted:           python3 /workspace/p53_rastav_11m.py --vrati   (then rerun -- it resumes)
Output: /workspace/scpilot/p53_rastav_11m.txt (+ scpilot/p53/p53_sve.json)
"""
import os, re, sys, json, time, shutil, hashlib, subprocess
import numpy as np
for a,b in (('float',float),('int',int),('bool',bool)):
    if not hasattr(np,a): setattr(np,a,b)
W=os.environ.get('P53_W','/workspace'); S=W+'/scpilot'; O=S+'/p53'; LOCK=O+'/p53.lock'; RE=O+'/ponovno'
SRC=W+'/mytracker/src/track.h'; BAK=SRC+'.bak_p53'; LIB=W+'/mytracker/libmoj.so'
PROBE=W+'/probe_moj.py'; CFG=W+'/config_tracker_NvSORT.yml'
INFER=W+'/DeepStream-Yolo/config_infer_11m.txt'
WD=W+'/det_glavni/B_yolo11m/weights'; ENG11=WD+'/best_11m_b1_gpu0_fp16.engine'; LAB11=WD+'/labels.txt'
PROT0=[W+'/model_b1_gpu0_fp16.engine',W+'/best_0904.engine',W+'/osnet_vessel.engine',ENG11]
SS16=[1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]; SS15=[s for s in SS16 if s!=10]
VID=W+'/SSMOT_video/video/SSMOT_%d.mp4'; GT=W+'/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
BASE={'maxAge':100,'wMeas':0.15,'galleryMax':5,'wReid':0.8,'maxCost':0.75,'inertia':0.10,'detNms':0.70,'minHits':20}
EXTRA_BASE={'reidAfterN':0}
OFF={'classGate':lambda v: v==1,'confHigh':lambda v: v>1,'maxAgeSec':lambda v: v<=0,'minHitsSec':lambda v: v<=0}
FINAL={'classGate':0,'classPenalty':0.5,'confHigh':0.8,'maxAgeSec':3.3333,'minHitsSec':0.6667}   # exactly as the p43/p52 "final"
CLS={'classGate':0,'classPenalty':0.5}; PROB={'confHigh':0.8,'maxAgeSec':3.3333,'minHitsSec':0.6667}
assert dict(CLS,**PROB)==FINAL and not set(CLS)&set(PROB)
VARS=[('base_t',{},'baseline (again, with timing)'),('cls',CLS,'class penalty only'),('prob',PROB,'probation only'),
      ('final_t',FINAL,'final (again, with timing)'),('l02',dict(FINAL,classPenalty=0.2),'final, class penalty 0.2'),
      ('l03',dict(FINAL,classPenalty=0.3),'final, class penalty 0.3'),('c07',dict(FINAL,confHigh=0.7),'final, confidence threshold 0.7')]
VN={v:n for v,_,n in VARS}; VC={v:c for v,c,_ in VARS}
def outp(v,s): return (RE+'/%s_%d.json'%(v[:-2],s)) if v.endswith('_t') else W+'/log_p53_%s_ss_%d.json'%(v,s)
P52=W+'/log_p52_%s_ss_%d.json'
LOCKS=[S+'/p43.lock',S+'/p27.lock']+[S+'/p%d/p%d.lock'%(i,i) for i in (44,45,46,47,48,52)]
T_RUN=900
REINSTALL=('apt-get update && apt-get install -y libeigen3-dev && pip install --no-deps git+https://github.com/JonathonLuiten/TrackEval.git '
           '&& python3 -c "import scipy, cv2, trackeval; print(\'ALL OK\')"')
L=[]
def P(x=''): print(x,flush=True); L.append(x)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def json_ok(p):
    try:
        with open(p) as f: d=json.load(f)
        return isinstance(d,list) and len(d)>0
    except Exception: return False
def alive(lock):
    if not os.path.exists(lock): return None
    pid=open(lock).read().strip()
    return pid if pid and os.path.exists('/proc/'+pid) else None
def stat(p): return (os.stat(p).st_size,os.stat(p).st_mtime_ns) if os.path.exists(p) else None
def frames(p):
    out={}
    for fd in json.load(open(p)):
        d=fd.get('detections') or []
        if d: out[int(fd['frame'])]=sorted((int(x['id']),tuple(round(float(v),2) for v in x['box'][:4])) for x in d)
    return out
def stanje():
    try: return json.load(open(O+'/stanje.json'))
    except Exception: return {}
def spremi(st): json.dump(st,open(O+'/stanje.json','w'),indent=1)
def fps(path):
    try:
        import cv2; f=cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)
        if f and f>0: return float(f)
    except Exception: pass
    return 25.0
# ======================================================================= track.h (same as p43/p52)
RX=lambda n: re.compile(r'^(\s*(int|float|double|bool)\s+'+re.escape(n)+r'\s*=\s*)([^;]+)(;)',re.M)
def getv(txt,n):
    m=list(RX(n).finditer(txt))
    if len(m)!=1: return None,None
    return m[0].group(2),float(m[0].group(3).strip().rstrip('f'))
def setv(txt,n,v):
    m=list(RX(n).finditer(txt)); assert len(m)==1,n
    ty=m[0].group(2); old=m[0].group(3).strip()
    val=('%d'%int(round(v))) if ty in ('int','bool') else ('%g'%v)+('f' if old.endswith('f') else '')
    if ty not in ('int','bool') and '.' not in val and 'e' not in val: val=val.replace('f','')+('.0f' if old.endswith('f') else '.0')
    return txt[:m[0].start(3)]+val+txt[m[0].end(3):]
def check_base(txt):
    bad=[]
    for n,v in list(BASE.items())+list(EXTRA_BASE.items()):
        ty,x=getv(txt,n)
        if x is None: bad.append('%s not found exactly once'%n)
        elif abs(x-v)>1e-6: bad.append('%s = %g (should be %g)'%(n,x,v))
    for n,okf in OFF.items():
        ty,x=getv(txt,n)
        if x is not None and not okf(x): bad.append('%s = %g (should be switched off)'%(n,x))
    for n in FINAL:
        if getv(txt,n)[1] is None: bad.append('%s (final variant) not found exactly once'%n)
    return bad
def build():
    t0=time.time()
    r=subprocess.run('cd %s/mytracker && make clean >/dev/null 2>&1; make 2>&1'%W,shell=True,capture_output=True,text=True)
    if r.returncode!=0 or not os.path.exists(LIB) or os.path.getmtime(LIB)<t0-2:
        raise RuntimeError('compilation failed:\n'+(r.stdout or '')[-1500:]+(r.stderr or '')[-1500:])
def vrati():
    if os.path.exists(BAK):
        shutil.copy(BAK,SRC)
        try: build()
        except Exception as e: print('!! compilation of the baseline track.h failed: %s'%e,flush=True)
        if open(SRC).read()==open(BAK).read(): os.remove(BAK)
        print('   track.h restored to the baseline configuration and compiled.',flush=True)
    for p in PROT0:
        b=p+'.p53bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p)==sha(b): os.remove(b); print('   %s: untouched, copy deleted'%os.path.basename(p),flush=True)
        else:
            if os.path.exists(p): os.replace(p,p+'.p53_nepoznat')
            os.replace(b,p); print('   %s: RESTORED from the backup copy'%os.path.basename(p),flush=True)
# ======================================================================= GPU
def runs():
    try:
        import scipy, trackeval, cv2   # noqa: F401
    except Exception as e: sys.exit('!! missing package (%s). The container was probably restarted -- first run:\n   %s'%(e,REINSTALL))
    os.makedirs(RE,exist_ok=True)
    if alive(LOCK): sys.exit('!! p53 is already running (PID %s) -- do not run twice.'%alive(LOCK))
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running (PID %s) -- run p53 when it finishes.'%(os.path.basename(lk).split('.')[0],alive(lk)))
    if os.environ.get('P53_NOGPU')!='1':
        try:
            used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
            print('GPU memory in use: %d MiB'%used,flush=True)
            if used>1500: sys.exit('GPU is busy (>1500 MiB) -- something else is running on the GPU. Wait until it finishes.')
        except (IndexError,ValueError,FileNotFoundError): print('warning: nvidia-smi gave no data, continuing',flush=True)
        if not os.path.isdir('/usr/include/eigen3'): sys.exit('!! no Eigen. First: apt-get install -y libeigen3-dev')
    if os.path.exists(BAK) or any(os.path.exists(p+'.p53bak') for p in PROT0):
        print('the previous run was interrupted -- restoring track.h and checking the engines:',flush=True); vrati()
    need=[PROBE,CFG,INFER,ENG11,SRC,W+'/id_smooth.py']+[VID%s for s in SS16]+[GT%s for s in SS16]+[P52%(v,s) for v in ('base','final') for s in SS16]
    miss=[p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! missing (p52 must be done):\n   '+'\n   '.join(miss))
    if shutil.disk_usage(W).free/2**30<3: sys.exit('!! less than 3 GB free on the disk')
    orig=open(SRC).read(); bad=check_base(orig)
    if bad: sys.exit('!! track.h is NOT at the baseline configuration -- nothing was run:\n   '+'\n   '.join(bad)+'\n   Send this output to Claude.')
    print('track.h at the baseline configuration (checked); minHitsHigh = %s (not changed)'%('%g'%getv(orig,'minHitsHigh')[1] if getv(orig,'minHitsHigh')[1] is not None else 'none'),flush=True)
    st=stanje(); h_inf=sha(INFER)
    if 'infer_sha' in st and st['infer_sha']!=h_inf: sys.exit('!! config_infer_11m.txt changed since the first p53 run -- tell Claude')
    st['infer_sha']=h_inf; st.setdefault('vrijeme',{}); spremi(st)
    todo={v:[s for s in SS16 if not json_ok(outp(v,s))] for v,_,_ in VARS}
    n_all=sum(len(t) for t in todo.values())
    print('runs remaining: %d (7 variants x 16 SSMOT sequences)'%n_all,flush=True)
    PROT=[p for p in PROT0 if os.path.exists(p)]
    print('protected engines: %s'%', '.join(os.path.basename(p) for p in PROT),flush=True)
    shutil.copy2(SRC,BAK); open(LOCK,'w').write(str(os.getpid())); sh={}
    t0=time.time(); done=[0]
    try:
        for p in PROT:
            sh[p]=sha(p); shutil.copy2(p,p+'.p53bak')
            if sha(p+'.p53bak')!=sh[p]: sys.exit('!! backup copy %s is not valid -- STOP'%p)
        def one(v,s):
            out=outp(v,s); st_={p:stat(p) for p in PROT}; env=dict(os.environ,MYTRACKER_FPS='%g'%fps(VID%s)); rc=None
            if os.path.exists(out): os.remove(out)
            ta=time.time()
            try:
                r=subprocess.run(['python3',PROBE,VID%s,CFG,out,INFER],cwd=W,capture_output=True,text=True,env=env,timeout=T_RUN)
                rc=r.returncode; err=(r.stdout or '')[-2500:]+'\n----\n'+(r.stderr or '')[-3500:]
            except subprocess.TimeoutExpired: err='time limit exceeded (%d s)'%T_RUN
            dt=time.time()-ta
            ch=[os.path.basename(p) for p in PROT if stat(p)!=st_[p]]
            if ch:
                if os.path.exists(out): os.remove(out)
                open(O+'/greska.txt','w').write(err)
                sys.exit('!! %s changed during the run (%s, SSMOT_%d) -- restoring from the copy; send to Claude: tail -40 /workspace/scpilot/p53/greska.txt'%(', '.join(ch),v,s))
            if json_ok(out):
                st['vrijeme']['%s|%d'%(v,s)]=[len(json.load(open(out))),dt]; spremi(st); return True
            if os.path.exists(out): os.remove(out)
            open(O+'/greska.txt','w').write('%s SSMOT_%d, return code %s\n'%(v,s,rc)+err); return False
        for v,cfg,name in VARS:
            if not todo[v]: continue
            txt=orig
            for k,x in cfg.items(): txt=setv(txt,k,x)
            open(SRC,'w').write(txt); build()
            print('%s: tracker compiled (%s)'%(name,', '.join('%s %g'%(k,x) for k,x in cfg.items()) or 'baseline track.h'),flush=True)
            for s in todo[v]:
                ok=one(v,s) or one(v,s)
                if not ok: sys.exit('!! %s, SSMOT %d failed in the second attempt too -- send to Claude: tail -40 /workspace/scpilot/p53/greska.txt'%(name,s))
                done[0]+=1; el=time.time()-t0
                print('  [%3d/%d] %-32s SSMOT %-2d ok   (about %.0f min left)'%(done[0],n_all,name,s,el/done[0]*(n_all-done[0])/60),flush=True)
    finally:
        shutil.copy(BAK,SRC)
        try: build(); okb=True
        except Exception as e: okb=False; print('!! compilation of the baseline track.h failed: %s'%e,flush=True)
        if okb and open(SRC).read()==open(BAK).read(): os.remove(BAK)
        print('track.h restored to the baseline configuration%s.'%(' and compiled' if okb else ' (COMPILATION FAILED -- tell Claude)'),flush=True)
        for p,h in sh.items():
            b=p+'.p53bak'
            if os.path.exists(p) and sha(p)==h:
                if os.path.exists(b): os.remove(b)
                print('   check at the end: %s the same as at the start (SHA256)'%os.path.basename(p),flush=True)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p,p+'.p53_nepoznat')
                os.replace(b,p); print('   !! %s changed -- RESTORED from the backup copy; tell Claude'%os.path.basename(p),flush=True)
        if os.path.exists(LOCK): os.remove(LOCK)
    print('GPU part done (%.0f min).'%((time.time()-t0)/60),flush=True)
    return True
# ======================================================================= evaluation
def ocjena():
    import trackeval
    from scipy.stats import wilcoxon
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict, Counter
    os.makedirs(O,exist_ok=True)
    H={}; exec(open(W+'/id_smooth.py').read().split('CFG={')[0].split('"""')[2],H)
    load_tr,load_gt,pick,iou=H['load_tr'],H['load_gt'],H['pick'],H['iou']
    ec=trackeval.Evaluator.get_default_eval_config()
    ec.update(PRINT_RESULTS=False,PRINT_CONFIG=False,OUTPUT_SUMMARY=False,OUTPUT_DETAILED=False,PLOT_CURVES=False,TIME_PROGRESS=False,USE_PARALLEL=False,DISPLAY_LESS_PROGRESS=True)
    try: C=json.load(open(O+'/te_cache.json'))
    except Exception: C={}
    rows=lambda d: sorted((int(f),int(o['id']),*map(float,o['box'][:4])) for f,Lst in d.items() for o in Lst)
    def ev(key,files):
        sg=[[os.path.basename(p),os.path.getsize(p),int(os.path.getmtime(p))] for _,p in sorted(files.items())]
        if key in C and C[key].get('sig')==sg: return {int(s):v for s,v in C[key]['res'].items()}
        base=O+'/te_tmp'; shutil.rmtree(base,ignore_errors=True); si={}
        for s,p in files.items():
            tr=load_tr(p); g=load_gt(GT%s,pick(GT%s,tr))
            G=rows(g); R_=rows(tr); seq='S%02d'%s; si[seq]=max([r[0] for r in G+R_]+[1])
            os.makedirs('%s/gt/%s/gt'%(base,seq),exist_ok=True); os.makedirs(base+'/tr/t/data',exist_ok=True)
            with open('%s/gt/%s/gt/gt.txt'%(base,seq),'w') as f:
                for x in G: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n'%x)
            with open('%s/tr/t/data/%s.txt'%(base,seq),'w') as f:
                for x in R_: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n'%x)
        dc=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(GT_FOLDER=base+'/gt',TRACKERS_FOLDER=base+'/tr',BENCHMARK='b',SPLIT_TO_EVAL='',SEQ_INFO=si,TRACKERS_TO_EVAL=['t'],DO_PREPROC=False,
                  SKIP_SPLIT_FOL=True,GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt',TRACKER_SUB_FOLDER='data',PRINT_CONFIG=False)
        M=[trackeval.metrics.HOTA({'PRINT_CONFIG':False}),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),trackeval.metrics.Identity({'PRINT_CONFIG':False})]
        out,_=trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)],M); Rt=out['MotChallenge2DBox']['t']; shutil.rmtree(base,ignore_errors=True)
        res={}
        for q in si:
            h=Rt[q]['pedestrian']; m=lambda k: float(np.mean(h['HOTA'][k]))
            res[int(q[1:])]={'HOTA':m('HOTA'),'DetA':m('DetA'),'AssA':m('AssA'),'AssRe':m('AssRe'),'AssPr':m('AssPr'),
                             'IDF1':float(h['Identity']['IDF1']),'MOTA':float(h['CLEAR']['MOTA']),'IDSW':float(h['CLEAR']['IDSW'])}
        C[key]={'sig':sg,'res':{str(s):v for s,v in res.items()}}; json.dump(C,open(O+'/te_cache.json','w')); return res
    P('=== p53: decomposition and sensitivity of the final variant, YOLO11m, 16 SSMOT sequences (TrackEval); plan written before the run (claude/p53_rastav_11m_plan.md)')
    FILES={'base':{s:P52%('base',s) for s in SS16},'final':{s:P52%('final',s) for s in SS16}}
    for v in ('cls','prob','l02','l03','c07'): FILES[v]={s:outp(v,s) for s in SS16}
    NAME={'base':'baseline','final':'final variant','cls':'class penalty only','prob':'probation only',
          'l02':'final, penalty 0.2','l03':'final, penalty 0.3','c07':'final, threshold 0.7'}
    R={}
    for k,f in FILES.items():
        miss=[s for s,p in f.items() if not json_ok(p)]
        if miss: P('  (no complete logs for "%s": %d of 16 missing)'%(NAME[k],len(miss))); continue
        R[k]=ev(k,f); print('   TrackEval %-24s ok'%NAME[k],flush=True)
    rng=np.random.default_rng(0)
    def wp(d):
        d=np.array(d,float)
        if not np.any(d!=0): return 1.0
        try: return float(wilcoxon(d).pvalue)
        except ValueError: return float('nan')
    def holm(ps):
        o=sorted(range(len(ps)),key=lambda i:ps[i]); out=[0]*len(ps); run=0
        for r,i in enumerate(o): run=max(run,min(1,ps[i]*(len(ps)-r))); out[i]=run
        return out
    def ci(d):
        d=np.array(d,float); b=rng.choice(d,(10000,len(d))).mean(1); return np.percentile(b,[2.5,97.5])
    fp=lambda p: '< 0.001' if p<0.001 else '%.3f'%p
    MET=['HOTA','DetA','AssA','AssPr','IDF1','IDSW']
    mean=lambda k,m,seqs=SS16: float(np.mean([R[k][s][m] for s in seqs]))
    def head(): P('  %-28s '%''+'  '.join('%6s'%m for m in MET))
    def row(k,seqs=SS16,lab=None): P('  %-28s '%(lab or NAME[k])+'  '.join(('%6.1f' if m=='IDSW' else '%6.3f')%mean(k,m,seqs) for m in MET))
    def dl(a,b,m,seqs=SS16): return [R[b][s][m]-R[a][s][m] for s in seqs]
    def show(pairs,seqs=SS16,holm_over=None,title=''):
        """pairs: [(a, b, label)] -> b - a"""
        P('  %s'%title); P('  %-42s %s'%('',' | '.join('%-25s'%m for m in MET)))
        res={}
        ps={m:[wp(dl(a,b,m,seqs)) for a,b,_ in pairs] for m in MET}
        hs={m:(holm(ps[m]) if holm_over else [None]*len(pairs)) for m in MET}
        for i,(a,b,lab) in enumerate(pairs):
            cells=[]
            for m in MET:
                d=dl(a,b,m,seqs); e=np.mean(d)
                cells.append((('%+6.2f' if m=='IDSW' else '%+.4f')%e+' p=%s%s'%(fp(ps[m][i]),(' H=%s'%fp(hs[m][i])) if hs[m][i] is not None else '')).ljust(25))
                res.setdefault(lab,{})[m]={'d':float(e),'p':ps[m][i],'holm':hs[m][i],'ci':[float(x) for x in ci(d)],
                                            'bolje':int(sum(1 for x in d if (x<0 if m=='IDSW' else x>0))),'losije':int(sum(1 for x in d if (x>0 if m=='IDSW' else x<0)))}
            P('  %-42s %s'%(lab,' | '.join(cells)))
        return res
    OUT={}
    st=stanje()
    # ---- A) reproducibility
    P('\n=== A) REPRODUCIBILITY: baseline and final run again, comparison with log_p52_* (all 16 sequences)')
    OUT['ponovljivost']={}
    for v in ('base','final'):
        ok=[s for s in SS16 if json_ok(outp(v+'_t',s))]
        same=[s for s in ok if frames(outp(v+'_t',s))==frames(P52%(v,s))]
        diff=[s for s in ok if s not in same]
        P('  %-18s identical on %d of %d repeated sequences%s'%(NAME[v]+':',len(same),len(ok),('; DIFFERENT on %s'%diff) if diff else ''))
        OUT['ponovljivost'][v]={'isto':same,'razlicito':diff}
    # ---- B) decomposition
    if all(k in R for k in ('base','cls','prob','final')):
        P('\n=== B) DECOMPOSITION (16 SSMOT sequences, mean per sequence)')
        head()
        for k in ('base','cls','prob','final'): row(k)
        OUT['rastav']=show([('base','cls','class penalty only - baseline'),('base','prob','probation only - baseline')],holm_over=True,
                           title='single components versus the baseline (Wilcoxon, n = 16; H = Holm over 2 components):')
        OUT['rastav'].update(show([('base','final','final - baseline'),('cls','final','final - penalty only (adding the probation)'),
                                   ('prob','final','final - probation only (adding the penalty)')],title='final variant and the addition of the other component:'))
        P('  additivity (sum of the single effects versus the effect of the final variant):')
        add={}
        for m in ('HOTA','DetA','IDF1','IDSW'):
            a=np.mean(dl('base','cls',m)); b=np.mean(dl('base','prob',m)); c=np.mean(dl('base','final',m)); add[m]=[float(a),float(b),float(c)]
            P('     %-5s penalty %+.4f + probation %+.4f = %+.4f   versus final %+.4f'%(m,a,b,a+b,c) if m!='IDSW' else
              '     %-5s penalty %+.2f + probation %+.2f = %+.2f   versus final %+.2f'%(m,a,b,a+b,c))
        OUT['aditivnost']=add
        P('\n  the same WITHOUT SSMOT 10 (descriptive; 15 sequences):')
        head()
        for k in ('base','cls','prob','final'): row(k,SS15)
        OUT['rastav_bez10']=show([('base','cls','penalty only - baseline'),('base','prob','probation only - baseline'),('base','final','final - baseline')],SS15,
                                 title='differences (Wilcoxon, n = 15):')
        P('  SSMOT 10 separately (IDSW / HOTA): '+' | '.join('%s %d / %.3f'%(NAME[k],R[k][10]['IDSW'],R[k][10]['HOTA']) for k in ('base','cls','prob','final')))
    # ---- C) sensitivity
    sens=[k for k in ('final','l02','l03','c07') if k in R]
    if len(sens)>1:
        P('\n=== C) SENSITIVITY of the method parameters (main setting = penalty 0.5 and threshold 0.8; the others are only reported)')
        head()
        for k in sens: row(k)
        OUT['osjetljivost']=show([('final',k,'%s - main'%NAME[k]) for k in sens[1:]],title='differences versus the main setting (Wilcoxon, n = 16):')
        for m in ('HOTA','IDF1','IDSW'):
            vals=[mean(k,m) for k in sens]; P('  range %-5s over %d settings: %.4f (%.4f - %.4f)'%(m,len(sens),max(vals)-min(vals),min(vals),max(vals)))
        P('  without SSMOT 10: '+' | '.join('%s IDSW %.1f HOTA %.3f'%(NAME[k],mean(k,'IDSW',SS15),mean(k,'HOTA',SS15)) for k in sens))
    # ---- D) switch diagnosis
    EPS=np.finfo('float').eps
    def r2(b): return [float('%.2f'%float(x)) for x in b[:4]]
    def load_cls(p):
        d={}; key_=None
        for fd in json.load(open(p)):
            for x in fd.get('detections') or []:
                if key_ is None: key_=next((k for k in ('class','class_id','classId','cls','label') if k in x),'')
                if key_: d[(fd['frame'],x['id'])]=x[key_]
        return d
    def te_match(gt,tr):
        prev_ts={}; prev={}; last={}; held=defaultdict(set); evs=[]
        for f in sorted(set(gt)|set(tr)):
            G=gt.get(f,[]); T=tr.get(f,[])
            if not G or not T: continue
            gb=[r2(o['box']) for o in G]; tb=[r2(t['box']) for t in T]
            sim=np.array([[iou(a,b) for b in tb] for a in gb])
            cont=np.array([[1.0 if prev_ts.get(o['id'])==t['id'] else 0.0 for t in T] for o in G])
            sc=1000*cont+sim; sc[sim<0.5-EPS]=0
            rr,cc=linear_sum_assignment(-sc); new_ts={}
            for a,b in zip(rr,cc):
                if sc[a,b]<=0+EPS: continue
                k=G[a]['id']; nid=T[b]['id']
                if k in prev and prev[k]!=nid: evs.append(dict(gt=k,f=f,gap=f-last[k]-1,ret=nid in held[k],old=prev[k],fold=last[k],new=nid))
                prev[k]=nid; last[k]=f; held[k].add(nid); new_ts[k]=nid
            prev_ts=new_ts
        return evs
    names=[l.strip() for l in open(LAB11) if l.strip()] if os.path.exists(LAB11) else None
    P('\n=== D) SWITCH DIAGNOSIS (TrackEval matching; self-check = count equal to TrackEval IDSW)')
    OUT['dijagnoza']={}
    ngt=sum(len({int(l.split(',')[1]) for l in open(GT%s) if l.strip()}) for s in SS16)
    for k in ('base','cls','prob','final'):
        if k not in R: continue
        allev=[]; okc=True; clsch=0; known=0; pairs=Counter()
        for s in SS16:
            p=FILES[k][s]; tr=load_tr(p); g=load_gt(GT%s,pick(GT%s,tr)); Cl=load_cls(p); evs=te_match(g,tr)
            if len(evs)!=int(R[k][s]['IDSW']): okc=False
            for e in evs:
                e['seq']=s; co=Cl.get((e['fold'],e['old'])); cn=Cl.get((e['f'],e['new']))
                if co is not None and cn is not None:
                    known+=1
                    if co!=cn: clsch+=1; a,b=sorted((int(co),int(cn))); pairs[(a,b)]+=1
            allev+=evs
        n=len(allev); g_=np.array([e['gap'] for e in allev]) if n else np.zeros(0)
        byk=defaultdict(list)
        for e in allev: byk[(e['seq'],e['gt'])].append(e['f'])
        iv=[x for v in byk.values() for x in np.diff(sorted(v))]
        perseq=Counter(e['seq'] for e in allev); worst=perseq.most_common(1)[0] if perseq else (None,0)
        nm=lambda i: names[i] if names and 0<=i<len(names) else 'class %d'%i
        P('  %-22s switches %3d (self-check %s)'%(NAME[k],n,'PASSED' if okc else 'FAILED'))
        if n:
            P('     matched in the previous frame %3.0f%% | return to an earlier held ID %3.0f%% | median gap between successive switches of the same track %s frames'%(
              100*(g_==0).mean(),100*np.mean([e['ret'] for e in allev]),('%.0f'%np.median(iv)) if iv else '-'))
            P('     GT tracks affected %d of %d | most in sequence SSMOT %s: %d (%.0f%%) | with a class change %d of %d (%.0f%%): %s'%(
              len(byk),ngt,worst[0],worst[1],100*worst[1]/n,clsch,known,100*clsch/max(known,1),', '.join('%s<->%s %d'%(nm(a),nm(b),c) for (a,b),c in pairs.most_common(3)) or '-'))
        OUT['dijagnoza'][k]={'n':n,'samoprovjera':okc,'prethodni_frejm':float((g_==0).mean()) if n else None,'povratak':float(np.mean([e['ret'] for e in allev])) if n else None,
                             'medijan_razmaka':float(np.median(iv)) if iv else None,'gt_tragova':len(byk),'gt_ukupno':ngt,'najgora':[worst[0],worst[1]],
                             'promjena_klase':clsch,'poznato':known}
    # ---- E) speed
    P('\n=== E) SPEED (whole sequence, offline; "with start-up" includes starting the pipeline; slope = pure throughput)')
    tm=st.get('vrijeme',{}); OUT['brzina']={}
    def fpsrow(lab,pts):
        if len(pts)<3: return
        fr=np.array([p[0] for p in pts],float); sec=np.array([p[1] for p in pts],float); slope,icpt=np.polyfit(fr,sec,1)
        P('  %-34s %5.1f FPS with start-up | from the slope %5.1f FPS, start-up about %3.0f s  (%d sequences)'%(lab,fr.sum()/sec.sum(),1/slope if slope>0 else float('nan'),icpt,len(pts)))
        OUT['brzina'][lab]={'s_pokretanjem':float(fr.sum()/sec.sum()),'nagib':float(1/slope) if slope>0 else None,'pokretanje_s':float(icpt),'n':len(pts)}
    for v,_,name in VARS: fpsrow('ours, '+name,[tm['%s|%d'%(v,s)] for s in SS16 if '%s|%d'%(v,s) in tm])
    try:
        t44=json.load(open(S+'/p44/vrijeme.json'))
        for t,lab in (('nvsort','NvSORT'),('deep','NvDeepSORT'),('dcf1','NvDCF r1'),('dcf2','NvDCF r2')):
            fpsrow('%s (p44, probe_to_json)'%lab,[t44['%s|%d'%(t,s)] for s in SS16 if '%s|%d'%(t,s) in t44])
    except Exception: P('  (no scpilot/p44/vrijeme.json for the NVIDIA trackers)')
    OUT['srednje']={k:{m:mean(k,m) for m in MET+['AssRe','MOTA']} for k in R}
    OUT['po_snimkama']={k:{str(s):v for s,v in R[k].items()} for k in R}
    json.dump(OUT,open(O+'/p53_sve.json','w'),indent=1,default=float)
    open(S+'/p53_rastav_11m.txt','w').write('\n'.join(L)+'\n'); print('\nresult: %s/p53_rastav_11m.txt'%S)
if __name__=='__main__':
    if '--vrati' in sys.argv:
        os.makedirs(O,exist_ok=True)
        if alive(LOCK): sys.exit('!! p53 is still running (PID %s) -- do not restore while it runs.'%alive(LOCK))
        vrati(); print('done.')
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
