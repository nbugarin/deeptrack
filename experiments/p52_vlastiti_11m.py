#!/usr/bin/env python3
"""p52: OUR TRACKER WITH THE YOLO11m DETECTOR (for IEEE) -- does the improvement also hold with a modern detector?

Plan written BEFORE the run:
  main outcome (YOLO11m, 16 SSMOT sequences as in MDPI): final variant versus baseline; the improvement HOLDS if IDSW is
  significantly lower (Wilcoxon, p < 0.05) and the mean dHOTA >= -0.005. Mechanism: share of switches with a class change (YOLO11m versus
  YOLOv8x). Comparison with NvSORT, NvDeepSORT, NvDCF r1/r2, Ultralytics ByteTrack/BoT-SORT (logs from p44, Holm). Own 5 secondary.

Steps (GPU):
  1. check that track.h is at the baseline configuration (as in p43) -- otherwise it does NOT start; copy track.h.bak_p52
  2. engines (model_b1_gpu0_fp16, best_0904, osnet_vessel, YOLO11m) -> backup copy (.p52bak), checked after every run
  3. baseline: compiles the tracker; REPRODUCIBILITY CHECK (old detector, one own sequence, must be identical to log_p43_base);
     then YOLO11m (config_infer_11m.txt from p44) on 16 SSMOT + own 5  -> /workspace/log_p52_base_<ss|own>_<n>.json
  4. final variant (the same values as the p43 "final"): the same -> log_p52_final_<ss|own>_<n>.json
  5. at the end (and on an error): track.h is restored and compiled, the engines are checked (SHA256) and restored if needed
  6. evaluation with TrackEval + switch diagnosis (class change) + comparisons
Run (container with DeepStream, GPU free):  nohup python3 /workspace/p52_vlastiti_11m.py > /workspace/p52_out.txt 2>&1 &
Monitoring:                                tail -3 /workspace/p52_out.txt
Evaluation only (without a GPU):           python3 /workspace/p52_vlastiti_11m.py --eval
If interrupted (e.g. computer shut down):  python3 /workspace/p52_vlastiti_11m.py --vrati   (then rerun -- it resumes)
Output: /workspace/scpilot/p52_vlastiti_11m.txt (+ scpilot/p52/p52_sve.json)
"""
import os, re, sys, json, time, shutil, hashlib, subprocess
import numpy as np
for a,b in (('float',float),('int',int),('bool',bool)):
    if not hasattr(np,a): setattr(np,a,b)
W=os.environ.get('P52_W','/workspace'); S=W+'/scpilot'; O=S+'/p52'; LOCK=O+'/p52.lock'
SRC=W+'/mytracker/src/track.h'; BAK=SRC+'.bak_p52'; LIB=W+'/mytracker/libmoj.so'
PROBE=W+'/probe_moj.py'; CFG=W+'/config_tracker_NvSORT.yml'
INFER=W+'/DeepStream-Yolo/config_infer_11m.txt'
WD=W+'/det_glavni/B_yolo11m/weights'; ENG11=WD+'/best_11m_b1_gpu0_fp16.engine'; LAB11=WD+'/labels.txt'
PROT0=[W+'/model_b1_gpu0_fp16.engine',W+'/best_0904.engine',W+'/osnet_vessel.engine',ENG11]
SS16=[1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]; SS15=[s for s in SS16 if s!=10]; OWN=[1,2,3,4,5]
SEQS={'ss':SS16,'own':OWN}
VID={'ss':W+'/SSMOT_video/video/SSMOT_%d.mp4','own':W+'/%d.mp4'}
GT={'ss':W+'/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt','own':W+'/gt_%d.txt'}
BASE={'maxAge':100,'wMeas':0.15,'galleryMax':5,'wReid':0.8,'maxCost':0.75,'inertia':0.10,'detNms':0.70,'minHits':20}
EXTRA_BASE={'reidAfterN':0}
OFF={'classGate':lambda v: v==1,'confHigh':lambda v: v>1,'maxAgeSec':lambda v: v<=0,'minHitsSec':lambda v: v<=0}
FINAL={'classGate':0,'classPenalty':0.5,'confHigh':0.8,'maxAgeSec':3.3333,'minHitsSec':0.6667}   # exactly as the p43 "final"
VARN={'base':'baseline','final':'final variant'}
LOG=W+'/log_p52_%s_%s_%d.json'                                                                    # (variant, set, sequence)
def own_find(name): return [W+'/log_%s_%%d.json'%name,W+'/log_ss_%s_%%d.json'%name,W+'/log_%s_ss_%%d.json'%name]
V8={('ss','base'):own_find('ssmot'),('ss','final'):own_find('combo_p0p5_c0p8h2'),
    ('own','base'):[W+'/log_p43_base_%d.json'],('own','final'):[W+'/log_p43_konacna_%d.json',W+'/log_combo_p0p5_c0p8h2_own_%d.json']}
REF11=[('NvSORT',W+'/log_ss11m_nvsort_%d.json'),('NvDeepSORT',W+'/log_ss11m_deep_%d.json'),('NvDCF r1',W+'/log_ss11m_dcf1_%d.json'),
       ('NvDCF r2',W+'/log_ss11m_dcf2_%d.json'),('Ultralytics ByteTrack',S+'/p44/ultra_bytetrack_%d.json'),('Ultralytics BoT-SORT',S+'/p44/ultra_botsort_%d.json')]
LOCKS=[S+'/p43.lock',S+'/p27.lock']+[S+'/p%d/p%d.lock'%(i,i) for i in (44,45,46,47,48)]
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
def first(pats,s):
    for pt in pats:
        if json_ok(pt%s): return pt%s
    return None
def stanje():
    try: return json.load(open(O+'/stanje.json'))
    except Exception: return {}
def spremi(st): json.dump(st,open(O+'/stanje.json','w'),indent=1)
def fps(path):
    try:
        import cv2; f=cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)
        if f and f>0: return float(f)
    except Exception: pass
    return 25.0 if '/SSMOT_' in path else 30.0
def nframes(path):
    try:
        import cv2; n=cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT)
        if n and n>0: return int(n)
    except Exception: pass
    return 10**9
# ======================================================================= track.h (same as p43)
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
        b=p+'.p52bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p)==sha(b): os.remove(b); print('   %s: untouched, copy deleted'%os.path.basename(p),flush=True)
        else:
            if os.path.exists(p): os.replace(p,p+'.p52_nepoznat')
            os.replace(b,p); print('   %s: RESTORED from the backup copy'%os.path.basename(p),flush=True)
# ======================================================================= GPU
def runs():
    try:
        import scipy, trackeval, cv2   # noqa: F401
    except Exception as e: sys.exit('!! missing package (%s). The container was probably restarted -- first run:\n   %s'%(e,REINSTALL))
    os.makedirs(O,exist_ok=True)
    if alive(LOCK): sys.exit('!! p52 is already running (PID %s) -- do not run twice.'%alive(LOCK))
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running (PID %s) -- run p52 when it finishes.'%(os.path.basename(lk).split('.')[0],alive(lk)))
    if os.environ.get('P52_NOGPU')!='1':
        try:
            used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
            print('GPU memory in use: %d MiB'%used,flush=True)
            if used>1500: sys.exit('GPU is busy (>1500 MiB) -- something else is running on the GPU. Wait until it finishes.')
        except (IndexError,ValueError,FileNotFoundError): print('warning: nvidia-smi gave no data, continuing',flush=True)
    if not os.path.isdir('/usr/include/eigen3') and os.environ.get('P52_NOGPU')!='1': sys.exit('!! no Eigen. First: apt-get install -y libeigen3-dev')
    if os.path.exists(BAK) or any(os.path.exists(p+'.p52bak') for p in PROT0):
        print('the previous run was interrupted -- restoring track.h and checking the engines:',flush=True); vrati()
    need=[PROBE,CFG,INFER,ENG11,SRC,W+'/id_smooth.py']+[VID[ds]%s for ds in SEQS for s in SEQS[ds]]+[GT[ds]%s for ds in SEQS for s in SEQS[ds]]
    miss=[p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! missing:\n   '+'\n   '.join(miss))
    if shutil.disk_usage(W).free/2**30<3: sys.exit('!! less than 3 GB free on the disk')
    orig=open(SRC).read(); bad=check_base(orig)
    if bad: sys.exit('!! track.h is NOT at the baseline configuration -- nothing was run:\n   '+'\n   '.join(bad)+'\n   Report this output.')
    mh=getv(orig,'minHitsHigh')[1]
    m=re.search(r'^\s*model-engine-file\s*=\s*(\S+)',open(INFER).read(),re.M)
    print('track.h at the baseline configuration (checked); minHitsHigh in track.h = %s (not changed, as in p43)'%('%g'%mh if mh is not None else 'none'),flush=True)
    print('config_infer_11m.txt: model-engine-file = %s'%(m.group(1) if m else 'NONE'),flush=True)
    st=stanje(); h_inf=sha(INFER)
    if 'infer_sha' in st and st['infer_sha']!=h_inf: sys.exit('!! config_infer_11m.txt changed since the first p52 run -- report this')
    st['infer_sha']=h_inf; spremi(st)
    rep=min(OWN,key=lambda s: nframes(VID['own']%s))
    todo={v:[(ds,s) for ds in ('ss','own') for s in SEQS[ds] if not json_ok(LOG%(v,ds,s))] for v in ('base','final')}
    n_all=sum(len(t) for t in todo.values())+sum(1 for v in ('base','final') if v+'_repro' not in st)
    print('runs remaining: %d (16 SSMOT + 5 own per variant, + reproducibility check on own sequence %d)'%(n_all,rep),flush=True)
    PROT=[p for p in PROT0 if os.path.exists(p)]
    print('protected engines: %s'%', '.join(os.path.basename(p) for p in PROT),flush=True)
    shutil.copy2(SRC,BAK); open(LOCK,'w').write(str(os.getpid())); sh={}
    t0=time.time(); done=[0]
    try:
        for p in PROT:
            sh[p]=sha(p); shutil.copy2(p,p+'.p52bak')
            if sha(p+'.p52bak')!=sh[p]: sys.exit('!! backup copy %s is not valid -- STOP'%p)
        def one(video,out,infer):
            st_={p:stat(p) for p in PROT}; env=dict(os.environ,MYTRACKER_FPS='%g'%fps(video)); rc=None
            try:
                r=subprocess.run(['python3',PROBE,video,CFG,out]+([infer] if infer else []),cwd=W,capture_output=True,text=True,env=env,timeout=T_RUN)
                rc=r.returncode; err=(r.stdout or '')[-2500:]+'\n----\n'+(r.stderr or '')[-3500:]
            except subprocess.TimeoutExpired: err='time limit exceeded (%d s)'%T_RUN
            ch=[os.path.basename(p) for p in PROT if stat(p)!=st_[p]]
            if ch:
                if os.path.exists(out): os.remove(out)
                open(O+'/greska.txt','w').write(err)
                sys.exit('!! %s changed during the run (%s) -- restoring from the copy; see: tail -40 /workspace/scpilot/p52/greska.txt'%(', '.join(ch),os.path.basename(video)))
            if json_ok(out): return True
            if os.path.exists(out): os.remove(out)
            open(O+'/greska.txt','w').write('%s, return code %s\n'%(video,rc)+err); return False
        def tick(lab):
            done[0]+=1; el=time.time()-t0
            print('  [%2d/%d] %-34s ok   (about %.0f min left)'%(done[0],n_all,lab,el/done[0]*(n_all-done[0])/60),flush=True)
        for v in ('base','final'):
            if not todo[v] and v+'_repro' in st: continue
            txt=orig
            if v=='final':
                for k,x in FINAL.items(): txt=setv(txt,k,x)
            open(SRC,'w').write(txt); build()
            print('%s: tracker compiled (%s)'%(VARN[v],'baseline track.h' if v=='base' else ', '.join('%s %g'%(k,x) for k,x in FINAL.items())),flush=True)
            if v+'_repro' not in st:
                out=O+'/repro_%s_%d.json'%(v,rep); ref=first(V8[('own',v)],rep)
                if os.path.exists(out): os.remove(out)
                ok=one(VID['own']%rep,out,None) or one(VID['own']%rep,out,None)
                if not ok: res={'video':rep,'ishod':'the run with the old detector failed (details in greska.txt)'}
                elif ref is None: res={'video':rep,'ishod':'no reference log'}
                else:
                    A=frames(out); B=frames(ref); nd=sum(1 for f in set(A)|set(B) if A.get(f)!=B.get(f))
                    res={'video':rep,'ref':os.path.basename(ref),'isto':nd==0,'razl_frejmova':nd,'frejmova':len(set(A)|set(B)),
                         'ishod':'IDENTICAL %s'%os.path.basename(ref) if nd==0 else '!! NOT identical %s (%d of %d frames different)'%(os.path.basename(ref),nd,len(set(A)|set(B)))}
                st[v+'_repro']=res; spremi(st); tick('reproducibility check (%s)'%VARN[v]); print('     -> %s'%res['ishod'],flush=True)
            for ds,s in todo[v]:
                ok=one(VID[ds]%s,LOG%(v,ds,s),INFER) or one(VID[ds]%s,LOG%(v,ds,s),INFER)
                if not ok: sys.exit('!! %s, %s %d failed in the second attempt too -- see: tail -40 /workspace/scpilot/p52/greska.txt'%(VARN[v],'SSMOT' if ds=='ss' else 'own',s))
                tick('%s, %s %d'%(VARN[v],'SSMOT' if ds=='ss' else 'own',s))
    finally:
        shutil.copy(BAK,SRC)
        try: build(); okb=True
        except Exception as e: okb=False; print('!! compilation of the baseline track.h failed: %s'%e,flush=True)
        if okb and open(SRC).read()==open(BAK).read(): os.remove(BAK)
        print('track.h restored to the baseline configuration%s.'%(' and compiled' if okb else ' (COMPILATION FAILED -- report this)'),flush=True)
        for p,h in sh.items():
            b=p+'.p52bak'
            if os.path.exists(p) and sha(p)==h:
                if os.path.exists(b): os.remove(b)
                print('   check at the end: %s the same as at the start (SHA256)'%os.path.basename(p),flush=True)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p,p+'.p52_nepoznat')
                os.replace(b,p); print('   !! %s changed -- RESTORED from the backup copy; report this'%os.path.basename(p),flush=True)
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
    def ev(key,files,ds):
        sg=[[os.path.basename(p),os.path.getsize(p),int(os.path.getmtime(p))] for _,p in sorted(files.items())]
        if key in C and C[key].get('sig')==sg: return {int(s):v for s,v in C[key]['res'].items()}
        base=O+'/te_tmp'; shutil.rmtree(base,ignore_errors=True); si={}
        for s,p in files.items():
            tr=load_tr(p); g=load_gt(GT[ds]%s,pick(GT[ds]%s,tr))
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
    # ---- sets of logs
    SETS={}
    def add(key,ds,seqs,pats):
        f={s:first(pats,s) for s in seqs}
        if all(f.values()): SETS[key]=(ds,f)
        else: P('  (no complete logs for "%s": %d of %d)'%(key,sum(1 for x in f.values() if x),len(seqs)))
    P('=== p52: our tracker with YOLO11m -- evaluation (TrackEval); plan written before the run')
    for v in ('base','final'):
        add('11m|ss|'+v,'ss',SS16,[W+'/log_p52_%s_ss_%%d.json'%v]); add('11m|own|'+v,'own',OWN,[W+'/log_p52_%s_own_%%d.json'%v])
        add('v8|ss|'+v,'ss',SS15,V8[('ss',v)]); add('v8|own|'+v,'own',OWN,V8[('own',v)])
    for n,pt in REF11: add('11m|ss|ref|'+n,'ss',SS16,[pt])
    R={}
    for k,(ds,f) in SETS.items(): R[k]=ev(k,f,ds); print('   TrackEval %-34s ok'%k,flush=True)
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
    mean=lambda k,m,seqs: float(np.mean([R[k][s][m] for s in seqs]))
    MET=['HOTA','DetA','AssA','AssPr','IDF1','IDSW']
    def row(lab,k,seqs): P('  %-30s '%lab+'  '.join(('%6.1f' if m=='IDSW' else '%6.3f')%mean(k,m,seqs) for m in MET))
    def head(): P('  %-30s '%''+'  '.join('%6s'%m for m in MET))
    def delta(a,b,seqs,lab):
        """b - a, paired differences over sequences"""
        out={}
        for m in MET:
            d=[R[b][s][m]-R[a][s][m] for s in seqs]; lo,hi=ci(d); p=wp(d)
            better=sum(1 for x in d if (x<0 if m=='IDSW' else x>0)); worse=sum(1 for x in d if (x>0 if m=='IDSW' else x<0))
            out[m]={'d':float(np.mean(d)),'lo':float(lo),'hi':float(hi),'p':p,'bolje':better,'losije':worse,'n':len(d)}
        P('  %s:'%lab)
        for m in MET:
            e=out[m]; f_='%+.2f [%+.2f, %+.2f]' if m=='IDSW' else '%+.4f [%+.4f, %+.4f]'
            P('     %-5s '%m+f_%(e['d'],e['lo'],e['hi'])+'  p = %-7s better on %d, worse on %d of %d'%(fp(e['p']),e['bolje'],e['losije'],e['n']))
        return out
    OUT={'repro':{k:v for k,v in stanje().items() if k.endswith('_repro')}}
    # ---- A) reproducibility
    P('\n=== A) REPRODUCIBILITY CHECK (old detector, the same own sequence; must be identical to the logs from p43/IEEE)')
    for v in ('base','final'):
        r=OUT['repro'].get(v+'_repro'); P('  %-18s %s'%(VARN[v]+':',('own %d: %s'%(r['video'],r['ishod'])) if r else 'not run'))
    # ---- B) main outcome
    kb,kf='11m|ss|base','11m|ss|final'
    if kb in R and kf in R:
        P('\n=== B) MAIN OUTCOME: YOLO11m, 16 SSMOT sequences, final variant versus baseline (mean per sequence)')
        head(); row('baseline',kb,SS16); row('final variant',kf,SS16)
        D=delta(kb,kf,SS16,'final - baseline (paired differences, Wilcoxon, n = 16; 95% bootstrap CI)')
        vr=D['IDSW']['p']<0.05 and D['IDSW']['d']<0 and D['HOTA']['d']>=-0.005
        P('  RULE (in advance): IDSW significantly lower AND dHOTA >= -0.005  ->  %s'%('IMPROVEMENT HOLDS with YOLO11m too' if vr else
          'improvement does NOT hold with YOLO11m (IDSW %+.2f, p = %s; dHOTA %+.4f)'%(D['IDSW']['d'],fp(D['IDSW']['p']),D['HOTA']['d'])))
        OUT['glavni']={'delta':D,'vrijedi':bool(vr)}
    # ---- C) same tracker, old detector
    if 'v8|ss|base' in R and 'v8|ss|final' in R:
        P('\n=== C) THE SAME TRACKER WITH THE OLD YOLOv8x DETECTOR (existing logs, 15 SSMOT sequences without 10) -- for comparison')
        head(); row('YOLOv8x baseline (carried over)','v8|ss|base',SS15); row('YOLOv8x final','v8|ss|final',SS15)
        if kb in R and kf in R: row('YOLO11m baseline (same 15)',kb,SS15); row('YOLO11m final (same 15)',kf,SS15)
        D8=delta('v8|ss|base','v8|ss|final',SS15,'YOLOv8x: final - baseline (n = 15)'); OUT['v8_ss']=D8
        if kb in R and kf in R:
            D11=delta(kb,kf,SS15,'YOLO11m: final - baseline, same 15 sequences')
            P('  difference of the gains, (final - baseline) with YOLO11m minus the same with YOLOv8x (exploratory, n = 15;')
            P('  for IDSW positive = SMALLER gain with YOLO11m; for HOTA and IDF1 positive = LARGER gain with YOLO11m):')
            for m in ('IDSW','HOTA','IDF1'):
                dd=[(R[kf][s][m]-R[kb][s][m])-(R['v8|ss|final'][s][m]-R['v8|ss|base'][s][m]) for s in SS15]
                P('     %-5s %+.3f  p = %s'%(m,np.mean(dd),fp(wp(dd))))
            OUT['v8_vs_11m_ss15']={'11m':D11}
    # ---- D) mechanism: class change at switches
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
    def names_of(paths):
        for lp in paths:
            if os.path.exists(lp): return [l.strip() for l in open(lp) if l.strip()]
        return None
    N11=names_of([LAB11]); N8=names_of([W+'/labels.txt',W+'/DeepStream-Yolo/labels.txt',W+'/labels_best_0904.txt'])
    P('\n=== D) MECHANISM: switches with a class change (TrackEval matching; self-check = count equal to TrackEval IDSW)')
    P('  class names: YOLO11m %s | YOLOv8x %s'%(N11 or 'unknown',N8 or 'unknown'))
    OUT['klasa']={}
    for k,seqs,lab,names in ((kb,SS16,'YOLO11m baseline, SSMOT 16',N11),(kf,SS16,'YOLO11m final, SSMOT 16',N11),(kb,SS15,'YOLO11m baseline, same 15',N11),
                             ('v8|ss|base',SS15,'YOLOv8x baseline, SSMOT 15',N8),('v8|ss|final',SS15,'YOLOv8x final, SSMOT 15',N8),
                             ('11m|own|base',OWN,'YOLO11m baseline, own 5',N11),('11m|own|final',OWN,'YOLO11m final, own 5',N11),
                             ('v8|own|base',OWN,'YOLOv8x baseline, own 5',N8),('v8|own|final',OWN,'YOLOv8x final, own 5',N8)):
        if k not in SETS: continue
        ds,files=SETS[k]; n=0; clsch=0; known=0; pairs=Counter(); okc=True
        for s in seqs:
            p=files[s]; tr=load_tr(p); g=load_gt(GT[ds]%s,pick(GT[ds]%s,tr)); Cl=load_cls(p); evs=te_match(g,tr)
            if len(evs)!=int(R[k][s]['IDSW']): okc=False
            n+=len(evs)
            for e in evs:
                co=Cl.get((e['fold'],e['old'])); cn=Cl.get((e['f'],e['new']))
                if co is not None and cn is not None:
                    known+=1
                    if co!=cn:
                        clsch+=1; a,b=sorted((int(co),int(cn))); pairs[(a,b)]+=1
        nm=lambda i: names[i] if names and 0<=i<len(names) else 'class %d'%i
        P('  %-30s switches %4d (self-check %s) | with a class change %4d of %4d with a known class (%3.0f%%) | %s'%(lab,n,'PASSED' if okc else 'FAILED',
          clsch,known,100*clsch/max(known,1),', '.join('%s<->%s %d'%(nm(a),nm(b),c) for (a,b),c in pairs.most_common(3)) or '-'))
        OUT['klasa'][lab]={'n':n,'promjena':clsch,'poznato':known,'samoprovjera':okc,'parovi':{'%d-%d'%ab:c for ab,c in pairs.items()}}
    # ---- E) comparison with other trackers (YOLO11m)
    refs=[n for n,_ in REF11 if '11m|ss|ref|'+n in R]
    if kf in R and refs:
        P('\n=== E) YOLO11m, 16 SSMOT sequences: our tracker versus other trackers (same detector, logs from p44 / MDPI)')
        head()
        for n in refs: row(n,'11m|ss|ref|'+n,SS16)
        if kb in R: row('ours, baseline',kb,SS16)
        row('ours, final variant',kf,SS16)
        P('  final variant minus the other tracker (Wilcoxon, Holm over %d comparisons per metric):'%len(refs))
        cmp={}
        for m in ('IDSW','HOTA','IDF1','AssA'):
            ds_=[[R[kf][s][m]-R['11m|ss|ref|'+n][s][m] for s in SS16] for n in refs]; ps=[wp(d) for d in ds_]; hs=holm(ps)
            for n,d,p,h in zip(refs,ds_,ps,hs): cmp.setdefault(n,{})[m]=(float(np.mean(d)),p,h,sum(1 for x in d if (x<0 if m=='IDSW' else x>0)))
        P('  %-24s %22s %26s %26s %26s'%('','IDSW (p, Holm, better)','HOTA','IDF1','AssA'))
        for n in refs:
            P('  %-24s '%n+' '.join(('%+6.2f' if m=='IDSW' else '%+7.4f')%cmp[n][m][0]+' (%s, %s, %d)'%(fp(cmp[n][m][1]),fp(cmp[n][m][2]),cmp[n][m][3]) for m in ('IDSW','HOTA','IDF1','AssA')))
        OUT['usporedba']=cmp
    # ---- F) own 5 (secondary)
    if '11m|own|base' in R and '11m|own|final' in R:
        P('\n=== F) OWN 5 SEQUENCES (secondary, n = 5: smallest possible p = 0.0625; labels incomplete)')
        head(); row('YOLO11m baseline','11m|own|base',OWN); row('YOLO11m final','11m|own|final',OWN)
        if 'v8|own|base' in R: row('YOLOv8x baseline (IEEE)','v8|own|base',OWN)
        if 'v8|own|final' in R: row('YOLOv8x final (IEEE)','v8|own|final',OWN)
        OUT['own']=delta('11m|own|base','11m|own|final',OWN,'YOLO11m: final - baseline (n = 5)')
        for v in ('base','final'):
            if 'v8|own|'+v in R: OUT['own_det_'+v]=delta('v8|own|'+v,'11m|own|'+v,OWN,'%s: YOLO11m - YOLOv8x (n = 5)'%VARN[v])
    OUT['srednje']={k:{m:mean(k,m,sorted(R[k])) for m in MET+['AssRe','MOTA']} for k in R}
    OUT['po_snimkama']={k:{str(s):v for s,v in R[k].items()} for k in R}
    json.dump(OUT,open(O+'/p52_sve.json','w'),indent=1,default=float)
    open(S+'/p52_vlastiti_11m.txt','w').write('\n'.join(L)+'\n'); print('\nresult: %s/p52_vlastiti_11m.txt'%S)
if __name__=='__main__':
    if '--vrati' in sys.argv:
        os.makedirs(O,exist_ok=True)
        if alive(LOCK): sys.exit('!! p52 is still running (PID %s) -- do not restore while it runs.'%alive(LOCK))
        vrati(); print('done.')
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
