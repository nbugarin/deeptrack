#!/usr/bin/env python3
"""p55: ALL DeepStream TRACKERS WITH THE STREAMMUX AT THE NATIVE RESOLUTION OF THE SEQUENCES (1280x720) -- YOLO11m, 16 SSMOT sequences (for IEEE)

Plan written BEFORE the run. Tracker values frozen; nothing is selected.
p54 showed: the sequences are 1280x720, while the probe scripts set the streammux to 1920x1080 -> the resampling degrades the detections (DetA -0.032).
Steps (GPU):
  1. copies of /workspace/probe_moj.py and /workspace/probe_to_json.py -> /workspace/probe_moj_p55.py and probe_to_json_p55.py, ONLY the streammux width/height changed
     (check: exactly one replacement of each, the rest of the file identical); the original files are NOT changed
  2. 4 NVIDIA configurations (the same yml as p44) -> log_p55_nv_<label>_ss_<n>.json
  3. our tracker, 7 variants as in p53 (baseline, final, penalty only, probation only, penalty 0.2, 0.3, threshold 0.7) -> log_p55_<v>_ss_<n>.json
     track.h -> track.h.bak_p55 (always restored and compiled); engines -> .p55bak (checked after every run)
  4. evaluation: main comparison with SOTA (Holm over 6), effect of the native resolution, decomposition, sensitivity, diagnosis, speed
Run (GPU free):  nohup python3 /workspace/p55_izvorna_rezolucija.py > /workspace/p55_out.txt 2>&1 &
Monitoring: tail -3 /workspace/p55_out.txt     Evaluation only: python3 /workspace/p55_izvorna_rezolucija.py --eval
If interrupted: python3 /workspace/p55_izvorna_rezolucija.py --vrati   (then rerun -- it resumes)
After the double run (13.10.): python3 /workspace/p55_izvorna_rezolucija.py --popravi   (restores track.h, moves the logs aside)
Output: /workspace/scpilot/p55_izvorna_rezolucija.txt (+ scpilot/p55/p55_sve.json)
"""
import os, re, sys, json, time, shutil, hashlib, subprocess
import numpy as np
for a,b in (('float',float),('int',int),('bool',bool)):
    if not hasattr(np,a): setattr(np,a,b)
W=os.environ.get('P55_W','/workspace'); S=W+'/scpilot'; O=S+'/p55'; LOCK=O+'/p55.lock'
SRC=W+'/mytracker/src/track.h'; BAK=SRC+'.bak_p55'; LIB=W+'/mytracker/libmoj.so'
PM_SRC=W+'/probe_moj.py'; PT_SRC=W+'/probe_to_json.py'; PM=W+'/probe_moj_p55.py'; PTJ=W+'/probe_to_json_p55.py'      # copies in the same folder as the originals (in case they use relative paths)
CFG_OWN=W+'/config_tracker_NvSORT.yml'
INFER=W+'/DeepStream-Yolo/config_infer_11m.txt'
WD=W+'/det_glavni/B_yolo11m/weights'; ENG11=WD+'/best_11m_b1_gpu0_fp16.engine'; LAB11=WD+'/labels.txt'
PROT0=[W+'/model_b1_gpu0_fp16.engine',W+'/best_0904.engine',W+'/osnet_vessel.engine',ENG11]
SS16=[1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]; SS15=[s for s in SS16 if s!=10]
VID=W+'/SSMOT_video/video/SSMOT_%d.mp4'; GT=W+'/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
BASE={'maxAge':100,'wMeas':0.15,'galleryMax':5,'wReid':0.8,'maxCost':0.75,'inertia':0.10,'detNms':0.70,'minHits':20}
EXTRA_BASE={'reidAfterN':0}
OFF={'classGate':lambda v: v==1,'confHigh':lambda v: v>1,'maxAgeSec':lambda v: v<=0,'minHitsSec':lambda v: v<=0}
FINAL={'classGate':0,'classPenalty':0.5,'confHigh':0.8,'maxAgeSec':3.3333,'minHitsSec':0.6667}
CLS={'classGate':0,'classPenalty':0.5}; PROB={'confHigh':0.8,'maxAgeSec':3.3333,'minHitsSec':0.6667}
assert dict(CLS,**PROB)==FINAL and not set(CLS)&set(PROB)
OWNV=[('base',{},'baseline'),('cls',CLS,'class penalty only'),('prob',PROB,'probation only'),('final',FINAL,'final variant'),
      ('l02',dict(FINAL,classPenalty=0.2),'final, penalty 0.2'),('l03',dict(FINAL,classPenalty=0.3),'final, penalty 0.3'),
      ('c07',dict(FINAL,confHigh=0.7),'final, threshold 0.7')]
NV=[('nvsort',W+'/config_tracker_NvSORT.yml','NvSORT'),('deep',W+'/config_tracker_maritime.yml','NvDeepSORT'),
    ('dcf1',W+'/cfg_dcf_novis.yml','NvDCF r1'),('dcf2',W+'/cfg_nvdcf.yml','NvDCF r2')]
def outp(k,s): return W+'/log_p55_%s_ss_%d.json'%(k,s)                        # k = variant of ours or 'nv_<label>'
OLD={'base':W+'/log_p52_base_ss_%d.json','final':W+'/log_p52_final_ss_%d.json','nv_nvsort':W+'/log_ss11m_nvsort_%d.json',
     'nv_deep':W+'/log_ss11m_deep_%d.json','nv_dcf1':W+'/log_ss11m_dcf1_%d.json','nv_dcf2':W+'/log_ss11m_dcf2_%d.json'}
ULT=[('Ultralytics ByteTrack',S+'/p44/ultra_bytetrack_%d.json'),('Ultralytics BoT-SORT',S+'/p44/ultra_botsort_%d.json')]
LOCKS=[S+'/p43.lock',S+'/p27.lock']+[S+'/p%d/p%d.lock'%(i,i) for i in (44,45,46,47,48,52,53,54)]
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
import atexit
def uzmi_lock(ime):
    """atomic lock at the very start -- a second run of the same script gives up immediately"""
    os.makedirs(os.path.dirname(LOCK),exist_ok=True)
    for _ in range(3):
        try:
            fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
            atexit.register(pusti_lock); return
        except FileExistsError:
            try: pid=open(LOCK).read().strip(); age=time.time()-os.path.getmtime(LOCK)
            except FileNotFoundError: continue
            if (pid and os.path.exists('/proc/'+pid)) or (not pid and age<10):
                sys.exit('!! %s is already running (PID %s) -- do NOT run twice (check with the command: jobs).'%(ime,pid or '?'))
            try: os.remove(LOCK)
            except FileNotFoundError: pass
    sys.exit('!! cannot take the lock %s'%LOCK)
def pusti_lock():
    try:
        if open(LOCK).read().strip()==str(os.getpid()): os.remove(LOCK)
    except Exception: pass
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
def vres(p):
    import cv2; c=cv2.VideoCapture(p); return int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
MUXRX={'width':re.compile(r"""(set_property\(\s*['"]width['"]\s*,\s*)1920(\s*\))"""),'height':re.compile(r"""(set_property\(\s*['"]height['"]\s*,\s*)1080(\s*\))""")}
def native_copy(src,dst,w,h):
    t=open(src).read(); t2=t
    for k,v in (('width',w),('height',h)):
        t2,n=MUXRX[k].subn(lambda m,v=v: m.group(1)+str(v)+m.group(2),t2)
        if n!=1: sys.exit('!! in %s, not exactly one streammux setting "%s" = %s was found (found %d) -- report this'%(os.path.basename(src),k,'1920' if k=='width' else '1080',n))
    a=t.splitlines(); b=t2.splitlines(); diff=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    if len(a)!=len(b) or len(diff)!=2: sys.exit('!! copy %s differs in more than 2 lines -- report this'%os.path.basename(src))
    open(dst,'w').write(t2); return [b[i].strip() for i in diff]
# ======================================================================= track.h (same as p43/p52/p53)
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
        b=p+'.p55bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p)==sha(b): os.remove(b); print('   %s: untouched, copy deleted'%os.path.basename(p),flush=True)
        else:
            if os.path.exists(p): os.replace(p,p+'.p55_nepoznat')
            os.replace(b,p); print('   %s: RESTORED from the backup copy'%os.path.basename(p),flush=True)
# ======================================================================= GPU
def runs():
    try:
        import scipy, trackeval, cv2   # noqa: F401
    except Exception as e: sys.exit('!! missing package (%s). First run:\n   %s'%(e,REINSTALL))
    os.makedirs(O,exist_ok=True)
    uzmi_lock('p55')
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running (PID %s) -- run p55 when it finishes.'%(os.path.basename(lk).split('.')[0],alive(lk)))
    if os.environ.get('P55_NOGPU')!='1':
        try:
            used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
            print('GPU memory in use: %d MiB'%used,flush=True)
            if used>1500: sys.exit('GPU is busy (>1500 MiB) -- something else is running on the GPU. Wait until it finishes.')
        except (IndexError,ValueError,FileNotFoundError): print('warning: nvidia-smi gave no data, continuing',flush=True)
        if not os.path.isdir('/usr/include/eigen3'): sys.exit('!! no Eigen. First: apt-get install -y libeigen3-dev')
    if os.path.exists(BAK) or any(os.path.exists(p+'.p55bak') for p in PROT0):
        print('the previous run was interrupted -- restoring track.h and checking the engines:',flush=True); vrati()
    need=[PM_SRC,PT_SRC,CFG_OWN,INFER,ENG11,SRC,W+'/id_smooth.py']+[c for _,c,_ in NV]+[VID%s for s in SS16]+[GT%s for s in SS16]
    miss=[p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! missing:\n   '+'\n   '.join(miss))
    if shutil.disk_usage(W).free/2**30<3: sys.exit('!! less than 3 GB free on the disk')
    res={vres(VID%s) for s in SS16}
    if len(res)!=1: sys.exit('!! the sequences do not have the same resolution (%s) -- report this'%sorted(res))
    w,h=res.pop(); print('native resolution of the sequences: %dx%d'%(w,h),flush=True)
    orig=open(SRC).read(); bad=check_base(orig)
    if bad: sys.exit('!! track.h is NOT at the baseline configuration -- nothing was run:\n   '+'\n   '.join(bad)+'\n   Report this output.')
    for src,dst in ((PM_SRC,PM),(PT_SRC,PTJ)):
        ch=native_copy(src,dst,w,h); print('copy %s: only changed -> %s'%(os.path.basename(src),' | '.join(ch)),flush=True)
    st=stanje(); hh={'infer':sha(INFER)}; hh.update({k:sha(c) for k,c,_ in NV})
    if 'sha' in st and st['sha']!=hh: sys.exit('!! config_infer_11m.txt or the yml of the NVIDIA trackers changed since the first p55 run -- report this')
    st['sha']=hh; st['rezolucija']=[w,h]; st.setdefault('vrijeme',{}); spremi(st)
    todo_nv={k:[s for s in SS16 if not json_ok(outp('nv_'+k,s))] for k,_,_ in NV}
    todo_own={v:[s for s in SS16 if not json_ok(outp(v,s))] for v,_,_ in OWNV}
    n_all=sum(len(t) for t in todo_nv.values())+sum(len(t) for t in todo_own.values())
    print('runs remaining: %d (4 NVIDIA + 7 variants of our tracker, x 16 SSMOT sequences)'%n_all,flush=True)
    PROT=[p for p in PROT0 if os.path.exists(p)]
    shutil.copy2(SRC,BAK); sh={}; t0=time.time(); done=[0]
    try:
        for p in PROT:
            sh[p]=sha(p); shutil.copy2(p,p+'.p55bak')
            if sha(p+'.p55bak')!=sh[p]: sys.exit('!! backup copy %s is not valid -- STOP'%p)
        def one(key,s,probe,cfg,lab):
            out=outp(key,s)
            for att in (1,2):
                st_={p:stat(p) for p in PROT}; env=dict(os.environ,MYTRACKER_FPS='%g'%fps(VID%s)); rc=None
                if os.path.exists(out): os.remove(out)
                ta=time.time()
                try:
                    r=subprocess.run(['python3',probe,VID%s,cfg,out,INFER],cwd=W,capture_output=True,text=True,env=env,timeout=T_RUN)
                    rc=r.returncode; err=(r.stdout or '')[-2500:]+'\n----\n'+(r.stderr or '')[-3500:]
                except subprocess.TimeoutExpired: err='time limit exceeded (%d s)'%T_RUN
                dt=time.time()-ta
                ch=[os.path.basename(p) for p in PROT if stat(p)!=st_[p]]
                if ch:
                    if os.path.exists(out): os.remove(out)
                    open(O+'/greska.txt','w').write(err)
                    sys.exit('!! %s changed during the run (%s, SSMOT_%d) -- restoring from the copy; see: tail -40 /workspace/scpilot/p55/greska.txt'%(', '.join(ch),lab,s))
                if json_ok(out):
                    st['vrijeme']['%s|%d'%(key,s)]=[len(json.load(open(out))),dt]; spremi(st)
                    done[0]+=1; el=time.time()-t0
                    print('  [%3d/%d] %-28s SSMOT %-2d ok   (about %.0f min left)'%(done[0],n_all,lab,s,el/done[0]*(n_all-done[0])/60),flush=True); return
                if os.path.exists(out): os.remove(out)
                open(O+'/greska.txt','w').write('%s SSMOT_%d, return code %s\n'%(lab,s,rc)+err)
            sys.exit('!! %s, SSMOT %d failed in the second attempt too -- see: tail -40 /workspace/scpilot/p55/greska.txt'%(lab,s))
        for k,cfg,lab in NV:
            for s in todo_nv[k]: one('nv_'+k,s,PTJ,cfg,lab)
        for v,cfg,lab in OWNV:
            if not todo_own[v]: continue
            txt=orig
            for kk,x in cfg.items(): txt=setv(txt,kk,x)
            open(SRC,'w').write(txt); build()
            print('ours, %s: tracker compiled (%s)'%(lab,', '.join('%s %g'%(kk,x) for kk,x in cfg.items()) or 'baseline track.h'),flush=True)
            for s in todo_own[v]: one(v,s,PM,CFG_OWN,'ours, '+lab)
    finally:
        if os.path.exists(BAK): shutil.copy(BAK,SRC)
        else: print('!! no %s -- track.h was NOT restored; run: python3 /workspace/p55_izvorna_rezolucija.py --popravi'%BAK,flush=True)
        try: build(); okb=True
        except Exception as e: okb=False; print('!! compilation of the baseline track.h failed: %s'%e,flush=True)
        if okb and os.path.exists(BAK) and open(SRC).read()==open(BAK).read(): os.remove(BAK)
        print('track.h restored to the baseline configuration%s.'%(' and compiled' if okb else ' (COMPILATION FAILED -- report this)'),flush=True)
        for p,hs in sh.items():
            b=p+'.p55bak'
            if os.path.exists(p) and sha(p)==hs:
                if os.path.exists(b): os.remove(b)
                print('   check at the end: %s the same as at the start (SHA256)'%os.path.basename(p),flush=True)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p,p+'.p55_nepoznat')
                os.replace(b,p); print('   !! %s changed -- RESTORED from the backup copy; report this'%os.path.basename(p),flush=True)
        pusti_lock()
    print('GPU part done (%.0f min).'%((time.time()-t0)/60),flush=True)
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
            hh=Rt[q]['pedestrian']; m=lambda k: float(np.mean(hh['HOTA'][k]))
            res[int(q[1:])]={'HOTA':m('HOTA'),'DetA':m('DetA'),'AssA':m('AssA'),'AssRe':m('AssRe'),'AssPr':m('AssPr'),'LocA':m('LocA'),
                             'IDF1':float(hh['Identity']['IDF1']),'MOTA':float(hh['CLEAR']['MOTA']),'IDSW':float(hh['CLEAR']['IDSW'])}
        C[key]={'sig':sg,'res':{str(s):v for s,v in res.items()}}; json.dump(C,open(O+'/te_cache.json','w')); return res
    NAME={}; FILES={}
    for k,_,lab in NV: FILES['nv_'+k]={s:outp('nv_'+k,s) for s in SS16}; NAME['nv_'+k]=lab
    for v,_,lab in OWNV: FILES[v]={s:outp(v,s) for s in SS16}; NAME[v]='ours, '+lab
    for lab,pt in ULT: FILES[lab]={s:pt%s for s in SS16}; NAME[lab]=lab
    for k,pt in OLD.items(): FILES['old_'+k]={s:pt%s for s in SS16}; NAME['old_'+k]=NAME[k]+' (1920x1080)'
    P('=== p55: DeepStream trackers with the streammux at the native resolution, YOLO11m, 16 SSMOT sequences (TrackEval)')
    st=stanje(); P('    native resolution: %s'%('x'.join(str(x) for x in st.get('rezolucija',['?','?']))))
    R={}
    for k,f in FILES.items():
        miss=[s for s,p in f.items() if not json_ok(p)]
        if miss: P('  (no complete logs for "%s": %d of 16 missing)'%(NAME[k],len(miss))); continue
        R[k]=ev(k,f); print('   TrackEval %-36s ok'%NAME[k],flush=True)
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
        d=np.array(d,float); b=rng.choice(d,(10000,len(d))).mean(1); return [float(x) for x in np.percentile(b,[2.5,97.5])]
    fp=lambda p: '< 0.001' if p<0.001 else '%.3f'%p
    MET=['HOTA','DetA','AssA','AssPr','IDF1','IDSW']
    mean=lambda k,m,seqs=SS16: float(np.mean([R[k][s][m] for s in seqs]))
    def head(): P('  %-36s '%''+'  '.join('%6s'%m for m in MET))
    def row(k,seqs=SS16,lab=None): P('  %-36s '%(lab or NAME[k])+'  '.join(('%6.1f' if m=='IDSW' else '%6.3f')%mean(k,m,seqs) for m in MET))
    def dl(a,b,m,seqs=SS16): return [R[b][s][m]-R[a][s][m] for s in seqs]
    def show(pairs,seqs=SS16,holm_over=False,title='',mets=MET):
        P('  %s'%title); P('  %-46s %s'%('',' | '.join('%-25s'%m for m in mets)))
        res={}; ps={m:[wp(dl(a,b,m,seqs)) for a,b,_ in pairs] for m in mets}
        hs={m:(holm(ps[m]) if holm_over else [None]*len(pairs)) for m in mets}
        for i,(a,b,lab) in enumerate(pairs):
            cells=[]
            for m in mets:
                d=dl(a,b,m,seqs); e=float(np.mean(d))
                cells.append((('%+6.2f' if m=='IDSW' else '%+.4f')%e+' p=%s%s'%(fp(ps[m][i]),(' H=%s'%fp(hs[m][i])) if hs[m][i] is not None else '')).ljust(25))
                res.setdefault(lab,{})[m]={'d':e,'p':ps[m][i],'holm':hs[m][i],'ci':ci(d),'bolje':int(sum(1 for x in d if (x<0 if m=='IDSW' else x>0))),
                                            'losije':int(sum(1 for x in d if (x>0 if m=='IDSW' else x<0)))}
            P('  %-46s %s'%(lab,' | '.join(cells)))
        return res
    OUT={}
    # ---- A) main comparison with SOTA
    refs=[k for k in ['nv_nvsort','nv_deep','nv_dcf1','nv_dcf2']+[l for l,_ in ULT] if k in R]
    if 'final' in R and refs:
        P('\n=== A) MAIN COMPARISON (IEEE): 16 SSMOT sequences, YOLO11m; DeepStream trackers at 1280x720, Ultralytics on the native sequences')
        head()
        for k in refs: row(k)
        if 'base' in R: row('base')
        row('final')
        OUT['sota']=show([(k,'final','final - %s'%NAME[k]) for k in refs],holm_over=True,mets=['HOTA','DetA','AssA','IDF1','IDSW'],
                         title='final variant minus reference (Wilcoxon, n = 16; H = Holm over %d comparisons per metric):'%len(refs))
        rank={m:sorted(refs+['final'],key=lambda k: mean(k,m),reverse=(m!='IDSW')) for m in ('HOTA','DetA','AssA','IDF1','IDSW')}
        P('  ranking (best first): '+' | '.join('%s: %s'%(m,', '.join(NAME[k].replace('ours, ','') for k in rank[m][:3])+' ...') for m in rank))
        OUT['poredak']={m:rank[m] for m in rank}
    # ---- B) rule p52 at 1280x720
    if 'base' in R and 'final' in R:
        P('\n=== B) REPEATING THE p52 RULE AT 1280x720 (final versus baseline; IDSW significantly lower and dHOTA >= -0.005)')
        d=show([('base','final','final - baseline')],title='')
        e=d['final - baseline']; ok=e['IDSW']['p']<0.05 and e['IDSW']['d']<0 and e['HOTA']['d']>=-0.005
        P('  -> %s'%('rule MET' if ok else 'rule NOT met (IDSW %+.2f, p = %s; dHOTA %+.4f)'%(e['IDSW']['d'],fp(e['IDSW']['p']),e['HOTA']['d'])))
        OUT['pravilo_p52']={'delta':e,'ispunjeno':bool(ok)}
    # ---- C) effect of the native resolution
    pairs=[('old_'+k,k,NAME[k]) for k in OLD if k in R and 'old_'+k in R]
    if pairs:
        P('\n=== C) EFFECT OF THE NATIVE STREAMMUX RESOLUTION (1280x720 minus 1920x1080, same tracker, paired differences)')
        OUT['rezolucija']=show(pairs,title='',mets=['HOTA','DetA','AssA','IDF1','IDSW'])
    # ---- D) decomposition
    if all(k in R for k in ('base','cls','prob','final')):
        P('\n=== D) DECOMPOSITION AT 1280x720')
        head()
        for k in ('base','cls','prob','final'): row(k)
        OUT['rastav']=show([('base','cls','penalty only - baseline'),('base','prob','probation only - baseline')],holm_over=True,title='single components (H = Holm over 2):')
        OUT['rastav'].update(show([('base','final','final - baseline'),('cls','final','final - penalty only (adding the probation)'),
                                   ('prob','final','final - probation only (adding the penalty)')],title='final and the addition of the other component:'))
        P('  additivity:')
        for m in ('HOTA','IDF1','IDSW'):
            a=np.mean(dl('base','cls',m)); b=np.mean(dl('base','prob',m)); c=np.mean(dl('base','final',m))
            P('     %-5s penalty %+.4f + probation %+.4f = %+.4f   versus final %+.4f'%(m,a,b,a+b,c))
        P('  without SSMOT 10 (descriptive):'); head()
        for k in ('base','cls','prob','final'): row(k,SS15)
        P('  SSMOT 10 (IDSW / HOTA): '+' | '.join('%s %d / %.3f'%(NAME[k],R[k][10]['IDSW'],R[k][10]['HOTA']) for k in ('base','cls','prob','final')))
    # ---- E) sensitivity
    sens=[k for k in ('final','l02','l03','c07') if k in R]
    if len(sens)>1:
        P('\n=== E) SENSITIVITY AT 1280x720 (main = penalty 0.5 and threshold 0.8)')
        head()
        for k in sens: row(k)
        OUT['osjetljivost']=show([('final',k,'%s - main'%NAME[k]) for k in sens[1:]],title='')
        for m in ('HOTA','IDF1','IDSW'):
            vals=[mean(k,m) for k in sens]; P('  range %-5s: %.4f (%.4f - %.4f)'%(m,max(vals)-min(vals),min(vals),max(vals)))
    # ---- F) switch diagnosis
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
    P('\n=== F) SWITCH DIAGNOSIS AT 1280x720 (self-check = count equal to TrackEval IDSW)')
    OUT['dijagnoza']={}
    for k in ('base','cls','prob','final'):
        if k not in R: continue
        allev=[]; okc=True; clsch=0; known=0; prs=Counter()
        for s in SS16:
            p=FILES[k][s]; tr=load_tr(p); g=load_gt(GT%s,pick(GT%s,tr)); Cl=load_cls(p); evs=te_match(g,tr)
            if len(evs)!=int(R[k][s]['IDSW']): okc=False
            for e in evs:
                e['seq']=s; co=Cl.get((e['fold'],e['old'])); cn=Cl.get((e['f'],e['new']))
                if co is not None and cn is not None:
                    known+=1
                    if co!=cn: clsch+=1; a,b=sorted((int(co),int(cn))); prs[(a,b)]+=1
            allev+=evs
        n=len(allev); g_=np.array([e['gap'] for e in allev]) if n else np.zeros(0)
        byk=defaultdict(list)
        for e in allev: byk[(e['seq'],e['gt'])].append(e['f'])
        iv=[x for v in byk.values() for x in np.diff(sorted(v))]
        nm=lambda i: names[i] if names and 0<=i<len(names) else 'class %d'%i
        P('  %-30s switches %3d (self-check %s)%s'%(NAME[k],n,'PASSED' if okc else 'FAILED',
          (' | previous frame %.0f%% | return %.0f%% | median %s | class change %d of %d (%.0f%%): %s'%(100*(g_==0).mean(),100*np.mean([e['ret'] for e in allev]),
           ('%.0f'%np.median(iv)) if iv else '-',clsch,known,100*clsch/max(known,1),', '.join('%s<->%s %d'%(nm(a),nm(b),c) for (a,b),c in prs.most_common(3)) or '-')) if n else ''))
        OUT['dijagnoza'][k]={'n':n,'samoprovjera':okc,'promjena_klase':clsch,'poznato':known}
    # ---- G) speed
    P('\n=== G) SPEED AT 1280x720 (whole sequence; slope = pure throughput)')
    tm=st.get('vrijeme',{}); OUT['brzina']={}
    for key,lab in [('nv_'+k,l) for k,_,l in NV]+[(v,'ours, '+l) for v,_,l in OWNV]:
        pts=[tm['%s|%d'%(key,s)] for s in SS16 if '%s|%d'%(key,s) in tm]
        if len(pts)>=3:
            fr=np.array([p[0] for p in pts],float); sec=np.array([p[1] for p in pts],float); sl,ic=np.polyfit(fr,sec,1)
            P('  %-36s %5.1f FPS with start-up | from the slope %5.1f FPS, start-up about %3.0f s'%(lab,fr.sum()/sec.sum(),1/sl if sl>0 else float('nan'),ic))
            OUT['brzina'][lab]={'s_pokretanjem':float(fr.sum()/sec.sum()),'nagib':float(1/sl) if sl>0 else None}
    OUT['srednje']={k:{m:mean(k,m) for m in MET+['AssRe','MOTA','LocA']} for k in R}
    OUT['po_snimkama']={k:{str(s):v for s,v in R[k].items()} for k in R}
    json.dump(OUT,open(O+'/p55_sve.json','w'),indent=1,default=float)
    open(S+'/p55_izvorna_rezolucija.txt','w').write('\n'.join(L)+'\n'); print('\nresult: %s/p55_izvorna_rezolucija.txt'%S)
def popravi():
    """13.10.: p55 was run twice at the same time -> track.h stayed at the last variant, logs unreliable.
    Restores the 5 method parameters to the baseline values (the only thing p55 changes), checks that nothing else differs,
    compiles, checks the engines and moves all p55 logs aside to scpilot/p55_dvostruko (nothing is deleted)."""
    uzmi_lock('p55')
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running -- wait.'%os.path.basename(lk))
    BASEV={'classGate':1,'classPenalty':0.0,'confHigh':2.0,'maxAgeSec':-1.0,'minHitsSec':-1.0}
    old=open(SRC).read(); txt=old
    print('track.h before the fix: '+', '.join('%s %g'%(n,getv(old,n)[1]) for n in BASEV),flush=True)
    for n,v in BASEV.items(): txt=setv(txt,n,v)
    bad=check_base(txt)
    if bad: sys.exit('!! after restoring, track.h is still not the baseline -- NOTHING was written:\n   '+'\n   '.join(bad))
    a=old.splitlines(); b=txt.splitlines()
    if len(a)!=len(b) or any(x!=y and not any(re.search(r'\b%s\b'%n,x) for n in BASEV) for x,y in zip(a,b)):
        sys.exit('!! track.h also differs in other lines -- NOTHING was written; report this')
    if txt!=old:
        shutil.copy2(SRC,SRC+'.p55_prije_popravka'); open(SRC,'w').write(txt)
    build(); print('track.h restored to the baseline values (%d lines changed) and compiled; copy before the fix: track.h.p55_prije_popravka'%sum(1 for x,y in zip(a,b) if x!=y),flush=True)
    vrati()
    for p in PROT0:
        if os.path.exists(p+'.p55bak'): print('   !! %s.p55bak still exists -- report this'%os.path.basename(p),flush=True)
    dst=S+'/p55_dvostruko'; os.makedirs(dst,exist_ok=True); n=0
    import glob
    for p in glob.glob(W+'/log_p55_*.json')+[O+'/stanje.json',O+'/te_cache.json']:
        if os.path.exists(p): os.replace(p,dst+'/'+os.path.basename(p)); n+=1
    print('moved %d files to %s -- p55 can be run from the start (ONCE).'%(n,dst),flush=True)
if __name__=='__main__':
    if '--popravi' in sys.argv: popravi()
    elif '--vrati' in sys.argv:
        os.makedirs(O,exist_ok=True)
        uzmi_lock('p55'); vrati(); print('done.')
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
