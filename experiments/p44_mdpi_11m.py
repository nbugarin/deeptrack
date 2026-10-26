#!/usr/bin/env python3
"""p44: MDPI WITH THE YOLO11m DETECTOR, SSMOT ONLY (16 sequences: 1-18 without 7 and 13).

Steps (everything is saved on the fly; a new run skips what is done):
  1. converts YOLO11m for DeepStream (DeepStream-Yolo/utils/export_yolo11.py -> best.onnx and labels.txt in the weights folder)
  2. makes a NEW file DeepStream-Yolo/config_infer_11m.txt: a copy of config_infer_boat.txt (detector from the MDPI paper) in which
     ONLY the paths (onnx, engine, labels) and the number of classes are changed. config_infer_boat.txt and the other files are NOT changed.
  3. builds the TensorRT engine (FP16) for YOLO11m once under its own name (weights/best_11m_b1_gpu0_fp16.engine), in a separate folder.
     The engine of detector v1 (/workspace/model_b1_gpu0_fp16.engine) stays in place: a backup copy is made before the start,
     it is checked after every run, and its SHA256 is compared at the end.
  4. four NVIDIA configurations (NvSORT, NvDeepSORT, NvDCF reidType 1 and 2) on 16 SSMOT sequences -> /workspace/log_ss11m_<label>_<N>.json
  5. Ultralytics ByteTrack and BoT-SORT with the same YOLO11m (threshold 0.1), as an external reference
  6. evaluation with TrackEval (HOTA, DetA, AssA, IDF1, MOTA, IDSW) and everything MDPI needs: per sequence, range and ratios, Wilcoxon (n = 16),
     number of distinct IDs, circumstances of the switches (with a self-check), chain resolution, detector coverage, comparison with
     the old detector best_0904 on the same 16 sequences, throughput (FPS)
NEEDS A GPU, about 1-1.5 h.   Output: /workspace/scpilot/p44_mdpi_11m.txt (+ all numbers in scpilot/p44/p44_sve.json)
Run (container with DeepStream):         nohup python3 /workspace/p44_mdpi_11m.py > /workspace/p44_out.txt 2>&1 &
Monitoring:                              tail -3 /workspace/p44_out.txt
Re-evaluation only (without a GPU):      python3 /workspace/p44_mdpi_11m.py --eval
If interrupted (e.g. computer shut down): python3 /workspace/p44_mdpi_11m.py --vrati   (checks and if needed restores the engine of detector v1)
"""
import os, sys, re, json, glob, time, shutil, hashlib, subprocess
import numpy as np
for a,b in (('float',float),('int',int),('bool',bool)):
    if not hasattr(np,a): setattr(np,a,b)
W='/workspace'; S=W+'/scpilot'; O=S+'/p44'
WD=W+'/det_glavni/B_yolo11m/weights'; PT=WD+'/best.pt'; ONNX=WD+'/best.onnx'; LAB=WD+'/labels.txt'; ARGS=os.path.dirname(WD)+'/args.yaml'
SHA_PT='3ae0e1dcb939e87968cedc0b5fa8e294fa7915d0abada1a5752e707a0e606ca7'
EXPORT=W+'/DeepStream-Yolo/utils/export_yolo11.py'
TPLS=[W+'/DeepStream-Yolo/config_infer_boat.txt',W+'/DeepStream-Yolo/config_infer_B.txt']
INFER=W+'/DeepStream-Yolo/config_infer_11m.txt'; ENG=WD+'/best_11m_b1_gpu0_fp16.engine'
V1ENG=W+'/model_b1_gpu0_fp16.engine'; V1BAK=V1ENG+'.p44bak'; SKL=V1ENG+'.p44_skl'
LOCK=O+'/p44.lock'; BD=O+'/gradnja'; PROBE=W+'/probe_to_json.py'
CFG={'nvsort':W+'/config_tracker_NvSORT.yml','deep':W+'/config_tracker_maritime.yml','dcf1':W+'/cfg_dcf_novis.yml','dcf2':W+'/cfg_nvdcf.yml'}
NAME={'nvsort':'NvSORT','deep':'NvDeepSORT','dcf1':'NvDCF, reidType 1','dcf2':'NvDCF, reidType 2',
      'ultra_bytetrack':'Ultralytics ByteTrack','ultra_botsort':'Ultralytics BoT-SORT'}
P38={'nvsort':'NvSORT','deep':'NvDeepSORT','dcf1':'NvDCF r1','dcf2':'NvDCF r2'}
SS=[1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]; N=len(SS)
VID=W+'/SSMOT_video/video/SSMOT_%d.mp4'; GT=W+'/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
NEW=W+'/log_ss11m_%s_%d.json'; OLD=W+'/log_ss_%s_%d.json'; ULT=O+'/ultra_%s_%d.json'
T_RUN=1200; T_BUILD=2700
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
        with open(p) as f: json.load(f)
        return True
    except Exception: return False
_LG={}
def last_gt(s):
    if s not in _LG: _LG[s]=max(int(l.split(',')[0]) for l in open(GT%s) if l.strip())
    return _LG[s]
def vstat(): return (os.stat(V1ENG).st_size,os.stat(V1ENG).st_mtime_ns) if os.path.exists(V1ENG) else None
def tail(r): return (r.stdout or '')[-2500:]+'\n----\n'+(r.stderr or '')[-3500:]
def imgsz():
    try:
        for l in open(ARGS):
            if l.startswith('imgsz:'): return int(l.split(':',1)[1].strip())
    except Exception: pass
    return 640
def preflight():
    try:
        import scipy, trackeval   # noqa: F401
    except Exception as e:
        sys.exit('!! missing package (%s). The container was probably restarted -- first run:\n   %s'%(e,REINSTALL))
def vrati():
    """restores the engine of detector v1 if the previous run was interrupted"""
    msg=[]
    if os.path.exists(SKL):
        if os.path.exists(V1ENG): os.replace(V1ENG,V1ENG+'.p44_nepoznat')
        os.replace(SKL,V1ENG); msg.append('moved-aside engine put back in place')
    if os.path.exists(V1BAK):
        if os.path.exists(V1ENG) and sha(V1ENG)==sha(V1BAK): os.remove(V1BAK); msg.append('untouched (SHA256 the same as the copy), copy deleted')
        else:
            if os.path.exists(V1ENG): os.replace(V1ENG,V1ENG+'.p44_nepoznat')
            os.replace(V1BAK,V1ENG); msg.append('RESTORED from the backup copy')
    print('   engine of detector v1: %s'%('; '.join(msg) if msg else 'no copy and no moved-aside engine -- nothing to restore'),flush=True)
# ======================================================================= 1-5: runs (GPU)
def write_infer(labels):
    tpl=None
    for t in TPLS:
        if not os.path.exists(t): continue
        txt=open(t).read()
        if re.search(r'^\s*\[class-attrs-\d+\]',txt,re.M) or re.search(r'^\s*filter-out-class-ids',txt,re.M):
            print('   %s has settings tied to individual classes -- not using it as a template'%os.path.basename(t)); continue
        tpl=t; break
    if tpl is None: sys.exit('!! no suitable template (config_infer_boat.txt / config_infer_B.txt) -- tell Claude')
    out=[]; seen=set()
    for line in open(tpl):
        k=line.split('=')[0].strip()
        if k=='onnx-file': line='onnx-file=%s\n'%ONNX
        elif k=='model-engine-file': line='model-engine-file=%s\n'%ENG
        elif k=='labelfile-path': line='labelfile-path=%s\n'%LAB
        elif k=='num-detected-classes': line='num-detected-classes=%d\n'%len(labels)
        else: k=None
        if k: seen.add(k)
        out.append(line)
    if len(seen)<4: sys.exit('!! %s does not have all the expected fields (found: %s) -- tell Claude'%(os.path.basename(tpl),sorted(seen)))
    open(INFER,'w').write(''.join(out))
    print('2) created %s (template %s; only onnx-file, model-engine-file, labelfile-path, num-detected-classes changed):'%(INFER,os.path.basename(tpl)),flush=True)
    for l in out:
        if l.strip() and not l.strip().startswith('#'): print('     '+l.rstrip())
def build_engine():
    shutil.rmtree(BD,ignore_errors=True); os.makedirs(BD)
    s0=min(SS,key=last_gt); t_all=time.time(); tails=[]
    print('3) building the TensorRT engine for YOLO11m (FP16, only once; on a GTX 1660 SUPER it can take 5-20 min) ...',flush=True)
    for k,cwd in ((1,BD),(2,W)):
        if k==2:
            print('   the first attempt gave no engine -- second attempt from /workspace (engine v1 temporarily moved aside; a copy exists)',flush=True)
            if os.path.exists(V1ENG): os.replace(V1ENG,SKL)
        t_start=time.time()
        try:
            r=subprocess.run(['python3',PROBE,VID%s0,CFG['nvsort'],BD+'/gradnja_%d.json'%k,INFER],cwd=cwd,capture_output=True,text=True,timeout=T_BUILD)
            txt=(r.stdout or '')+'\n'+(r.stderr or '')
        except subprocess.TimeoutExpired:
            txt='time limit exceeded (%d s)'%T_BUILD
        tails.append('--- attempt %d (working folder %s)\n'%(k,cwd)+txt[-6000:])
        if not os.path.exists(ENG):
            cand=[]
            for p in re.findall(r'serialize cuda engine to file:\s*(\S+)\s+successfully',txt):
                p=p if os.path.isabs(p) else os.path.join(cwd,p)
                if os.path.exists(p): cand=[p]
            if not cand:
                cand=[p for d in (cwd,WD,W,W+'/DeepStream-Yolo') for p in glob.glob(d+'/*.engine') if os.path.getmtime(p)>=t_start-1]
            if cand:
                src=max(cand,key=os.path.getmtime); shutil.move(src,ENG); print('   engine: %s -> %s'%(src,ENG),flush=True)
        if k==2 and os.path.exists(SKL):
            if os.path.exists(V1ENG): os.replace(V1ENG,V1ENG+'.p44_nepoznat')
            os.replace(SKL,V1ENG)
        if os.path.exists(ENG): break
    if not os.path.exists(ENG):
        open(O+'/greska.txt','w').write('\n\n'.join(tails))
        sys.exit('!! the engine for YOLO11m was not created -- send to Claude: tail -60 /workspace/scpilot/p44/greska.txt')
    print('   engine done: %.0f MB, %.1f min'%(os.path.getsize(ENG)/2**20,(time.time()-t_all)/60),flush=True)
    shutil.rmtree(BD,ignore_errors=True)
def deepstream():
    v1sha=None
    if os.path.exists(V1ENG):
        v1sha=sha(V1ENG); shutil.copy2(V1ENG,V1BAK)
        if sha(V1BAK)!=v1sha: os.remove(V1BAK); sys.exit('!! the backup copy of engine v1 is not valid -- STOP, tell Claude')
        print('0) engine of detector v1: backup copy made',flush=True)
    try:
        isz=imgsz()
        if not (os.path.exists(ONNX) and os.path.exists(LAB)):
            print('1) conversion YOLO11m -> ONNX (imgsz %d, 1-3 min) ...'%isz,flush=True); t0=time.time()
            r=subprocess.run(['python3',EXPORT,'-w','best.pt','-s',str(isz)],cwd=WD,capture_output=True,text=True)
            if not os.path.exists(ONNX):
                c=[p for p in glob.glob(WD+'/*.onnx') if os.path.getmtime(p)>=t0-1]
                if len(c)==1: shutil.move(c[0],ONNX)
            if not (os.path.exists(ONNX) and os.path.exists(LAB)): sys.exit('!! conversion failed -- send to Claude:\n'+tail(r))
        labels=[l.strip() for l in open(LAB) if l.strip()]
        print('   ONNX %s (%.0f MB) | classes %d: %s'%(ONNX,os.path.getsize(ONNX)/2**20,len(labels),', '.join(labels)),flush=True)
        write_infer(labels)
        if not os.path.exists(ENG): build_engine()
        else: print('3) the engine for YOLO11m already exists: %s'%ENG,flush=True)
        todo=[(t,s) for t in CFG for s in SS if not json_ok(NEW%(t,s))]
        tm=json.load(open(O+'/vrijeme.json')) if os.path.exists(O+'/vrijeme.json') else {}
        print('4) DeepStream runs remaining: %d of %d'%(len(todo),len(CFG)*N),flush=True); t_loop=time.time()
        for i,(t,s) in enumerate(todo,1):
            st=vstat(); t0=time.time()
            try:
                r=subprocess.run(['python3',PROBE,VID%s,CFG[t],NEW%(t,s),INFER],cwd=W,capture_output=True,text=True,timeout=T_RUN); err=tail(r)
            except subprocess.TimeoutExpired: err='time limit exceeded (%d s)'%T_RUN
            dt=time.time()-t0
            if vstat()!=st:
                if os.path.exists(NEW%(t,s)): os.remove(NEW%(t,s))
                open(O+'/greska.txt','w').write(err)
                sys.exit('!! the engine of detector v1 changed during the run (did DeepStream build a new engine?) -- restoring from the copy; '
                         'send to Claude: tail -40 /workspace/scpilot/p44/greska.txt')
            ok=json_ok(NEW%(t,s)); note=''
            if ok:
                d=json.load(open(NEW%(t,s))); mx=max([int(x['frame']) for x in d]+[0])
                tm['%s|%d'%(t,s)]=[len(d),dt]; json.dump(tm,open(O+'/vrijeme.json','w'))
                if mx<last_gt(s)-2: note='  !! the log ends at frame %d, the annotations at %d'%(mx,last_gt(s))
            eta=(time.time()-t_loop)/i*(len(todo)-i)/60
            print('   [%2d/%d] %-7s SSMOT_%-2d %s (%.0f s, about %.0f min of DeepStream left)%s'%(i,len(todo),t,s,'ok' if ok else 'ERROR',dt,eta,note),flush=True)
            if not ok:
                open(O+'/greska.txt','w').write(err)
                sys.exit('!! the run failed -- send to Claude: tail -40 /workspace/scpilot/p44/greska.txt')
    finally:
        if v1sha is not None:
            vrati()
            print('   check at the end: engine of detector v1 %s'%('the same as at the start (SHA256)' if os.path.exists(V1ENG) and sha(V1ENG)==v1sha
                  else '!! NOT THE SAME AS AT THE START -- tell Claude'),flush=True)
def ultra():
    todo=[(k,s) for k in ('bytetrack','botsort') for s in SS if not json_ok(ULT%(k,s))]
    if not todo: return
    print('5) Ultralytics reference with YOLO11m: %d runs remaining (about 1 min per sequence)'%len(todo),flush=True)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('   !! Ultralytics cannot be loaded (%s) -- skipping the reference, the rest is evaluated'%e,flush=True); return
    isz=imgsz()
    for k,s in todo:
        try:
            t0=time.time(); m=YOLO(PT); T={}
            for f,res in enumerate(m.track(source=VID%s,tracker=k+'.yaml',conf=0.1,imgsz=isz,stream=True,persist=True,verbose=False),1):
                if res.boxes is None or res.boxes.id is None: continue
                xy=res.boxes.xyxy.cpu().numpy(); ids=res.boxes.id.cpu().numpy().astype(int)
                T[f]=[{'id':int(i),'box':[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])]} for i,b in zip(ids,xy)]
            json.dump([{'frame':f,'detections':v} for f,v in sorted(T.items())],open(ULT%(k,s)+'.tmp','w')); os.replace(ULT%(k,s)+'.tmp',ULT%(k,s)); del m
            print('   Ultralytics %-9s SSMOT_%-2d ok (%.0f s)'%(k,s,time.time()-t0),flush=True)
        except Exception as e:
            print('   !! Ultralytics %s SSMOT_%d failed (%s: %s) -- skipping the reference, the rest is evaluated'%(k,s,type(e).__name__,e),flush=True); return
def runs():
    preflight(); os.makedirs(O,exist_ok=True)
    try:
        used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
        print('GPU memory in use: %d MiB'%used,flush=True)
        if used>1500: sys.exit('GPU is busy (>1500 MiB) -- something else is running on the GPU. Wait until it finishes.')
    except (IndexError,ValueError,FileNotFoundError): print('warning: nvidia-smi gave no data, continuing')
    if os.path.exists(LOCK):
        pid=open(LOCK).read().strip()
        if pid and os.path.exists('/proc/'+pid): sys.exit('!! p44 is already running (PID %s).'%pid)
    if os.path.exists(V1BAK) or os.path.exists(SKL):
        print('the previous run was interrupted -- first checking the engine of detector v1:'); vrati()
    need=[PT,EXPORT,PROBE]+list(CFG.values())+[VID%s for s in SS]+[GT%s for s in SS]
    miss=[p for p in need if not os.path.exists(p)]+([] if any(os.path.exists(t) for t in TPLS) else [TPLS[0]])
    if miss: sys.exit('!! missing:\n   '+'\n   '.join(miss))
    free=shutil.disk_usage(W).free/2**30
    if free<2: sys.exit('!! only %.1f GB free on the disk -- free some space and run again'%free)
    if os.environ.get('P44_TEST')!='1' and sha(PT)!=SHA_PT: sys.exit('!! best.pt is NOT the frozen YOLO11m (SHA256 does not match) -- tell Claude.')
    open(LOCK,'w').write(str(os.getpid()))
    try:
        deepstream(); ultra()
    finally:
        if os.path.exists(LOCK): os.remove(LOCK)
    return True
# ======================================================================= 6: evaluation (CPU)
def ocjena():
    preflight(); os.makedirs(O,exist_ok=True)
    import trackeval
    from scipy.stats import wilcoxon
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict
    H={}; exec(open(W+'/id_smooth.py').read().split('CFG={')[0].split('"""')[2],H)   # load_gt, load_tr, iou, pick, smooth
    load_tr,load_gt,pick,iou,smooth=H['load_tr'],H['load_gt'],H['pick'],H['iou'],H['smooth']
    EPS=np.finfo('float').eps
    ec=trackeval.Evaluator.get_default_eval_config()
    ec.update(PRINT_RESULTS=False,PRINT_CONFIG=False,OUTPUT_SUMMARY=False,OUTPUT_DETAILED=False,PLOT_CURVES=False,TIME_PROGRESS=False,USE_PARALLEL=False,DISPLAY_LESS_PROGRESS=True)
    rows=lambda d: sorted((int(f),int(o['id']),*map(float,o['box'][:4])) for f,Lst in d.items() for o in Lst)
    def te(pairs,tag):                               # pairs: {s: (gt, tr)} -> {s: metrics}
        base=O+'/te_'+tag; shutil.rmtree(base,ignore_errors=True); si={}
        for s,(g,tr) in pairs.items():
            G=rows(g); R_=rows(tr); seq='S%02d'%s; si[seq]=max([r[0] for r in G+R_]+[1])
            os.makedirs('%s/gt/%s/gt'%(base,seq),exist_ok=True); os.makedirs(base+'/tr/t/data',exist_ok=True)
            with open('%s/gt/%s/gt/gt.txt'%(base,seq),'w') as f:
                for fr,i,x,y,w,h in G: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n'%(fr,i,x,y,w,h))
            with open('%s/tr/t/data/%s.txt'%(base,seq),'w') as f:
                for fr,i,x,y,w,h in R_: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n'%(fr,i,x,y,w,h))
        dc=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(GT_FOLDER=base+'/gt',TRACKERS_FOLDER=base+'/tr',BENCHMARK='b',SPLIT_TO_EVAL='',SEQ_INFO=si,TRACKERS_TO_EVAL=['t'],DO_PREPROC=False,
                  SKIP_SPLIT_FOL=True,GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt',TRACKER_SUB_FOLDER='data',PRINT_CONFIG=False)
        M=[trackeval.metrics.HOTA({'PRINT_CONFIG':False}),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),trackeval.metrics.Identity({'PRINT_CONFIG':False})]
        out,_=trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)],M); Rt=out['MotChallenge2DBox']['t']; shutil.rmtree(base,ignore_errors=True)
        res={}
        for seq in si:
            h=Rt[seq]['pedestrian']
            res[int(seq[1:])]={'HOTA':float(np.mean(h['HOTA']['HOTA'])),'DetA':float(np.mean(h['HOTA']['DetA'])),'AssA':float(np.mean(h['HOTA']['AssA'])),
                               'IDF1':float(h['Identity']['IDF1']),'MOTA':float(h['CLEAR']['MOTA']),'IDSW':float(h['CLEAR']['IDSW'])}
        return res
    def load(p,s):
        tr=load_tr(p); return (load_gt(GT%s,pick(GT%s,tr)),tr)
    def dedup(sm):
        out={}; n=0
        for f,Lst in sm.items():
            seen=set(); o2=[]
            for o in Lst:
                o=dict(o)
                if o['id'] in seen: n+=1; o['id']=10**7+n
                seen.add(o['id']); o2.append(o)
            out[f]=o2
        return out
    r2=lambda b: [float('%.2f'%float(x)) for x in b[:4]]
    def te_match(gt,tr):                             # matching exactly as TrackEval CLEAR (as in p38)
        prev_ts={}; prev={}; last={}; held=defaultdict(set); ev=[]
        for f in sorted(set(gt)|set(tr)):
            G=gt.get(f,[]); Tt=tr.get(f,[])
            if not G or not Tt: continue
            gb=[r2(o['box']) for o in G]; tb=[r2(t['box']) for t in Tt]
            sim=np.array([[iou(a,b) for b in tb] for a in gb])
            cont=np.array([[1.0 if prev_ts.get(o['id'])==t['id'] else 0.0 for t in Tt] for o in G])
            sc=1000*cont+sim; sc[sim<0.5-EPS]=0
            rr,cc=linear_sum_assignment(-sc); new_ts={}
            for a,b in zip(rr,cc):
                if sc[a,b]<=0+EPS: continue
                k=G[a]['id']; nid=Tt[b]['id']
                if k in prev and prev[k]!=nid: ev.append((k,f,f-last[k]-1,nid in held[k]))
                prev[k]=nid; last[k]=f; held[k].add(nid); new_ts[k]=nid
            prev_ts=new_ts
        return ev
    def sig(files): return [[os.path.basename(p),os.path.getsize(p),int(os.path.getmtime(p))] for _,p in sorted(files.items())]
    cache=O+'/rezultati.json'
    try: C=json.load(open(cache))
    except Exception: C={}
    def cached(key,files,fn):
        sg=sig(files)
        if key in C and C[key].get('sig')==sg: return {int(s):v for s,v in C[key]['res'].items()}
        r=fn(); C[key]={'sig':sg,'res':{str(s):v for s,v in r.items()}}; json.dump(C,open(cache,'w')); return r
    # ---- loading and TrackEval
    sets={'11m|'+t:{s:NEW%(t,s) for s in SS} for t in CFG}
    sets.update({'11m|ultra_'+k:{s:ULT%(k,s) for s in SS} for k in ('bytetrack','botsort')})
    sets.update({'old|'+t:{s:OLD%(t,s) for s in SS} for t in CFG})
    R={}; D={}
    print('evaluating with TrackEval ...',flush=True)
    for k,files in sets.items():
        have=[s for s,p in files.items() if json_ok(p)]
        if len(have)<N: P('  (%s: incomplete, %d of %d logs -- skipping)'%(k,len(have),N)); continue
        def ld(k=k,files=files):
            if k not in D: D[k]={s:load(p,s) for s,p in files.items()}
            return D[k]
        R[k]=cached(k,files,lambda: te(ld(),k.replace('|','_')))
        if k.startswith('11m|') and not k.startswith('11m|ultra'): ld()
    T4=[t for t in CFG if '11m|'+t in R]
    if not T4: P('not a single complete configuration with YOLO11m -- nothing to evaluate'); return
    mean=lambda r,m,seqs=SS: float(np.mean([r[s][m] for s in seqs]))
    fp=lambda p: '< 0.0001' if p<0.0001 else '%.4f'%p
    def paired(a,b,m):
        d=np.array([a[s][m]-b[s][m] for s in SS]); rng=np.random.default_rng(0); lo,hi=np.percentile(rng.choice(d,(10000,len(d))).mean(1),[2.5,97.5])
        if not np.any(d!=0): p=1.0
        else:
            try: p=float(wilcoxon(d).pvalue)
            except ValueError: p=float('nan')
        return float(d.mean()),float(lo),float(hi),p,(int((d>0).sum()) if m!='IDSW' else int((d<0).sum()))
    def holm(ps):
        ps=[1.0 if p!=p else p for p in ps]; o=np.argsort(ps); adj=[0.0]*len(ps); run=0.0
        for r,i in enumerate(o): run=max(run,min(1.0,(len(ps)-r)*ps[i])); adj[i]=run
        return adj
    g11={t:R['11m|'+t] for t in T4}
    J={'te':{k:{str(s):v for s,v in r.items()} for k,r in R.items()}}
    # ---- A) main table
    P('=== p44: MDPI WITH YOLO11m, SSMOT %d sequences (%s), TrackEval, mean per sequence'%(N,', '.join(map(str,SS))))
    most={t:sum(1 for s in SS if g11[t][s]['IDSW']>max([g11[u][s]['IDSW'] for u in T4 if u!=t]+[-1])) for t in T4}
    mosteq={t:sum(1 for s in SS if g11[t][s]['IDSW']>=max(g11[u][s]['IDSW'] for u in T4)) for t in T4}
    P('%-24s %6s %6s %6s %6s %6s %7s   %s'%('system','MOTA','IDF1','HOTA','DetA','AssA','IDSW','most IDSW of %d (strict / with ties)'%len(T4)))
    for k in ['11m|'+t for t in CFG]+['11m|ultra_bytetrack','11m|ultra_botsort']:
        t=k.split('|')[1]
        if k not in R: P('  %-22s missing'%NAME[t]); continue
        r=R[k]; P('  %-22s %6.3f %6.3f %6.3f %6.3f %6.3f %7.1f   %s'%(NAME[t],mean(r,'MOTA'),mean(r,'IDF1'),mean(r,'HOTA'),mean(r,'DetA'),mean(r,'AssA'),mean(r,'IDSW'),
          ('%d/%d / %d/%d'%(most[t],N,mosteq[t],N)) if t in most else '(reference)'))
    P('  HOTA = sqrt(DetA x AssA): DetA = finding the vessels (detector), AssA = keeping the identity (tracker)')
    # ---- B) per sequence
    gtd={s:D['11m|'+T4[0]][s][0] for s in SS}
    nfr={s:max(gtd[s]) for s in SS}; ngt={s:len({o['id'] for f in gtd[s] for o in gtd[s][f]}) for s in SS}
    P('\n=== per sequence: HOTA / IDSW  (frames = last annotated frame, GT = number of GT tracks)')
    P('  %-4s %6s %4s  '%('SS','frames','GT')+' '.join('%-16s'%P38[t] for t in T4))
    for s in SS: P('  %-4d %6d %4d  '%(s,nfr[s],ngt[s])+' '.join('%.3f / %-8d'%(g11[t][s]['HOTA'],g11[t][s]['IDSW']) for t in T4))
    P('  TOTAL frames %d, GT tracks %d'%(sum(nfr.values()),sum(ngt.values())))
    if len(T4)==4:
        app=['deep','dcf1','dcf2']; I={t:mean(g11[t],'IDSW') for t in CFG}; Hm={t:mean(g11[t],'HOTA') for t in CFG}
        P('\n=== range and ratios (MDPI 4.2)')
        P('  IDSW: NvDeepSORT/NvDCF r2 = %.1f/%.1f = %.2fx;  /NvDCF r1 = %.2fx;  /NvSORT = %.2fx'%(I['deep'],I['dcf2'],I['deep']/max(I['dcf2'],1e-9),
          I['deep']/max(I['dcf1'],1e-9),I['deep']/max(I['nvsort'],1e-9)))
        P('  three with appearance (same appearance engine): HOTA from %.3f to %.3f (range %.3f); IDSW from %.1f to %.1f (ratio %.1f)'%(min(Hm[t] for t in app),max(Hm[t] for t in app),
          max(Hm[t] for t in app)-min(Hm[t] for t in app),min(I[t] for t in app),max(I[t] for t in app),max(I[t] for t in app)/max(min(I[t] for t in app),1e-9)))
        P('  SD of HOTA over sequences: '+', '.join('%s %.3f'%(NAME[t],np.std([g11[t][s]['HOTA'] for s in SS],ddof=1)) for t in CFG))
        low=sum(1 for s in SS if g11['deep'][s]['HOTA']<=min(g11[u][s]['HOTA'] for u in CFG))
        lowa=sum(1 for s in SS if g11['deep'][s]['HOTA']<=min(g11[u][s]['HOTA'] for u in app))
        P('  NvDeepSORT: most switches strictly in %d/%d (with ties %d/%d); lowest HOTA of all four in %d/%d, of the three with appearance in %d/%d'%(
          most['deep'],N,mosteq['deep'],N,low,N,lowa,N))
        P('  highest HOTA per sequence (with ties): '+', '.join('%s %d'%(NAME[t],sum(1 for s in SS if g11[t][s]['HOTA']>=max(g11[u][s]['HOTA'] for u in CFG))) for t in CFG))
        # ---- C) Wilcoxon
        P('\n=== Wilcoxon (paired, two-sided), n = %d; 95 %% bootstrap; Holm over 5 pairs for each metric'%N)
        pairs=[('deep','dcf2'),('deep','dcf1'),('deep','nvsort'),('dcf1','dcf2'),('nvsort','dcf1')]
        for m in ('IDSW','HOTA','IDF1'):
            res=[paired(g11[a],g11[b],m) for a,b in pairs]; adj=holm([x[3] for x in res])
            for (a,b),x,pa in zip(pairs,res,adj):
                fmt='%+.1f [%+.1f, %+.1f]' if m=='IDSW' else '%+.3f [%+.3f, %+.3f]'
                P('    %-18s - %-18s %-4s  '%(NAME[a],NAME[b],m)+fmt%(x[0],x[1],x[2])+'  p = %s  (Holm %s)  first better on %d/%d'%(fp(x[3]),fp(pa),x[4],N))
        # ---- D) number of distinct IDs
        P('\n=== number of distinct IDs per sequence (MDPI Table 5); first row = number of GT vessels')
        nid={t:{s:len({o['id'] for f in D['11m|'+t][s][1] for o in D['11m|'+t][s][1][f]}) for s in SS} for t in CFG}
        P('  %-20s '%'GT vessels'+' '.join('%4d'%ngt[s] for s in SS)+'  mean %.1f'%np.mean([ngt[s] for s in SS]))
        for t in CFG: P('  %-20s '%NAME[t]+' '.join('%4d'%nid[t][s] for s in SS)+'  mean %.1f'%np.mean([nid[t][s] for s in SS]))
        P('  NvDeepSORT has the most IDs (with ties) in %d/%d sequences'%(sum(1 for s in SS if nid['deep'][s]>=max(nid[u][s] for u in CFG)),N))
        J['ids']={t:{str(s):nid[t][s] for s in SS} for t in CFG}; J['gt']={str(s):[nfr[s],ngt[s]] for s in SS}
        # ---- E) circumstances of the switches
        P('\n=== circumstances of the switches (MDPI Table 6), matching exactly as TrackEval, summed over %d sequences'%N)
        ok=True; SW={}
        for t in CFG:
            ev=[]
            for s in SS:
                e=te_match(*D['11m|'+t][s])
                if len(e)!=int(g11[t][s]['IDSW']): ok=False; P('  !! %s SSMOT_%d: ours %d, TrackEval %d'%(NAME[t],s,len(e),g11[t][s]['IDSW']))
                ev+=[x+(s,) for x in e]
            gp=np.array([x[2] for x in ev]); n=max(len(gp),1)
            st=dict(n=len(gp),gap0=int((gp==0).sum()),gap1_5=int(((gp>=1)&(gp<=5)).sum()),gap_gt5=int((gp>5).sum()),ret=int(sum(x[3] for x in ev)),
                    med=float(np.median(gp+1)) if len(gp) else 0.0,tracks=len({(x[4],x[0]) for x in ev})); SW[t]=st
            P('  %-18s n=%4d  previous frame %5.1f%% (%d)  gap 1-5 %5.1f%% (%d)  >5 %5.1f%% (%d)  return to a held ID %5.1f%% (%d)  median %.0f  GT tracks %d'%(
              NAME[t],st['n'],100*st['gap0']/n,st['gap0'],100*st['gap1_5']/n,st['gap1_5'],100*st['gap_gt5']/n,st['gap_gt5'],100*st['ret']/n,st['ret'],st['med'],st['tracks']))
        P('  SELF-CHECK (switches per sequence = TrackEval IDSW): %s'%('PASSED' if ok else 'FAILED -- do NOT use these numbers'))
        J['sw']=SW; J['sw_selfcheck']=ok
        # ---- F) chain resolution
        P('\n=== chain resolution (id_smooth: LINK_IOU 0.5, MAXGAP 3), mean of %d sequences: before -> after'%N)
        J['chain']={}
        for t in CFG:
            def fn(t=t):
                pr={}
                for s in SS:
                    sm=smooth(D['11m|'+t][s][1]); sm=sm[0] if isinstance(sm,tuple) else sm; pr[s]=(D['11m|'+t][s][0],dedup(sm))
                return te(pr,'ch_'+t)
            c=cached('chain|'+t,sets['11m|'+t],fn); J['chain'][t]={str(s):v for s,v in c.items()}
            P('  %-18s HOTA %.3f->%.3f  IDF1 %.3f->%.3f  MOTA %.3f->%.3f  IDSW %.1f->%.1f'%(NAME[t],mean(g11[t],'HOTA'),mean(c,'HOTA'),mean(g11[t],'IDF1'),mean(c,'IDF1'),
              mean(g11[t],'MOTA'),mean(c,'MOTA'),mean(g11[t],'IDSW'),mean(c,'IDSW')))
        # ---- G) detector coverage
        rec=[]; prec=[]
        for s in SS:
            g,tr=D['11m|nvsort'][s]; miss=tot=extra=nd=0
            for f in sorted(set(g)&set(tr)):
                G=g[f]; Tt=tr[f]; tot+=len(G); nd+=len(Tt); used=set()
                for o in G:
                    hh=[k for k,tt in enumerate(Tt) if iou(o['box'],tt['box'])>=0.3]
                    if not hh: miss+=1
                    used.update(hh)
                extra+=len(Tt)-len(used)
            rec.append(1-miss/max(tot,1)); prec.append(1-extra/max(nd,1))
        P('\n=== detector coverage (NvSORT output, IoU >= 0.3), mean per sequence: recall %.3f   precision %.3f'%(np.mean(rec),np.mean(prec)))
        J['coverage']={'recall':rec,'precision':prec}
    # ---- H) external reference
    ref=[k for k in ('ultra_bytetrack','ultra_botsort') if '11m|'+k in R]
    if ref:
        P('\n=== NVIDIA configurations versus the Ultralytics reference (same YOLO11m), paired, n = %d, two-sided Wilcoxon'%N)
        for k in ref:
            for t in T4:
                out=[]
                for m in ('HOTA','DetA','AssA','IDSW'):
                    x=paired(g11[t],R['11m|'+k],m); out.append(('%s %+.1f p=%s' if m=='IDSW' else '%s %+.3f p=%s')%(m,x[0],fp(x[3])))
                P('  %-18s - %-21s %s'%(NAME[t],NAME[k],' | '.join(out)))
    # ---- I) new versus old detector
    olds=[t for t in T4 if 'old|'+t in R]
    if olds:
        P('\n=== new detector (YOLO11m) versus the old one (best_0904), same tracker configuration, same %d sequences'%N)
        p38=None
        try: p38=json.load(open(S+'/p38_ssmot17.json'))['te']
        except Exception: pass
        for t in olds:
            ro=R['old|'+t]; out=[]
            for m in ('HOTA','DetA','AssA','IDSW'):
                x=paired(g11[t],ro,m); out.append(('%s %+.1f p=%s' if m=='IDSW' else '%s %+.3f p=%s')%(m,x[0],fp(x[3])))
            chk=''
            if p38 and P38[t] in p38 and all(str(s) in p38[P38[t]] for s in SS):
                h38=np.mean([p38[P38[t]][str(s)]['HOTA'] for s in SS]); chk='  [self-check against p38: %s]'%('OK' if abs(h38-mean(ro,'HOTA'))<0.0005 else 'DIFFERENCE %.4f'%(mean(ro,'HOTA')-h38))
            P('  %-18s old HOTA %.3f IDSW %.1f -> new HOTA %.3f IDSW %.1f | %s%s'%(NAME[t],mean(ro,'HOTA'),mean(ro,'IDSW'),mean(g11[t],'HOTA'),mean(g11[t],'IDSW'),' | '.join(out),chk))
        late=[]
        for t in olds:
            mo=min(os.path.getmtime(OLD%(t,s)) for s in SS)
            if os.path.getmtime(CFG[t])>mo: late.append('%s (configuration %s, oldest old log %s)'%(NAME[t],time.strftime('%d.%m. %H:%M',time.localtime(os.path.getmtime(CFG[t]))),
                                                                                                      time.strftime('%d.%m. %H:%M',time.localtime(mo))))
        if late: P('  !! configuration file changed after the old logs -- the old/new comparison may not be clean: '+'; '.join(late))
        else: P('  configuration files not changed after the old logs (the old/new comparison uses the same tracker settings)')
    # ---- J) throughput
    if os.path.exists(O+'/vrijeme.json'):
        tm=json.load(open(O+'/vrijeme.json')); J['fps']={}
        P('\n=== DeepStream throughput (offline: probe_to_json, whole sequence; "with start-up" includes starting the pipeline, slope = pure throughput)')
        P('  %-18s '%'FPS per sequence'+' '.join('%5d'%s for s in SS))
        for t in T4:
            v={s:tm.get('%s|%d'%(t,s)) for s in SS}
            P('  %-18s '%NAME[t]+' '.join(('%5.1f'%(v[s][0]/v[s][1])) if v[s] and v[s][1] else '    -' for s in SS))
            pts=[v[s] for s in SS if v[s] and v[s][1]]
            if len(pts)>=3:
                fr=np.array([p[0] for p in pts],float); sec=np.array([p[1] for p in pts],float); slope,icpt=np.polyfit(fr,sec,1)
                J['fps'][t]={'s_pokretanjem':float(fr.sum()/sec.sum()),'nagib':float(1/slope) if slope>0 else None,'pokretanje_s':float(icpt)}
                P('  %-18s total %.1f FPS with start-up | from the slope %.1f FPS, start-up about %.0f s'%('',fr.sum()/sec.sum(),1/slope if slope>0 else float('nan'),icpt))
    json.dump(J,open(O+'/p44_sve.json','w'),indent=1)
    open(S+'/p44_mdpi_11m.txt','w').write('\n'.join(L)+'\n'); print('\nresult: %s/p44_mdpi_11m.txt  (all numbers: %s/p44_sve.json)'%(S,O))
# =======================================================================
if __name__=='__main__':
    if '--vrati' in sys.argv:
        pid=open(LOCK).read().strip() if os.path.exists(LOCK) else ''
        if pid and os.path.exists('/proc/'+pid): sys.exit('!! p44 is still running (PID %s) -- --vrati is used only when p44 was interrupted.'%pid)
        vrati()
    elif '--eval' in sys.argv: ocjena()
    else:
        if runs(): ocjena()
