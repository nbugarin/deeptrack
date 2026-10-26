#!/usr/bin/env python3
"""p52: VLASTITI TRACKER S DETEKTOROM YOLO11m (za IEEE) -- vrijedi li poboljsanje i s modernim detektorom?

Plan zapisan PRIJE pokretanja (projekt: claude/p52_vlastiti_11m_plan.md):
  glavni ishod (YOLO11m, 16 SSMOT snimaka kao u MDPI-ju): konacna varijanta prema polaznoj; poboljsanje VRIJEDI ako je IDSW
  znacajno manji (Wilcoxon, p < 0.05) i prosjecni dHOTA >= -0.005. Mehanizam: udio zamjena s promjenom klase (YOLO11m prema
  YOLOv8x). Usporedba s NvSORT, NvDeepSORT, NvDCF r1/r2, Ultralytics ByteTrack/BoT-SORT (logovi iz p44, Holm). Vlastitih 5 sporedno.

Koraci (GPU):
  1. provjeri da je track.h na polaznoj konfiguraciji (kao p43) -- inace NE pocinje; kopija track.h.bak_p52
  2. engine-i (model_b1_gpu0_fp16, best_0904, osnet_vessel, YOLO11m) -> sigurnosna kopija (.p52bak), provjera nakon svakog pokretanja
  3. polazna: prevede tracker; PROVJERA PONOVLJIVOSTI (stari detektor, jedna vlastita snimka, mora biti identicno log_p43_base);
     zatim YOLO11m (config_infer_11m.txt iz p44) na 16 SSMOT + vlastitih 5  -> /workspace/log_p52_base_<ss|own>_<n>.json
  4. konacna varijanta (iste vrijednosti kao p43 "konacna"): isto -> log_p52_final_<ss|own>_<n>.json
  5. na kraju (i kod greske): track.h se vraca i prevodi, engine-i se provjeravaju (SHA256) i po potrebi vracaju
  6. ocjena TrackEvalom + dijagnoza zamjena (promjena klase) + usporedbe
Pokretanje (kontejner s DeepStreamom, GPU slobodan):  nohup python3 /workspace/p52_vlastiti_11m.py > /workspace/p52_out.txt 2>&1 &
Pracenje:                                           tail -3 /workspace/p52_out.txt
Samo ocjena (bez GPU-a):                            python3 /workspace/p52_vlastiti_11m.py --eval
Ako je prekinuto (npr. gasenje racunala):           python3 /workspace/p52_vlastiti_11m.py --vrati   (pa ponovno pokreni -- nastavlja)
Izlaz: /workspace/scpilot/p52_vlastiti_11m.txt (+ scpilot/p52/p52_sve.json)
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
FINAL={'classGate':0,'classPenalty':0.5,'confHigh':0.8,'maxAgeSec':3.3333,'minHitsSec':0.6667}   # tocno kao p43 "konacna"
VARN={'base':'polazna','final':'konacna varijanta'}
LOG=W+'/log_p52_%s_%s_%d.json'                                                                    # (varijanta, skup, snimka)
def own_find(name): return [W+'/log_%s_%%d.json'%name,W+'/log_ss_%s_%%d.json'%name,W+'/log_%s_ss_%%d.json'%name]
V8={('ss','base'):own_find('ssmot'),('ss','final'):own_find('combo_p0p5_c0p8h2'),
    ('own','base'):[W+'/log_p43_base_%d.json'],('own','final'):[W+'/log_p43_konacna_%d.json',W+'/log_combo_p0p5_c0p8h2_own_%d.json']}
REF11=[('NvSORT',W+'/log_ss11m_nvsort_%d.json'),('NvDeepSORT',W+'/log_ss11m_deep_%d.json'),('NvDCF r1',W+'/log_ss11m_dcf1_%d.json'),
       ('NvDCF r2',W+'/log_ss11m_dcf2_%d.json'),('Ultralytics ByteTrack',S+'/p44/ultra_bytetrack_%d.json'),('Ultralytics BoT-SORT',S+'/p44/ultra_botsort_%d.json')]
LOCKS=[S+'/p43.lock',S+'/p27.lock']+[S+'/p%d/p%d.lock'%(i,i) for i in (44,45,46,47,48)]
T_RUN=900
REINSTALL=('apt-get update && apt-get install -y libeigen3-dev && pip install --no-deps git+https://github.com/JonathonLuiten/TrackEval.git '
           '&& python3 -c "import scipy, cv2, trackeval; print(\'SVE OK\')"')
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
# ======================================================================= track.h (isto kao p43)
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
        if x is None: bad.append('%s nije nadjen tocno jednom'%n)
        elif abs(x-v)>1e-6: bad.append('%s = %g (treba %g)'%(n,x,v))
    for n,okf in OFF.items():
        ty,x=getv(txt,n)
        if x is not None and not okf(x): bad.append('%s = %g (treba biti iskljuceno)'%(n,x))
    for n in FINAL:
        if getv(txt,n)[1] is None: bad.append('%s (konacna varijanta) nije nadjen tocno jednom'%n)
    return bad
def build():
    t0=time.time()
    r=subprocess.run('cd %s/mytracker && make clean >/dev/null 2>&1; make 2>&1'%W,shell=True,capture_output=True,text=True)
    if r.returncode!=0 or not os.path.exists(LIB) or os.path.getmtime(LIB)<t0-2:
        raise RuntimeError('prevodenje nije uspjelo:\n'+(r.stdout or '')[-1500:]+(r.stderr or '')[-1500:])
def vrati():
    if os.path.exists(BAK):
        shutil.copy(BAK,SRC)
        try: build()
        except Exception as e: print('!! prevodenje polaznog track.h nije uspjelo: %s'%e,flush=True)
        if open(SRC).read()==open(BAK).read(): os.remove(BAK)
        print('   track.h vracen na polaznu konfiguraciju i preveden.',flush=True)
    for p in PROT0:
        b=p+'.p52bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p)==sha(b): os.remove(b); print('   %s: netaknut, kopija obrisana'%os.path.basename(p),flush=True)
        else:
            if os.path.exists(p): os.replace(p,p+'.p52_nepoznat')
            os.replace(b,p); print('   %s: VRACEN iz sigurnosne kopije'%os.path.basename(p),flush=True)
# ======================================================================= GPU
def runs():
    try:
        import scipy, trackeval, cv2   # noqa: F401
    except Exception as e: sys.exit('!! nedostaje paket (%s). Kontejner je vjerojatno ponovno pokrenut -- prvo pokreni:\n   %s'%(e,REINSTALL))
    os.makedirs(O,exist_ok=True)
    if alive(LOCK): sys.exit('!! p52 vec radi (PID %s) -- ne pokreci dvaput.'%alive(LOCK))
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s jos radi (PID %s) -- pokreni p52 kad zavrsi.'%(os.path.basename(lk).split('.')[0],alive(lk)))
    if os.environ.get('P52_NOGPU')!='1':
        try:
            used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
            print('GPU memorija u upotrebi: %d MiB'%used,flush=True)
            if used>1500: sys.exit('GPU je zauzet (>1500 MiB) -- nesto drugo radi na GPU-u. Pricekaj da zavrsi.')
        except (IndexError,ValueError,FileNotFoundError): print('upozorenje: nvidia-smi nije dao podatak, nastavljam',flush=True)
    if not os.path.isdir('/usr/include/eigen3') and os.environ.get('P52_NOGPU')!='1': sys.exit('!! nema Eigena. Prvo: apt-get install -y libeigen3-dev')
    if os.path.exists(BAK) or any(os.path.exists(p+'.p52bak') for p in PROT0):
        print('prethodno pokretanje je prekinuto -- vracam track.h i provjeravam engine-e:',flush=True); vrati()
    need=[PROBE,CFG,INFER,ENG11,SRC,W+'/id_smooth.py']+[VID[ds]%s for ds in SEQS for s in SEQS[ds]]+[GT[ds]%s for ds in SEQS for s in SEQS[ds]]
    miss=[p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! nedostaje:\n   '+'\n   '.join(miss))
    if shutil.disk_usage(W).free/2**30<3: sys.exit('!! na disku je slobodno manje od 3 GB')
    orig=open(SRC).read(); bad=check_base(orig)
    if bad: sys.exit('!! track.h NIJE na polaznoj konfiguraciji -- nista nije pokrenuto:\n   '+'\n   '.join(bad)+'\n   Posalji ovaj ispis Claudeu.')
    mh=getv(orig,'minHitsHigh')[1]
    m=re.search(r'^\s*model-engine-file\s*=\s*(\S+)',open(INFER).read(),re.M)
    print('track.h na polaznoj konfiguraciji (provjereno); minHitsHigh u track.h = %s (ne mijenja se, kao u p43)'%('%g'%mh if mh is not None else 'nema'),flush=True)
    print('config_infer_11m.txt: model-engine-file = %s'%(m.group(1) if m else 'NEMA'),flush=True)
    st=stanje(); h_inf=sha(INFER)
    if 'infer_sha' in st and st['infer_sha']!=h_inf: sys.exit('!! config_infer_11m.txt promijenjen od prvog pokretanja p52 -- javi Claudeu')
    st['infer_sha']=h_inf; spremi(st)
    rep=min(OWN,key=lambda s: nframes(VID['own']%s))
    todo={v:[(ds,s) for ds in ('ss','own') for s in SEQS[ds] if not json_ok(LOG%(v,ds,s))] for v in ('base','final')}
    n_all=sum(len(t) for t in todo.values())+sum(1 for v in ('base','final') if v+'_repro' not in st)
    print('pokretanja preostalo: %d (16 SSMOT + 5 vlastitih po varijanti, + provjera ponovljivosti na vlastitoj snimci %d)'%(n_all,rep),flush=True)
    PROT=[p for p in PROT0 if os.path.exists(p)]
    print('zasticeni engine-i: %s'%', '.join(os.path.basename(p) for p in PROT),flush=True)
    shutil.copy2(SRC,BAK); open(LOCK,'w').write(str(os.getpid())); sh={}
    t0=time.time(); done=[0]
    try:
        for p in PROT:
            sh[p]=sha(p); shutil.copy2(p,p+'.p52bak')
            if sha(p+'.p52bak')!=sh[p]: sys.exit('!! sigurnosna kopija %s nije ispravna -- STANI'%p)
        def one(video,out,infer):
            st_={p:stat(p) for p in PROT}; env=dict(os.environ,MYTRACKER_FPS='%g'%fps(video)); rc=None
            try:
                r=subprocess.run(['python3',PROBE,video,CFG,out]+([infer] if infer else []),cwd=W,capture_output=True,text=True,env=env,timeout=T_RUN)
                rc=r.returncode; err=(r.stdout or '')[-2500:]+'\n----\n'+(r.stderr or '')[-3500:]
            except subprocess.TimeoutExpired: err='prekoraceno vrijeme (%d s)'%T_RUN
            ch=[os.path.basename(p) for p in PROT if stat(p)!=st_[p]]
            if ch:
                if os.path.exists(out): os.remove(out)
                open(O+'/greska.txt','w').write(err)
                sys.exit('!! promijenio se %s tijekom pokretanja (%s) -- vraca se iz kopije; posalji Claudeu: tail -40 /workspace/scpilot/p52/greska.txt'%(', '.join(ch),os.path.basename(video)))
            if json_ok(out): return True
            if os.path.exists(out): os.remove(out)
            open(O+'/greska.txt','w').write('%s, povratni kod %s\n'%(video,rc)+err); return False
        def tick(lab):
            done[0]+=1; el=time.time()-t0
            print('  [%2d/%d] %-34s ok   (jos oko %.0f min)'%(done[0],n_all,lab,el/done[0]*(n_all-done[0])/60),flush=True)
        for v in ('base','final'):
            if not todo[v] and v+'_repro' in st: continue
            txt=orig
            if v=='final':
                for k,x in FINAL.items(): txt=setv(txt,k,x)
            open(SRC,'w').write(txt); build()
            print('%s: tracker preveden (%s)'%(VARN[v],'polazni track.h' if v=='base' else ', '.join('%s %g'%(k,x) for k,x in FINAL.items())),flush=True)
            if v+'_repro' not in st:
                out=O+'/repro_%s_%d.json'%(v,rep); ref=first(V8[('own',v)],rep)
                if os.path.exists(out): os.remove(out)
                ok=one(VID['own']%rep,out,None) or one(VID['own']%rep,out,None)
                if not ok: res={'video':rep,'ishod':'pokretanje sa starim detektorom nije uspjelo (detalji u greska.txt)'}
                elif ref is None: res={'video':rep,'ishod':'nema referentnog loga'}
                else:
                    A=frames(out); B=frames(ref); nd=sum(1 for f in set(A)|set(B) if A.get(f)!=B.get(f))
                    res={'video':rep,'ref':os.path.basename(ref),'isto':nd==0,'razl_frejmova':nd,'frejmova':len(set(A)|set(B)),
                         'ishod':'IDENTICNO %s'%os.path.basename(ref) if nd==0 else '!! NIJE identicno %s (%d od %d frejmova razlicito)'%(os.path.basename(ref),nd,len(set(A)|set(B)))}
                st[v+'_repro']=res; spremi(st); tick('provjera ponovljivosti (%s)'%VARN[v]); print('     -> %s'%res['ishod'],flush=True)
            for ds,s in todo[v]:
                ok=one(VID[ds]%s,LOG%(v,ds,s),INFER) or one(VID[ds]%s,LOG%(v,ds,s),INFER)
                if not ok: sys.exit('!! %s, %s %d nije uspjelo ni u drugom pokusaju -- posalji Claudeu: tail -40 /workspace/scpilot/p52/greska.txt'%(VARN[v],'SSMOT' if ds=='ss' else 'vlastita',s))
                tick('%s, %s %d'%(VARN[v],'SSMOT' if ds=='ss' else 'vlastita',s))
    finally:
        shutil.copy(BAK,SRC)
        try: build(); okb=True
        except Exception as e: okb=False; print('!! prevodenje polaznog track.h nije uspjelo: %s'%e,flush=True)
        if okb and open(SRC).read()==open(BAK).read(): os.remove(BAK)
        print('track.h vracen na polaznu konfiguraciju%s.'%(' i preveden' if okb else ' (PREVODENJE NIJE USPJELO -- javi Claudeu)'),flush=True)
        for p,h in sh.items():
            b=p+'.p52bak'
            if os.path.exists(p) and sha(p)==h:
                if os.path.exists(b): os.remove(b)
                print('   provjera na kraju: %s isti kao na pocetku (SHA256)'%os.path.basename(p),flush=True)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p,p+'.p52_nepoznat')
                os.replace(b,p); print('   !! %s se promijenio -- VRACEN iz sigurnosne kopije; javi Claudeu'%os.path.basename(p),flush=True)
        if os.path.exists(LOCK): os.remove(LOCK)
    print('GPU dio gotov (%.0f min).'%((time.time()-t0)/60),flush=True)
    return True
# ======================================================================= ocjena
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
    # ---- skupovi logova
    SETS={}
    def add(key,ds,seqs,pats):
        f={s:first(pats,s) for s in seqs}
        if all(f.values()): SETS[key]=(ds,f)
        else: P('  (nema potpunih logova za "%s": %d od %d)'%(key,sum(1 for x in f.values() if x),len(seqs)))
    P('=== p52: vlastiti tracker s YOLO11m -- ocjena (TrackEval); plan zapisan prije pokretanja (claude/p52_vlastiti_11m_plan.md)')
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
        """b - a, uparene razlike po snimkama"""
        out={}
        for m in MET:
            d=[R[b][s][m]-R[a][s][m] for s in seqs]; lo,hi=ci(d); p=wp(d)
            better=sum(1 for x in d if (x<0 if m=='IDSW' else x>0)); worse=sum(1 for x in d if (x>0 if m=='IDSW' else x<0))
            out[m]={'d':float(np.mean(d)),'lo':float(lo),'hi':float(hi),'p':p,'bolje':better,'losije':worse,'n':len(d)}
        P('  %s:'%lab)
        for m in MET:
            e=out[m]; f_='%+.2f [%+.2f, %+.2f]' if m=='IDSW' else '%+.4f [%+.4f, %+.4f]'
            P('     %-5s '%m+f_%(e['d'],e['lo'],e['hi'])+'  p = %-7s bolje na %d, losije na %d od %d'%(fp(e['p']),e['bolje'],e['losije'],e['n']))
        return out
    OUT={'repro':{k:v for k,v in stanje().items() if k.endswith('_repro')}}
    # ---- A) ponovljivost
    P('\n=== A) PROVJERA PONOVLJIVOSTI (stari detektor, ista vlastita snimka; mora biti identicno logovima iz p43/IEEE)')
    for v in ('base','final'):
        r=OUT['repro'].get(v+'_repro'); P('  %-18s %s'%(VARN[v]+':',('vlastita %d: %s'%(r['video'],r['ishod'])) if r else 'nije pokrenuta'))
    # ---- B) glavni ishod
    kb,kf='11m|ss|base','11m|ss|final'
    if kb in R and kf in R:
        P('\n=== B) GLAVNI ISHOD: YOLO11m, 16 SSMOT snimaka, konacna varijanta prema polaznoj (prosjek po snimci)')
        head(); row('polazna',kb,SS16); row('konacna varijanta',kf,SS16)
        D=delta(kb,kf,SS16,'konacna - polazna (uparene razlike, Wilcoxon, n = 16; 95% bootstrap CI)')
        vr=D['IDSW']['p']<0.05 and D['IDSW']['d']<0 and D['HOTA']['d']>=-0.005
        P('  PRAVILO (unaprijed): IDSW znacajno manji I dHOTA >= -0.005  ->  %s'%('POBOLJSANJE VRIJEDI i s YOLO11m' if vr else
          'poboljsanje NE vrijedi s YOLO11m (IDSW %+.2f, p = %s; dHOTA %+.4f)'%(D['IDSW']['d'],fp(D['IDSW']['p']),D['HOTA']['d'])))
        OUT['glavni']={'delta':D,'vrijedi':bool(vr)}
    # ---- C) isti tracker, stari detektor
    if 'v8|ss|base' in R and 'v8|ss|final' in R:
        P('\n=== C) ISTI TRACKER SA STARIM DETEKTOROM YOLOv8x (postojeci logovi, 15 SSMOT snimaka bez 10) -- za usporedbu')
        head(); row('YOLOv8x polazna (preneseni)','v8|ss|base',SS15); row('YOLOv8x konacna','v8|ss|final',SS15)
        if kb in R and kf in R: row('YOLO11m polazna (iste 15)',kb,SS15); row('YOLO11m konacna (iste 15)',kf,SS15)
        D8=delta('v8|ss|base','v8|ss|final',SS15,'YOLOv8x: konacna - polazna (n = 15)'); OUT['v8_ss']=D8
        if kb in R and kf in R:
            D11=delta(kb,kf,SS15,'YOLO11m: konacna - polazna, iste 15 snimaka')
            P('  razlika dobitaka, (konacna - polazna) s YOLO11m minus isto s YOLOv8x (istrazivacki, n = 15;')
            P('  za IDSW pozitivno = s YOLO11m MANJI dobitak; za HOTA i IDF1 pozitivno = s YOLO11m VECI dobitak):')
            for m in ('IDSW','HOTA','IDF1'):
                dd=[(R[kf][s][m]-R[kb][s][m])-(R['v8|ss|final'][s][m]-R['v8|ss|base'][s][m]) for s in SS15]
                P('     %-5s %+.3f  p = %s'%(m,np.mean(dd),fp(wp(dd))))
            OUT['v8_vs_11m_ss15']={'11m':D11}
    # ---- D) mehanizam: promjena klase kod zamjena
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
    P('\n=== D) MEHANIZAM: zamjene s promjenom klase (TrackEval uparivanje; samoprovjera = broj jednak TrackEval IDSW)')
    P('  imena klasa: YOLO11m %s | YOLOv8x %s'%(N11 or 'nepoznato',N8 or 'nepoznato'))
    OUT['klasa']={}
    for k,seqs,lab,names in ((kb,SS16,'YOLO11m polazna, SSMOT 16',N11),(kf,SS16,'YOLO11m konacna, SSMOT 16',N11),(kb,SS15,'YOLO11m polazna, iste 15',N11),
                             ('v8|ss|base',SS15,'YOLOv8x polazna, SSMOT 15',N8),('v8|ss|final',SS15,'YOLOv8x konacna, SSMOT 15',N8),
                             ('11m|own|base',OWN,'YOLO11m polazna, vlastitih 5',N11),('11m|own|final',OWN,'YOLO11m konacna, vlastitih 5',N11),
                             ('v8|own|base',OWN,'YOLOv8x polazna, vlastitih 5',N8),('v8|own|final',OWN,'YOLOv8x konacna, vlastitih 5',N8)):
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
        nm=lambda i: names[i] if names and 0<=i<len(names) else 'klasa %d'%i
        P('  %-30s zamjena %4d (samoprovjera %s) | s promjenom klase %4d od %4d s poznatom klasom (%3.0f%%) | %s'%(lab,n,'PROSLA' if okc else 'NIJE PROSLA',
          clsch,known,100*clsch/max(known,1),', '.join('%s<->%s %d'%(nm(a),nm(b),c) for (a,b),c in pairs.most_common(3)) or '-'))
        OUT['klasa'][lab]={'n':n,'promjena':clsch,'poznato':known,'samoprovjera':okc,'parovi':{'%d-%d'%ab:c for ab,c in pairs.items()}}
    # ---- E) usporedba s drugim trackerima (YOLO11m)
    refs=[n for n,_ in REF11 if '11m|ss|ref|'+n in R]
    if kf in R and refs:
        P('\n=== E) YOLO11m, 16 SSMOT snimaka: vlastiti tracker prema drugim trackerima (isti detektor, logovi iz p44 / MDPI)')
        head()
        for n in refs: row(n,'11m|ss|ref|'+n,SS16)
        if kb in R: row('vlastiti, polazna',kb,SS16)
        row('vlastiti, konacna varijanta',kf,SS16)
        P('  konacna varijanta minus drugi tracker (Wilcoxon, Holm preko %d usporedbi po metrici):'%len(refs))
        cmp={}
        for m in ('IDSW','HOTA','IDF1','AssA'):
            ds_=[[R[kf][s][m]-R['11m|ss|ref|'+n][s][m] for s in SS16] for n in refs]; ps=[wp(d) for d in ds_]; hs=holm(ps)
            for n,d,p,h in zip(refs,ds_,ps,hs): cmp.setdefault(n,{})[m]=(float(np.mean(d)),p,h,sum(1 for x in d if (x<0 if m=='IDSW' else x>0)))
        P('  %-24s %22s %26s %26s %26s'%('','IDSW (p, Holm, bolje)','HOTA','IDF1','AssA'))
        for n in refs:
            P('  %-24s '%n+' '.join(('%+6.2f' if m=='IDSW' else '%+7.4f')%cmp[n][m][0]+' (%s, %s, %d)'%(fp(cmp[n][m][1]),fp(cmp[n][m][2]),cmp[n][m][3]) for m in ('IDSW','HOTA','IDF1','AssA')))
        OUT['usporedba']=cmp
    # ---- F) vlastitih 5 (sporedno)
    if '11m|own|base' in R and '11m|own|final' in R:
        P('\n=== F) VLASTITIH 5 SNIMAKA (sporedno, n = 5: najmanji moguci p = 0.0625; oznake nepotpune)')
        head(); row('YOLO11m polazna','11m|own|base',OWN); row('YOLO11m konacna','11m|own|final',OWN)
        if 'v8|own|base' in R: row('YOLOv8x polazna (IEEE)','v8|own|base',OWN)
        if 'v8|own|final' in R: row('YOLOv8x konacna (IEEE)','v8|own|final',OWN)
        OUT['own']=delta('11m|own|base','11m|own|final',OWN,'YOLO11m: konacna - polazna (n = 5)')
        for v in ('base','final'):
            if 'v8|own|'+v in R: OUT['own_det_'+v]=delta('v8|own|'+v,'11m|own|'+v,OWN,'%s: YOLO11m - YOLOv8x (n = 5)'%VARN[v])
    OUT['srednje']={k:{m:mean(k,m,sorted(R[k])) for m in MET+['AssRe','MOTA']} for k in R}
    OUT['po_snimkama']={k:{str(s):v for s,v in R[k].items()} for k in R}
    json.dump(OUT,open(O+'/p52_sve.json','w'),indent=1,default=float)
    open(S+'/p52_vlastiti_11m.txt','w').write('\n'.join(L)+'\n'); print('\nrezultat: %s/p52_vlastiti_11m.txt'%S)
if __name__=='__main__':
    if '--vrati' in sys.argv:
        os.makedirs(O,exist_ok=True)
        if alive(LOCK): sys.exit('!! p52 jos radi (PID %s) -- ne vracaj dok radi.'%alive(LOCK))
        vrati(); print('gotovo.')
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
