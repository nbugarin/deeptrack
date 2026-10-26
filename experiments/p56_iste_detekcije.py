#!/usr/bin/env python3
"""p56: USPOREDBA SA SOTA TRACKERIMA NA ISTIM DETEKCIJAMA -- YOLO11m, DeepStream detekcije na 1280x720 (p54 ds_mux), 16 SSMOT snimaka

Plan zapisan PRIJE pokretanja (projekt: claude/p56_iste_detekcije_plan.md).
Svi trackeri dobivaju iste detekcije: BoxMOT trackeri citaju scpilot/p54/det/ds_mux_<n>.json (izlaz nvinfer, config_infer_11m.txt,
streammux 1280x720); vlastiti tracker i NVIDIA trackeri u p55 dobivaju upravo te detekcije unutar DeepStreama.
BoxMOT: ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, Hybrid-SORT, BoostTrack (ImprAssoc) -- koliko ih instalirana verzija ima;
ZADANE postavke biblioteke, model izgleda /workspace/osnet_x1_0_vessel.pt (kao p31), frame_rate = FPS snimke, obvezni parametri bez
zadane vrijednosti = 0.25. Nista se ne podesava. Ne dira track.h, engine-e ni konfiguracije; pise samo u scpilot/p56.
Pokretanje (GPU slobodan, p55 gotov):  nohup python3 /workspace/p56_iste_detekcije.py > /workspace/p56_out.txt 2>&1 &
Pracenje: tail -3 /workspace/p56_out.txt     Samo ocjena: python3 /workspace/p56_iste_detekcije.py --eval
Provjera BoxMOT-a prije pokretanja (par sekundi): python3 /workspace/p56_iste_detekcije.py --provjera
Izlaz: /workspace/scpilot/p56_iste_detekcije.txt (+ scpilot/p56/p56_sve.json); ponovno pokretanje nastavlja (gotovi trackeri se preskacu)
"""
import os, sys, json, time, shutil, importlib, inspect, subprocess
import numpy as np
for a,b in (('float',float),('int',int),('bool',bool)):
    if not hasattr(np,a): setattr(np,a,b)
W=os.environ.get('P56_W','/workspace'); S=W+'/scpilot'; O=S+'/p56'; LOCK=O+'/p56.lock'; TR=O+'/trk'
SS16=[1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]
VID=W+'/SSMOT_video/video/SSMOT_%d.mp4'; GT=W+'/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
DET=S+'/p54/det/ds_mux_%d.json'; REID=W+'/osnet_x1_0_vessel.pt'
CAND=[('ByteTrack',['BYTETracker','ByteTrack']),('BoT-SORT',['BoTSORT','BotSort']),('OC-SORT',['OCSORT','OcSort']),
      ('Deep OC-SORT',['DeepOCSORT','DeepOcSort']),('StrongSORT',['StrongSORT','StrongSort']),('Hybrid-SORT',['HybridSORT','HybridSort']),
      ('BoostTrack',['BoostTrack']),('ImprAssoc',['ImprAssocTrack'])]
OWN={'vlastiti, konacna':W+'/log_p55_final_ss_%d.json','vlastiti, polazna':W+'/log_p55_base_ss_%d.json'}
NV={'NvSORT':W+'/log_p55_nv_nvsort_ss_%d.json','NvDeepSORT':W+'/log_p55_nv_deep_ss_%d.json','NvDCF r1':W+'/log_p55_nv_dcf1_ss_%d.json','NvDCF r2':W+'/log_p55_nv_dcf2_ss_%d.json'}
ULT={'Ultralytics ByteTrack (vlastite detekcije)':S+'/p44/ultra_bytetrack_%d.json','Ultralytics BoT-SORT (vlastite detekcije)':S+'/p44/ultra_botsort_%d.json'}
LOCKS=[S+'/p%d/p%d.lock'%(i,i) for i in (44,45,46,47,48,52,53,54,55)]+[S+'/p43.lock',S+'/p27.lock']
REINSTALL=('pip install --no-deps git+https://github.com/JonathonLuiten/TrackEval.git && python3 -c "import scipy, cv2, trackeval; print(\'SVE OK\')"')
L=[]
def P(x=''): print(x,flush=True); L.append(x)
def json_ok(p):
    try:
        with open(p) as f: d=json.load(f)
        return isinstance(d,list) and len(d)>0
    except Exception: return False
def alive(lock):
    if not os.path.exists(lock): return None
    pid=open(lock).read().strip()
    return pid if pid and os.path.exists('/proc/'+pid) else None
import atexit
def uzmi_lock(ime):
    """atomsko zakljucavanje na samom pocetku -- drugo pokretanje iste skripte odmah odustaje"""
    os.makedirs(os.path.dirname(LOCK),exist_ok=True)
    for _ in range(3):
        try:
            fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
            atexit.register(pusti_lock); return
        except FileExistsError:
            try: pid=open(LOCK).read().strip(); age=time.time()-os.path.getmtime(LOCK)
            except FileNotFoundError: continue
            if (pid and os.path.exists('/proc/'+pid)) or (not pid and age<10):
                sys.exit('!! %s vec radi (PID %s) -- NE pokreci dvaput (provjeri naredbom: jobs).'%(ime,pid or '?'))
            try: os.remove(LOCK)
            except FileNotFoundError: pass
    sys.exit('!! ne mogu uzeti zakljucavanje %s'%LOCK)
def pusti_lock():
    try:
        if open(LOCK).read().strip()==str(os.getpid()): os.remove(LOCK)
    except Exception: pass
def stanje():
    try: return json.load(open(O+'/stanje.json'))
    except Exception: return {}
def spremi(st): json.dump(st,open(O+'/stanje.json','w'),indent=1)
def outp(name,s): return TR+'/%s_%d.json'%(name.replace(' ','_').replace('-',''),s)
# ======================================================================= BoxMOT
PRESKOK={'IndexError':0,'ValueError':0}   # koliko je puta preskocen ponovni izracun Kalmana nakon prekida (unfreeze)
def boxmot_setup():
    for _o,_n in [('ultralytics.yolo','ultralytics'),('ultralytics.yolo.utils','ultralytics.utils'),('ultralytics.yolo.v8','ultralytics.models.yolo'),
                  ('ultralytics.yolo.v8.detect','ultralytics.models.yolo.detect'),('ultralytics.yolo.engine','ultralytics.engine'),('ultralytics.yolo.engine.model','ultralytics.engine.model')]:
        try: sys.modules.setdefault(_o,importlib.import_module(_n))
        except Exception: pass
    import boxmot
    try:   # zakrpa kao p31 (IndexError, OC-SORT) + ValueError: unfreeze u BoxMOT 10.0.83 raspakira 4 broja, a Hybrid-SORT cuva 5 (s pouzdanoscu)
        import boxmot.motion.kalman_filters.xysr_kf as _X
        for _c in list(vars(_X).values()):
            if isinstance(_c,type) and hasattr(_c,'unfreeze'):
                def _safe(self,_u=_c.unfreeze):
                    try: _u(self)
                    except IndexError: PRESKOK['IndexError']+=1
                    except ValueError: PRESKOK['ValueError']+=1
                _c.unfreeze=_safe
    except Exception: pass
    return boxmot
def kwargs_for(cls_,dev,fr):
    from pathlib import Path
    kw={}; used={}
    for n,p in inspect.signature(cls_.__init__).parameters.items():
        if n=='self' or p.kind in (p.VAR_POSITIONAL,p.VAR_KEYWORD): continue
        if n in ('model_weights','reid_weights'): kw[n]=Path(REID)
        elif n=='device': kw[n]=dev
        elif n in ('fp16','half'): kw[n]=False
        elif n=='frame_rate': kw[n]=int(round(fr))
        elif n=='with_reid': kw[n]=True
        elif n=='per_class': kw[n]=False
        elif p.default is inspect.Parameter.empty:
            if n in ('det_thresh','track_thresh','min_conf','conf'): kw[n]=0.25
            else: raise TypeError('obvezni parametar bez zadane vrijednosti: %s'%n)
        else: continue
        used[n]=str(kw[n])
    return kw,used
def runs():
    try:
        import scipy, trackeval, cv2, torch   # noqa: F401
    except Exception as e: sys.exit('!! nedostaje paket (%s). Prvo: %s'%(e,REINSTALL))
    os.makedirs(TR,exist_ok=True)
    uzmi_lock('p56')
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s jos radi (PID %s) -- pokreni p56 kad zavrsi.'%(os.path.basename(lk).split('.')[0],alive(lk)))
    if os.environ.get('P56_NOGPU')!='1':
        try:
            used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
            print('GPU memorija u upotrebi: %d MiB'%used,flush=True)
            if used>1500: sys.exit('GPU je zauzet (>1500 MiB) -- nesto drugo radi na GPU-u. Pricekaj da zavrsi.')
        except (IndexError,ValueError,FileNotFoundError): print('upozorenje: nvidia-smi nije dao podatak, nastavljam',flush=True)
    miss=[p for p in [REID]+[DET%s for s in SS16]+[VID%s for s in SS16] if not os.path.exists(p)]
    if miss: sys.exit('!! nedostaje (p54 mora biti gotov; model izgleda kao u p31):\n   '+'\n   '.join(miss))
    import cv2, torch
    boxmot=boxmot_setup(); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('BoxMOT verzija: %s; uredaj: %s'%(getattr(boxmot,'__version__','?'),dev),flush=True)
    st=stanje(); st.setdefault('postavke',{}); st.setdefault('vrijeme',{}); st.setdefault('nema',[])
    avail=[]
    for name,cands in CAND:
        c=next((getattr(boxmot,x) for x in cands if hasattr(boxmot,x)),None)
        if c is None: print('  %-13s nema u ovoj verziji BoxMOT-a -- preskacem'%name,flush=True); st['nema']=sorted(set(st['nema'])|{name}); continue
        avail.append((name,c))
    spremi(st)
    DE={s:json.load(open(DET%s)) for s in SS16}
    t0=time.time()
    try:
        for name,cls_ in avail:
            todo=[s for s in SS16 if not json_ok(outp(name,s))]
            if not todo: print('  %-13s vec gotov'%name,flush=True); continue
            pal=False
            for s in todo:
                PRESKOK['IndexError']=0; PRESKOK['ValueError']=0
                cap=cv2.VideoCapture(VID%s); fr=cap.get(cv2.CAP_PROP_FPS) or 25.0
                try: kw,used=kwargs_for(cls_,dev,fr); t=cls_(**kw)
                except Exception as e:
                    print('  !! %s se ne moze napraviti (%s: %s) -- preskacem'%(name,type(e).__name__,e),flush=True); st['nema']=sorted(set(st['nema'])|{name}); spremi(st); break
                st['postavke'][name]=used
                byf={int(fd['frame']):fd.get('detections') or [] for fd in DE[s]}; f=0; T=[]; ttr=0.0; nf_det=len(DE[s])
                try:
                    while True:
                        ok,img=cap.read()
                        if not ok: break
                        f+=1; dl=byf.get(f,[])
                        d=np.array([[x['box'][0],x['box'][1],x['box'][0]+x['box'][2],x['box'][1]+x['box'][3],x.get('conf',1.0),x.get('class',0)] for x in dl],float) if dl else np.zeros((0,6))
                        ta=time.time(); o=t.update(d,img); ttr+=time.time()-ta
                        T.append({'frame':f,'detections':[{'id':int(r[4]),'box':[float(r[0]),float(r[1]),float(r[2]-r[0]),float(r[3]-r[1])]} for r in (o if o is not None else [])]})
                except Exception as e:
                    print('  !! %s pao na SSMOT_%d, frejm %d (%s: %s) -- preskacem taj tracker'%(name,s,f,type(e).__name__,e),flush=True)
                    st['nema']=sorted(set(st['nema'])|{name}); st.setdefault('greske',{})[name]='%s: %s'%(type(e).__name__,e); spremi(st); pal=True; break
                if f!=nf_det: print('  napomena: SSMOT_%d -- snimka ima %d frejmova, detekcije %d'%(s,f,nf_det),flush=True)
                json.dump(T,open(outp(name,s)+'.tmp','w')); os.replace(outp(name,s)+'.tmp',outp(name,s))
                st['vrijeme']['%s|%d'%(name,s)]=[f,ttr]; st.setdefault('preskok',{})['%s|%d'%(name,s)]=dict(PRESKOK); spremi(st); del t
                print('  %-13s SSMOT %-2d ok (%.0f s praćenja; ukupno %.0f min)'%(name,s,ttr,(time.time()-t0)/60),flush=True)
            if not pal and all(json_ok(outp(name,s)) for s in SS16):
                st['nema']=[x for x in st['nema'] if x!=name]; st.get('greske',{}).pop(name,None); spremi(st)
    finally:
        pusti_lock()
    print('GPU dio gotov (%.0f min).'%((time.time()-t0)/60),flush=True)
# ======================================================================= ocjena
def ocjena():
    import trackeval
    from scipy.stats import wilcoxon
    os.makedirs(O,exist_ok=True)
    H={}; exec(open(W+'/id_smooth.py').read().split('CFG={')[0].split('"""')[2],H)
    load_tr,load_gt,pick=H['load_tr'],H['load_gt'],H['pick']
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
            res[int(q[1:])]={'HOTA':m('HOTA'),'DetA':m('DetA'),'AssA':m('AssA'),'AssPr':m('AssPr'),'AssRe':m('AssRe'),'LocA':m('LocA'),
                             'IDF1':float(hh['Identity']['IDF1']),'MOTA':float(hh['CLEAR']['MOTA']),'IDSW':float(hh['CLEAR']['IDSW'])}
        C[key]={'sig':sg,'res':{str(s):v for s,v in res.items()}}; json.dump(C,open(O+'/te_cache.json','w')); return res
    st=stanje()
    SETS={}
    for k,pt in list(OWN.items())+list(NV.items()): SETS[k]=('ds',{s:pt%s for s in SS16})
    for name,_ in CAND: SETS[name]=('bm',{s:outp(name,s) for s in SS16})
    for k,pt in ULT.items(): SETS[k]=('ul',{s:pt%s for s in SS16})
    P('=== p56: usporedba na ISTIM detekcijama (DeepStream, YOLO11m, streammux 1280x720, prag 0.25), 16 SSMOT snimaka, TrackEval')
    P('    plan zapisan prije pokretanja: claude/p56_iste_detekcije_plan.md; BoxMOT sa zadanim postavkama, model izgleda osnet_x1_0_vessel.pt')
    if st.get('nema'): P('    nema u instaliranoj verziji BoxMOT-a / ne moze se napraviti: %s'%', '.join(st['nema']))
    R={}; GRP={}
    for k,(g,f) in SETS.items():
        if all(json_ok(p) for p in f.values()): R[k]=ev(k,f); GRP[k]=g; print('   TrackEval %-44s ok'%k,flush=True)
        elif g!='bm' or k not in st.get('nema',[]): P('  (nema svih logova za "%s")'%k)
    def wp(d):
        d=np.array(d,float)
        if not np.any(d!=0): return 1.0
        try: return float(wilcoxon(d).pvalue)
        except ValueError: return float('nan')
    def holm(ps):
        o=sorted(range(len(ps)),key=lambda i:ps[i]); out=[0]*len(ps); run=0
        for r,i in enumerate(o): run=max(run,min(1,ps[i]*(len(ps)-r))); out[i]=run
        return out
    fp=lambda p: '< 0.001' if p<0.001 else '%.3f'%p
    MET=['HOTA','DetA','AssA','AssPr','IDF1','IDSW']
    mean=lambda k,m: float(np.mean([R[k][s][m] for s in SS16]))
    P('\n=== A) SVI TRACKERI (prosjek po snimci)')
    P('  %-46s '%''+'  '.join('%6s'%m for m in MET))
    for g,lab in (('ds','iste detekcije -- DeepStream trackeri (p55)'),('bm','iste detekcije -- BoxMOT, zadane postavke'),('ul','DRUGE detekcije -- Ultralytics end-to-end (p44)')):
        ks=[k for k in R if GRP[k]==g]
        if not ks: continue
        P('  [%s]'%lab)
        for k in sorted(ks,key=lambda k:-mean(k,'HOTA')): P('  %-46s '%k+'  '.join(('%6.1f' if m=='IDSW' else '%6.3f')%mean(k,m) for m in MET))
    OUT={'srednje':{k:{m:mean(k,m) for m in MET+['AssRe','MOTA','LocA']} for k in R},'po_snimkama':{k:{str(s):v for s,v in R[k].items()} for k in R},
         'postavke':st.get('postavke',{}),'nema':st.get('nema',[])}
    me='vlastiti, konacna'
    refs=[k for k in R if GRP[k] in ('bm','ds') and not k.startswith('vlastiti')]
    if me in R and refs:
        P('\n=== B) VLASTITI (konacna) MINUS SVAKI TRACKER NA ISTIM DETEKCIJAMA (Wilcoxon, n = 16; H = Holm preko %d usporedbi po metrici)'%len(refs))
        mets=['HOTA','IDF1','AssA','DetA','IDSW']; ps={m:[wp([R[me][s][m]-R[k][s][m] for s in SS16]) for k in refs] for m in mets}; hs={m:holm(ps[m]) for m in mets}
        P('  %-26s %s'%('',' | '.join('%-30s'%m for m in mets)))
        OUT['usporedba']={}
        for i,k in enumerate(refs):
            cells=[]
            for m in mets:
                d=[R[me][s][m]-R[k][s][m] for s in SS16]; b=sum(1 for x in d if (x<0 if m=='IDSW' else x>0))
                cells.append((('%+6.2f' if m=='IDSW' else '%+.4f')%np.mean(d)+' p=%s H=%s %d/16'%(fp(ps[m][i]),fp(hs[m][i]),b)).ljust(30))
                OUT['usporedba'].setdefault(k,{})[m]={'d':float(np.mean(d)),'p':ps[m][i],'holm':hs[m][i],'bolje':b}
            P('  %-26s %s'%(k,' | '.join(cells)))
        for m in mets:
            better=[k for i,k in enumerate(refs) if hs[m][i]<0.05 and (np.mean([R[me][s][m]-R[k][s][m] for s in SS16])<0 if m=='IDSW' else np.mean([R[me][s][m]-R[k][s][m] for s in SS16])>0)]
            worse=[k for i,k in enumerate(refs) if hs[m][i]<0.05 and k not in better]
            P('  %-5s znacajno BOLJI (Holm < 0.05) od: %s | znacajno LOSIJI od: %s'%(m,', '.join(better) or 'nijednog',', '.join(worse) or 'nijednog'))
        rank={m:sorted(refs+[me],key=lambda k:mean(k,m),reverse=(m!='IDSW')) for m in mets}
        P('  mjesto vlastitog (od %d trackera na istim detekcijama): '%(len(refs)+1)+', '.join('%s %d.'%(m,rank[m].index(me)+1) for m in mets))
        OUT['mjesto']={m:rank[m].index(me)+1 for m in mets}
    tm=st.get('vrijeme',{})
    if tm:
        P('\n=== C) BRZINA BoxMOT trackera (samo pracenje, bez detekcije; frejmova / sekundi pracenja)')
        for name,_ in CAND:
            pts=[tm['%s|%d'%(name,s)] for s in SS16 if '%s|%d'%(name,s) in tm]
            if pts: P('  %-13s %6.1f FPS'%(name,sum(p[0] for p in pts)/max(sum(p[1] for p in pts),1e-9)))
    pk=st.get('preskok',{})
    if pk:
        P('\n=== E) ZAKRPA KALMANOVA FILTRA (unfreeze u BoxMOT 10.0.83): koliko je puta preskocen ponovni izracun nakon prekida, zbroj preko snimaka')
        for name,_ in CAND:
            v=[pk['%s|%d'%(name,s)] for s in SS16 if '%s|%d'%(name,s) in pk]
            if v: P('  %-13s IndexError %d, ValueError %d  (zabiljezeno za %d od 16 snimaka)'%(name,sum(x['IndexError'] for x in v),sum(x['ValueError'] for x in v),len(v)))
    if st.get('postavke'):
        P('\n=== D) POSTAVKE PREDANE KONSTRUKTORIMA (ostalo = zadano u biblioteci)')
        for k,v in st['postavke'].items(): P('  %-13s %s'%(k,', '.join('%s=%s'%(a,b) for a,b in v.items())))
    json.dump(OUT,open(O+'/p56_sve.json','w'),indent=1,default=float)
    open(S+'/p56_iste_detekcije.txt','w').write('\n'.join(L)+'\n'); print('\nrezultat: %s/p56_iste_detekcije.txt'%S)
def provjera():
    """brza provjera prije dugog pokretanja: koji trackeri postoje, sto im se predaje i rade li na umjetnom frejmu"""
    import cv2, torch   # noqa: F401
    boxmot=boxmot_setup(); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('BoxMOT verzija: %s; uredaj: %s; model izgleda %s: %s'%(getattr(boxmot,'__version__','?'),dev,REID,'POSTOJI' if os.path.exists(REID) else '!! NEMA'),flush=True)
    img=np.zeros((720,1280,3),np.uint8); img[90:160,90:210]=200; img[290:390,390:530]=120
    d0=np.array([[100,100,200,150,0.9,0],[400,300,520,380,0.8,1]],float)
    for name,cands in CAND:
        c=next((getattr(boxmot,x) for x in cands if hasattr(boxmot,x)),None)
        if c is None: print('  %-13s nema u ovoj verziji'%name,flush=True); continue
        try:
            kw,used=kwargs_for(c,dev,25.0); t=c(**kw); o=None
            for i in range(5): o=t.update(d0+np.array([2*i,0,2*i,0,0,0]),img)
            print('  %-13s OK   predano: %s | izlaz zadnjeg frejma: %s'%(name,', '.join('%s=%s'%(k,v) for k,v in used.items()) or 'nista (sve zadano)',
                  None if o is None else np.asarray(o).shape),flush=True)
        except Exception as e:
            print('  %-13s GRESKA %s: %s'%(name,type(e).__name__,str(e)[:300]),flush=True)
def dijagnoza(name,s):
    """pokrene jedan tracker na jednoj snimci i ispise cijeli traceback na prvoj gresci (za popravak)"""
    import cv2, torch, traceback
    boxmot=boxmot_setup(); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cands=dict(CAND)[name]; c=next((getattr(boxmot,x) for x in cands if hasattr(boxmot,x)),None)
    cap=cv2.VideoCapture(VID%s); kw,used=kwargs_for(c,dev,cap.get(cv2.CAP_PROP_FPS) or 25.0); t=c(**kw)
    byf={int(fd['frame']):fd.get('detections') or [] for fd in json.load(open(DET%s))}; f=0
    while True:
        ok,img=cap.read()
        if not ok: print('prosao cijelu snimku bez greske (%d frejmova)'%f); return
        f+=1; dl=byf.get(f,[])
        d=np.array([[x['box'][0],x['box'][1],x['box'][0]+x['box'][2],x['box'][1]+x['box'][3],x.get('conf',1.0),x.get('class',0)] for x in dl],float) if dl else np.zeros((0,6))
        try: t.update(d,img)
        except Exception:
            print('GRESKA na frejmu %d, detekcija %d, oblik %s:'%(f,len(d),d.shape)); print(np.round(d,2)); print(traceback.format_exc()); return
if __name__=='__main__':
    if '--dijagnoza' in sys.argv: dijagnoza('Hybrid-SORT',2)
    elif '--provjera' in sys.argv: provjera()
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
