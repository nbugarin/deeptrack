#!/usr/bin/env python3
"""p54: ZASTO DeepStream S ISTIM YOLO11m TEZINAMA DAJE NIZI DetA OD Ultralyticsa? -- CISTA USPOREDBA DETEKCIJA (bez trackera)

Plan zapisan PRIJE pokretanja (projekt: claude/p54_detekcije_plan.md). Dijagnostika; tracker se ne dira, nista se ne bira na test skupu.
Detekcije (16 SSMOT snimaka kao MDPI), prag pouzdanosti 0.25 svugdje; svaka detekcija dobiva jedinstveni ID, ocjenjuju se samo
DetA, DetRe, DetPr, LocA (TrackEval HOTA podmjere) i broj detekcija:
  DeepStream (nova skripta probe_det.py = ista cijev kao probe_to_json.py, ali BEZ nvtrackera; okviri izravno iz nvinfer):
    ds_ref    streammux 1920x1080, KOPIJA config_infer_11m.txt bez promjene (= ulaz trackerima u p44/p52)
    ds_nms07  isto, nms-iou-threshold 0.7 (Ultralytics zadano) umjesto 0.45
    ds_bil    isto, scaling-filter=1 (bilinearno skaliranje u nvinfer)
    ds_mux    streammux na izvornoj rezoluciji snimke (samo ako snimke nisu 1920x1080)
    ds_sve    sve primjenjive promjene zajedno
  Ultralytics (best.pt YOLO11m, imgsz iz args.yaml, conf 0.25):
    ul_iou07  NMS IoU 0.7 (zadano, FP32)       ul_iou045  NMS IoU 0.45 (kao DeepStream)       ul_half  IoU 0.7, FP16
Konfiguracije su KOPIJE u scpilot/p54/cfg/ -- config_infer_11m.txt i config_infer_boat.txt se NE mijenjaju. DeepStream se pokrece
iz zasebne radne mape (scpilot/p54/rad), a engine-i dobivaju sigurnosnu kopiju (.p54bak) i provjeru nakon svakog pokretanja.
Pokretanje (GPU slobodan, p53 gotov):  nohup python3 /workspace/p54_detekcije.py > /workspace/p54_out.txt 2>&1 &
Pracenje: tail -3 /workspace/p54_out.txt      Samo ocjena: python3 /workspace/p54_detekcije.py --eval
Ako je prekinuto: python3 /workspace/p54_detekcije.py --vrati  (pa ponovno pokreni -- nastavlja)
Izlaz: /workspace/scpilot/p54_detekcije.txt (+ scpilot/p54/p54_sve.json)
"""
import os, re, sys, json, time, shutil, hashlib, subprocess
import numpy as np
for a,b in (('float',float),('int',int),('bool',bool)):
    if not hasattr(np,a): setattr(np,a,b)
W=os.environ.get('P54_W','/workspace'); S=W+'/scpilot'; O=S+'/p54'; LOCK=O+'/p54.lock'; DD=O+'/det'; CF=O+'/cfg'; RAD=O+'/rad'
INFER=W+'/DeepStream-Yolo/config_infer_11m.txt'
WD=W+'/det_glavni/B_yolo11m/weights'; PT=WD+'/best.pt'; ENG11=WD+'/best_11m_b1_gpu0_fp16.engine'; ARGS=os.path.dirname(WD)+'/args.yaml'
PROT0=[W+'/model_b1_gpu0_fp16.engine',W+'/best_0904.engine',W+'/osnet_vessel.engine',ENG11]
SS16=[1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]
VID=W+'/SSMOT_video/video/SSMOT_%d.mp4'; GT=W+'/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
PROBE=O+'/probe_det.py'
LOCKS=[S+'/p43.lock',S+'/p27.lock']+[S+'/p%d/p%d.lock'%(i,i) for i in (44,45,46,47,48,52,53)]
T_RUN=900
TRK={'NvSORT (p44)':W+'/log_ss11m_nvsort_%d.json','NvDCF r1 (p44)':W+'/log_ss11m_dcf1_%d.json',
     'vlastiti, konacna (p52)':W+'/log_p52_final_ss_%d.json','Ultralytics ByteTrack (p44)':S+'/p44/ultra_bytetrack_%d.json'}
REINSTALL=('apt-get update && apt-get install -y libeigen3-dev && pip install --no-deps git+https://github.com/JonathonLuiten/TrackEval.git '
           '&& python3 -c "import scipy, cv2, trackeval; print(\'SVE OK\')"')
PROBE_SRC=r'''import sys, json, gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
Gst.init(None)
VIDEO, OUT, INFER, MW, MH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
log = []; n = [0]; uid = [0]
def probe_fn(pad, info, u):
    buf = info.get_buffer()
    if not buf: return Gst.PadProbeReturn.OK
    bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not bm: return Gst.PadProbeReturn.OK
    lf = bm.frame_meta_list
    while lf:
        try: fm = pyds.NvDsFrameMeta.cast(lf.data)
        except StopIteration: break
        n[0] += 1; dets = []; lo = fm.obj_meta_list
        while lo:
            try: o = pyds.NvDsObjectMeta.cast(lo.data)
            except StopIteration: break
            uid[0] += 1; r = o.rect_params
            dets.append({'id': uid[0], 'class': int(o.class_id), 'box': [float(r.left), float(r.top), float(r.width), float(r.height)], 'conf': float(o.confidence)})
            try: lo = lo.next
            except StopIteration: break
        log.append({'frame': n[0], 'detections': dets})
        try: lf = lf.next
        except StopIteration: break
    return Gst.PadProbeReturn.OK
pl = Gst.Pipeline()
src = Gst.ElementFactory.make("filesrc", "source"); dec = Gst.ElementFactory.make("decodebin", "decoder")
mux = Gst.ElementFactory.make("nvstreammux", "mux"); pgie = Gst.ElementFactory.make("nvinfer", "pgie"); sink = Gst.ElementFactory.make("fakesink", "sink")
src.set_property('location', VIDEO)
mux.set_property('width', MW); mux.set_property('height', MH); mux.set_property('batch-size', 1); mux.set_property('batched-push-timeout', 4000000)
pgie.set_property('config-file-path', INFER); sink.set_property('sync', False)
for e in (src, dec, mux, pgie, sink): pl.add(e)
src.link(dec)
def on_pad(d, pad, m):
    caps = pad.get_current_caps() or pad.query_caps(None)
    if caps and not caps.get_structure(0).get_name().startswith('video'): return
    sp = m.get_request_pad("sink_0")
    if not sp.is_linked(): pad.link(sp)
dec.connect("pad-added", on_pad, mux)
mux.link(pgie); pgie.link(sink)
pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe_fn, 0)
loop = GLib.MainLoop(); ok = [False]
def bus_call(bus, msg, lp):
    if msg.type == Gst.MessageType.EOS:
        json.dump(log, open(OUT, 'w')); ok[0] = True; print("frejmova: %d, detekcija: %d" % (n[0], uid[0])); lp.quit()
    elif msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error(); print("ERROR: %s | %s" % (err, dbg)); lp.quit()
    return True
bus = pl.get_bus(); bus.add_signal_watch(); bus.connect("message", bus_call, loop)
pl.set_state(Gst.State.PLAYING)
try: loop.run()
except Exception: pass
pl.set_state(Gst.State.NULL)
sys.exit(0 if ok[0] else 1)
'''
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
def stanje():
    try: return json.load(open(O+'/stanje.json'))
    except Exception: return {}
def spremi(st): json.dump(st,open(O+'/stanje.json','w'),indent=1)
def vres(p):
    import cv2; c=cv2.VideoCapture(p); return int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(c.get(cv2.CAP_PROP_FRAME_COUNT))
def imgsz():
    try:
        for l in open(ARGS):
            if l.startswith('imgsz:'): return int(l.split(':',1)[1].strip())
    except Exception: pass
    return 640
def cfg_copy(nms=None,bil=False):
    txt=open(INFER).read(); out=[]; sec=None; done_bil=False
    for line in txt.splitlines():
        m=re.match(r'^\s*\[([^\]]+)\]',line)
        if m:
            if sec=='property' and bil and not done_bil: out.append('scaling-filter=1'); done_bil=True
            sec=m.group(1).strip()
        k=line.split('=')[0].strip()
        if nms is not None and sec and sec.startswith('class-attrs') and k=='nms-iou-threshold': line='nms-iou-threshold=%g'%nms
        if bil and k=='scaling-filter': line='scaling-filter=1'; done_bil=True
        out.append(line)
    if bil and not done_bil:
        if sec!='property': out.append('[property]')
        out.append('scaling-filter=1')
    res='\n'.join(out)+'\n'
    if nms is not None: assert re.search(r'^nms-iou-threshold=%g$'%nms,res,re.M),'nms nije postavljen'
    if bil: assert re.search(r'^scaling-filter=1$',res,re.M),'scaling-filter nije postavljen'
    return res
def variants(native):
    V=[('ds_ref',dict(mux=(1920,1080),nms=None,bil=False),'DeepStream, kao u p44/p52 (streammux 1920x1080, NMS 0.45)'),
       ('ds_nms07',dict(mux=(1920,1080),nms=0.7,bil=False),'DeepStream, NMS IoU 0.7'),
       ('ds_bil',dict(mux=(1920,1080),nms=None,bil=True),'DeepStream, bilinearno skaliranje (scaling-filter=1)')]
    if native!=(1920,1080): V.append(('ds_mux',dict(mux=native,nms=None,bil=False),'DeepStream, streammux %dx%d (izvorno)'%native))
    V.append(('ds_sve',dict(mux=native,nms=0.7,bil=True),'DeepStream, sve promjene zajedno'))
    U=[('ul_iou07',dict(iou=0.7,half=False),'Ultralytics, NMS IoU 0.7 (zadano), FP32'),
       ('ul_iou045',dict(iou=0.45,half=False),'Ultralytics, NMS IoU 0.45 (kao DeepStream)'),
       ('ul_half',dict(iou=0.7,half=True),'Ultralytics, NMS IoU 0.7, FP16')]
    return V,U
def outp(v,s): return DD+'/%s_%d.json'%(v,s)
def vrati():
    for p in PROT0:
        b=p+'.p54bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p)==sha(b): os.remove(b); print('   %s: netaknut, kopija obrisana'%os.path.basename(p),flush=True)
        else:
            if os.path.exists(p): os.replace(p,p+'.p54_nepoznat')
            os.replace(b,p); print('   %s: VRACEN iz sigurnosne kopije'%os.path.basename(p),flush=True)
# ======================================================================= GPU
def runs():
    try:
        import scipy, trackeval, cv2   # noqa: F401
    except Exception as e: sys.exit('!! nedostaje paket (%s). Prvo pokreni:\n   %s'%(e,REINSTALL))
    for d in (DD,CF,RAD): os.makedirs(d,exist_ok=True)
    if alive(LOCK): sys.exit('!! p54 vec radi (PID %s).'%alive(LOCK))
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s jos radi (PID %s) -- pokreni p54 kad zavrsi.'%(os.path.basename(lk).split('.')[0],alive(lk)))
    if os.environ.get('P54_NOGPU')!='1':
        try:
            used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
            print('GPU memorija u upotrebi: %d MiB'%used,flush=True)
            if used>1500: sys.exit('GPU je zauzet (>1500 MiB) -- nesto drugo radi na GPU-u. Pricekaj da zavrsi.')
        except (IndexError,ValueError,FileNotFoundError): print('upozorenje: nvidia-smi nije dao podatak, nastavljam',flush=True)
    if any(os.path.exists(p+'.p54bak') for p in PROT0): print('prethodno pokretanje je prekinuto -- provjeravam engine-e:',flush=True); vrati()
    need=[INFER,ENG11,PT,W+'/id_smooth.py']+[VID%s for s in SS16]+[GT%s for s in SS16]
    miss=[p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! nedostaje:\n   '+'\n   '.join(miss))
    if shutil.disk_usage(W).free/2**30<3: sys.exit('!! na disku je slobodno manje od 3 GB')
    st=stanje(); h=sha(INFER)
    if 'infer_sha' in st and st['infer_sha']!=h: sys.exit('!! config_infer_11m.txt promijenjen od prvog pokretanja p54 -- javi Claudeu')
    st['infer_sha']=h; st.setdefault('vrijeme',{})
    res={s:vres(VID%s) for s in SS16}; nat=sorted({r[:2] for r in res.values()})
    print('rezolucija snimaka: %s'%', '.join('%dx%d (%d snimaka)'%(w,hh,sum(1 for r in res.values() if r[:2]==(w,hh))) for w,hh in nat),flush=True)
    if len(nat)!=1: sys.exit('!! snimke nemaju istu rezoluciju (%s) -- javi Claudeu'%nat)
    native=nat[0]; st['rezolucija']=list(native); spremi(st)
    V,U=variants(native)
    for v,c,_ in V:
        open(CF+'/%s.txt'%v,'w').write(cfg_copy(c['nms'],c['bil']))
    norm=lambda x: [l.rstrip() for l in x.strip().splitlines()]
    if norm(open(CF+'/ds_ref.txt').read())!=norm(open(INFER).read()): sys.exit('!! kopija ds_ref se razlikuje od config_infer_11m.txt -- javi Claudeu')
    if os.environ.get('P54_PROBE_MOCK'): shutil.copy(os.environ['P54_PROBE_MOCK'],PROBE)
    else: open(PROBE,'w').write(PROBE_SRC)
    todo=[(v,s) for v,_,_ in V+U for s in SS16 if not json_ok(outp(v,s))]
    print('pokretanja preostalo: %d (%d DeepStream + %d Ultralytics varijanti x 16 snimaka)'%(len(todo),len(V),len(U)),flush=True)
    PROT=[p for p in PROT0 if os.path.exists(p)]
    open(LOCK,'w').write(str(os.getpid())); sh={}; t0=time.time(); done=[0]
    try:
        for p in PROT:
            sh[p]=sha(p); shutil.copy2(p,p+'.p54bak')
            if sha(p+'.p54bak')!=sh[p]: sys.exit('!! sigurnosna kopija %s nije ispravna -- STANI'%p)
        def tick(lab,s):
            done[0]+=1; el=time.time()-t0
            print('  [%3d/%d] %-48s SSMOT %-2d ok   (jos oko %.0f min)'%(done[0],len(todo),lab,s,el/done[0]*(len(todo)-done[0])/60),flush=True)
        for v,c,lab in V:
            for s in [s for vv,s in todo if vv==v]:
                ok=False
                for att in (1,2):
                    st_={p:stat(p) for p in PROT}; out=outp(v,s); ta=time.time()
                    try:
                        r=subprocess.run(['python3',PROBE,VID%s,out,CF+'/%s.txt'%v,str(c['mux'][0]),str(c['mux'][1])],cwd=RAD,capture_output=True,text=True,timeout=T_RUN)
                        err=(r.stdout or '')[-2500:]+'\n----\n'+(r.stderr or '')[-3500:]
                    except subprocess.TimeoutExpired: err='prekoraceno vrijeme'
                    ch=[os.path.basename(p) for p in PROT if stat(p)!=st_[p]]
                    if ch:
                        if os.path.exists(out): os.remove(out)
                        open(O+'/greska.txt','w').write(err)
                        sys.exit('!! promijenio se %s tijekom pokretanja (%s, SSMOT_%d) -- vraca se iz kopije; posalji Claudeu: tail -40 /workspace/scpilot/p54/greska.txt'%(', '.join(ch),v,s))
                    if json_ok(out):
                        st['vrijeme']['%s|%d'%(v,s)]=[len(json.load(open(out))),time.time()-ta]; spremi(st); ok=True; break
                    if os.path.exists(out): os.remove(out)
                    open(O+'/greska.txt','w').write('%s SSMOT_%d\n'%(v,s)+err)
                if not ok: sys.exit('!! %s, SSMOT %d nije uspjelo ni u drugom pokusaju -- posalji Claudeu: tail -40 /workspace/scpilot/p54/greska.txt'%(lab,s))
                tick(lab,s)
        if any(vv.startswith('ul_') for vv,_ in todo):
            try: from ultralytics import YOLO
            except Exception as e: sys.exit('!! Ultralytics se ne moze ucitati (%s) -- DeepStream dio je spremljen; javi Claudeu'%e)
            isz=imgsz(); print('Ultralytics: imgsz %d'%isz,flush=True)
            for v,c,lab in U:
                for s in [s for vv,s in todo if vv==v]:
                    m=YOLO(PT); T=[]; uid=0; ta=time.time()
                    for f,rr in enumerate(m.predict(source=VID%s,conf=0.25,iou=c['iou'],half=c['half'],imgsz=isz,stream=True,verbose=False),1):
                        dets=[]
                        if rr.boxes is not None and len(rr.boxes):
                            xy=rr.boxes.xyxy.cpu().numpy(); cf=rr.boxes.conf.cpu().numpy(); cl=rr.boxes.cls.cpu().numpy().astype(int)
                            for b,q,k in zip(xy,cf,cl):
                                uid+=1; dets.append({'id':uid,'class':int(k),'box':[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],'conf':float(q)})
                        T.append({'frame':f,'detections':dets})
                    json.dump(T,open(outp(v,s)+'.tmp','w')); os.replace(outp(v,s)+'.tmp',outp(v,s)); del m
                    st['vrijeme']['%s|%d'%(v,s)]=[len(T),time.time()-ta]; spremi(st); tick(lab,s)
    finally:
        for p,hh in sh.items():
            b=p+'.p54bak'
            if os.path.exists(p) and sha(p)==hh:
                if os.path.exists(b): os.remove(b)
                print('   provjera na kraju: %s isti kao na pocetku (SHA256)'%os.path.basename(p),flush=True)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p,p+'.p54_nepoznat')
                os.replace(b,p); print('   !! %s se promijenio -- VRACEN iz sigurnosne kopije; javi Claudeu'%os.path.basename(p),flush=True)
        if os.path.exists(LOCK): os.remove(LOCK)
    print('GPU dio gotov (%.0f min).'%((time.time()-t0)/60),flush=True)
# ======================================================================= ocjena
def ocjena():
    import trackeval
    from scipy.stats import wilcoxon
    H={}; exec(open(W+'/id_smooth.py').read().split('CFG={')[0].split('"""')[2],H)
    load_tr,load_gt,pick=H['load_tr'],H['load_gt'],H['pick']
    ec=trackeval.Evaluator.get_default_eval_config()
    ec.update(PRINT_RESULTS=False,PRINT_CONFIG=False,OUTPUT_SUMMARY=False,OUTPUT_DETAILED=False,PLOT_CURVES=False,TIME_PROGRESS=False,USE_PARALLEL=False,DISPLAY_LESS_PROGRESS=True)
    try: C=json.load(open(O+'/te_cache.json'))
    except Exception: C={}
    rows=lambda d: sorted((int(f),int(o['id']),*map(float,o['box'][:4])) for f,Lst in d.items() for o in Lst)
    SC={}
    def ev(key,files):
        sg=[[os.path.basename(p),os.path.getsize(p),int(os.path.getmtime(p))] for _,p in sorted(files.items())]
        if key in C and C[key].get('sig')==sg: return {int(s):v for s,v in C[key]['res'].items()}
        base=O+'/te_tmp'; shutil.rmtree(base,ignore_errors=True); si={}; ndet={}
        for s,p in files.items():
            tr=load_tr(p); sc=pick(GT%s,tr); SC.setdefault(key,{})[s]=sc; g=load_gt(GT%s,sc)
            G=rows(g); R_=rows(tr); seq='S%02d'%s; si[seq]=max([r[0] for r in G+R_]+[1]); ndet[s]=len(R_)/max(len(json.load(open(p))),1)
            os.makedirs('%s/gt/%s/gt'%(base,seq),exist_ok=True); os.makedirs(base+'/tr/t/data',exist_ok=True)
            with open('%s/gt/%s/gt/gt.txt'%(base,seq),'w') as f:
                for x in G: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n'%x)
            with open('%s/tr/t/data/%s.txt'%(base,seq),'w') as f:
                for x in R_: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n'%x)
        dc=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(GT_FOLDER=base+'/gt',TRACKERS_FOLDER=base+'/tr',BENCHMARK='b',SPLIT_TO_EVAL='',SEQ_INFO=si,TRACKERS_TO_EVAL=['t'],DO_PREPROC=False,
                  SKIP_SPLIT_FOL=True,GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt',TRACKER_SUB_FOLDER='data',PRINT_CONFIG=False)
        out,_=trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)],[trackeval.metrics.HOTA({'PRINT_CONFIG':False})])
        Rt=out['MotChallenge2DBox']['t']; shutil.rmtree(base,ignore_errors=True); res={}
        for q in si:
            hh=Rt[q]['pedestrian']['HOTA']; s=int(q[1:])
            res[s]={k:float(np.mean(hh[k])) for k in ('HOTA','DetA','DetRe','DetPr','LocA','AssA')}; res[s]['det_po_frejmu']=ndet[s]
        C[key]={'sig':sg,'res':{str(s):v for s,v in res.items()}}; json.dump(C,open(O+'/te_cache.json','w')); return res
    st=stanje(); native=tuple(st.get('rezolucija',(1920,1080))); V,U=variants(native)
    P('=== p54: cista usporedba detekcija YOLO11m (bez trackera), 16 SSMOT snimaka, prag 0.25; plan zapisan prije pokretanja (claude/p54_detekcije_plan.md)')
    P('    izvorna rezolucija snimaka: %dx%d; svaka detekcija ima jedinstveni ID -> vrijede samo DetA, DetRe, DetPr, LocA'%native)
    R={}; NAME={}
    for v,_,lab in V+U:
        f={s:outp(v,s) for s in SS16}
        if all(json_ok(p) for p in f.values()): R[v]=ev(v,f); NAME[v]=lab; print('   TrackEval %-10s ok'%v,flush=True)
        else: P('  (nema svih detekcija za %s)'%v)
    for lab,pt in TRK.items():
        f={s:pt%s for s in SS16}
        if all(json_ok(p) for p in f.values()): R[lab]=ev('trk|'+lab,f); NAME[lab]=lab+' -- izlaz trackera'
    def wp(d):
        d=np.array(d,float)
        if not np.any(d!=0): return 1.0
        try: return float(wilcoxon(d).pvalue)
        except ValueError: return float('nan')
    fp=lambda p: '< 0.001' if p<0.001 else '%.3f'%p
    MET=['DetA','DetRe','DetPr','LocA','det_po_frejmu']
    mean=lambda k,m: float(np.mean([R[k][s][m] for s in SS16]))
    P('\n=== A) PROSJEK PO SNIMCI')
    P('  %-62s '%''+'  '.join('%7s'%m[:7] for m in MET))
    for k in R: P('  %-62s '%NAME[k]+'  '.join('%7.3f'%mean(k,m) for m in MET))
    OUT={'srednje':{k:{m:mean(k,m) for m in MET} for k in R},'po_snimkama':{k:{str(s):v for s,v in R[k].items()} for k in R},'mjerilo_gt':{k:{str(s):v for s,v in d.items()} for k,d in SC.items()}}
    if 'ds_ref' in R and 'ul_iou07' in R:
        gap=mean('ul_iou07','DetA')-mean('ds_ref','DetA')
        P('\n=== B) RAZLIKE PREMA DeepStream ds_ref (uparene, Wilcoxon, n = 16) i udio zatvorene razlike do Ultralyticsa (ul_iou07)')
        P('  razlika u DetA Ultralytics - DeepStream (ds_ref): %+.4f'%gap)
        OUT['razlike']={}
        for k in [x for x in R if x!='ds_ref' and x not in TRK]:
            cells=[]; e={}
            for m in ('DetA','DetRe','DetPr','LocA'):
                d=[R[k][s][m]-R['ds_ref'][s][m] for s in SS16]; p=wp(d); e[m]={'d':float(np.mean(d)),'p':p,'bolje':int(sum(1 for x in d if x>0))}
                cells.append('%s %+.4f (p %s, %d/16)'%(m,np.mean(d),fp(p),e[m]['bolje']))
            zat=(e['DetA']['d']/gap*100) if abs(gap)>1e-9 else float('nan')
            P('  %-44s %s | zatvoreno %4.0f%%'%(k,' | '.join(cells),zat)); e['zatvoreno_pct']=zat; OUT['razlike'][k]=e
    trk=[k for k in R if k in TRK]
    if trk and 'ds_ref' in R:
        P('\n=== C) OKVIRI IZ TRACKERA PREMA IZVORNIM DETEKCIJAMA (ds_ref): koliko LocA/DetA mijenja sam tracker (opisno)')
        for k in trk:
            P('  %-34s '%k+' | '.join('%s %+.4f (p %s)'%(m,np.mean([R[k][s][m]-R['ds_ref'][s][m] for s in SS16]),fp(wp([R[k][s][m]-R['ds_ref'][s][m] for s in SS16]))) for m in ('DetA','DetRe','DetPr','LocA')))
    tm=st.get('vrijeme',{})
    if tm:
        P('\n=== D) BRZINA DETEKCIJE (cijela snimka, ukljucuje pokretanje; DeepStream bez trackera, Ultralytics predict)')
        for v,_,lab in V+U:
            pts=[tm['%s|%d'%(v,s)] for s in SS16 if '%s|%d'%(v,s) in tm]
            if len(pts)>=3:
                fr=np.array([p[0] for p in pts],float); sec=np.array([p[1] for p in pts],float); sl,ic=np.polyfit(fr,sec,1)
                P('  %-62s %6.1f FPS s pokretanjem | iz nagiba %6.1f FPS, pokretanje oko %3.0f s'%(lab,fr.sum()/sec.sum(),1/sl if sl>0 else float('nan'),ic))
    json.dump(OUT,open(O+'/p54_sve.json','w'),indent=1,default=float)
    open(S+'/p54_detekcije.txt','w').write('\n'.join(L)+'\n'); print('\nrezultat: %s/p54_detekcije.txt'%S)
if __name__=='__main__':
    if '--vrati' in sys.argv:
        os.makedirs(O,exist_ok=True)
        if alive(LOCK): sys.exit('!! p54 jos radi (PID %s).'%alive(LOCK))
        vrati(); print('gotovo.')
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
