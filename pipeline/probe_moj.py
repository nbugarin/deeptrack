import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import json
import numpy as np

Gst.init(None)

frame_count = 0
tracking_log = []
prev_locs = {}
original_switches = 0

def probe_fn(pad, info, u_data):
    global frame_count, original_switches, prev_locs
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    frame_count += 1
    current_locs = {}
    frame_detections = []

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_id = int(obj.object_id)
            cls = int(obj.class_id)
            left = float(obj.rect_params.left)
            top = float(obj.rect_params.top)
            width = float(obj.rect_params.width)
            height = float(obj.rect_params.height)
            conf = float(obj.confidence)

            loc = (int((left+width/2)//100), int((top+height/2)//100))
            if loc in prev_locs and prev_locs[loc] != obj_id:
                original_switches += 1

            current_locs[loc] = obj_id
            frame_detections.append({
                'id': obj_id,
                'class': cls,
                'box': [left, top, width, height],
                'conf': conf
            })

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        tracking_log.append({
            'frame': frame_count,
            'detections': frame_detections
        })
        prev_locs = current_locs

        if frame_count % 100 == 0:
            print("Frame=" + str(frame_count) +
                  " Switches=" + str(original_switches))

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# Pipeline
VIDEO = sys.argv[1]
OUTPUT_JSON = sys.argv[3]

pipeline = Gst.Pipeline()
source = Gst.ElementFactory.make("filesrc", "source")
decoder = Gst.ElementFactory.make("decodebin", "decoder")
streammux = Gst.ElementFactory.make("nvstreammux", "mux")
pgie = Gst.ElementFactory.make("nvinfer", "pgie")
tracker = Gst.ElementFactory.make("nvtracker", "tracker")
osd = Gst.ElementFactory.make("nvdsosd", "osd")
fakesink = Gst.ElementFactory.make("fakesink", "sink")

source.set_property('location', VIDEO)
streammux.set_property('width', 1920)
streammux.set_property('height', 1080)
streammux.set_property('batch-size', 1)
streammux.set_property('batched-push-timeout', 4000000)
pgie.set_property('config-file-path',
    (sys.argv[4] if len(sys.argv)>4 else '/workspace/DeepStream-Yolo/config_infer_boat.txt'))
tracker.set_property('ll-lib-file',
    '/workspace/mytracker/libmoj.so')
tracker.set_property('ll-config-file',
    sys.argv[2])
tracker.set_property('tracker-width', 640)
tracker.set_property('tracker-height', 384)
fakesink.set_property('sync', False)

for e in [source, decoder, streammux, pgie, tracker, osd, fakesink]:
    pipeline.add(e)

source.link(decoder)

def on_pad_added(dec, pad, mux):
    sinkpad = mux.get_request_pad("sink_0")
    if not sinkpad.is_linked():
        pad.link(sinkpad)

decoder.connect("pad-added", on_pad_added, streammux)
streammux.link(pgie)
pgie.link(tracker)
tracker.link(osd)
osd.link(fakesink)

osd_pad = osd.get_static_pad("sink")
osd_pad.add_probe(Gst.PadProbeType.BUFFER, probe_fn, 0)
print("Writing tracking log...")

loop = GLib.MainLoop()

def bus_call(bus, msg, loop):
    if msg.type == Gst.MessageType.EOS:
        print("DONE! Writing JSON...")
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(tracking_log, f)
        print("Frames: " + str(frame_count))
        print("ID switches: " + str(original_switches))
        print("JSON saved: " + OUTPUT_JSON)
        loop.quit()
    elif msg.type == Gst.MessageType.ERROR:
        err, _ = msg.parse_error()
        print("ERROR: " + str(err))
        loop.quit()
    return True

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)
pipeline.set_state(Gst.State.PLAYING)
print("Pipeline started!")

try:
    loop.run()
except:
    pass
pipeline.set_state(Gst.State.NULL)
