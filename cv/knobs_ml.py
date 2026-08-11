#!/usr/bin/env python3
"""ML knob finder: the second tier, for frames the classical one cannot read.

knobs2.py is fast, dependency-free and precise when the lighting behaves. It
gives up when the pedal is blown out and the knobs fall into shadow, because
its cap-versus-body split needs those two to be distinguishable. A
zero-shot detector does not care about any of that, so it covers exactly the
frames the classical tier drops.

YOLO-World takes class names as plain text, so there is no training set and
no labelling. Grounding DINO and OWL-ViT were considered and rejected: both
run at seconds per frame on this hardware, where YOLO-World is tens of
milliseconds on the Mac and about a second on the Pi. Detection only runs
between arm moves, so a second is fine.

    ml = MLKnobs()                      # loads on first use, not on import
    knobs = ml.find(frame)              # same dict schema as knobs2.find

The weights are downloaded on demand and are gitignored; they are not source.
Import of this module never pulls in torch, so the Pi can run knobs2 without
having ultralytics installed at all.
"""
import numpy as np

# Measured, not guessed. A SINGLE class name returns nothing at all here:
# 'knob', 'dial', 'black knob' and 'guitar pedal knob' each scored zero boxes
# across the whole test set, even at conf 0.01. Two descriptive prompts fire
# on every frame. YOLO-World scores an image region against the class
# embeddings it was given, so with one class there is nothing to be more like,
# and everything stays below threshold. Give it at least two.
PROMPTS = ['round black dial', 'silver cap']
CONF = 0.02              # these knobs score low; the useful range really is here
MAX_AREA = 0.25          # a box bigger than this fraction of the frame is junk
IOU_MERGE = 0.4          # boxes overlapping more than this are the same knob
MODEL = 'yolov8s-world.pt'


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


class MLKnobs:
    """Lazy wrapper so importing this file costs nothing until it is used."""

    def __init__(self, model=MODEL, prompts=PROMPTS, conf=CONF):
        self.model_name, self.prompts, self.conf = model, prompts, conf
        self._m = None

    @property
    def model(self):
        if self._m is None:
            from ultralytics import YOLOWorld          # imported here on purpose
            self._m = YOLOWorld(self.model_name)
            self._m.set_classes(self.prompts)
        return self._m

    def find(self, frame, view=None):
        """-> the same list of dicts knobs2.find returns, source='ml'.

        The raw output needs cleaning: it emits several boxes per knob and
        occasionally one the size of the whole frame.
        """
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        area = frame.shape[0] * frame.shape[1]
        boxes = sorted(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()),
                       key=lambda t: -t[1])          # best first, so it wins ties
        kept, out = [], []
        for (x0, y0, x1, y1), conf in boxes:
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
            if not (0.5 < w / h < 2.0):        # a knob is roughly as wide as tall
                continue
            if w * h > MAX_AREA * area:        # frame-sized box, not a knob
                continue
            if any(_iou((x0, y0, x1, y1), k) > IOU_MERGE for k in kept):
                continue                       # another box on the same knob
            kept.append((x0, y0, x1, y1))
            out.append(dict(cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                            r_px=float((w + h) / 4),
                            conf=round(float(conf), 3), source='ml'))
        if view is not None:
            for k in out:
                k['cx_mm'], k['cy_mm'] = view.to_mm(k['cx'], k['cy'])
            out.sort(key=lambda k: (k['cx_mm'], k['cy_mm']))
        else:
            out.sort(key=lambda k: -k['conf'])
        for i, k in enumerate(out):
            k['name'] = f'knob{i}'
        return out

    def save_baked(self, path='knob-yolo.pt'):
        """Freeze the prompt into a plain detector, for export to the Pi."""
        self.model.save(path)
        return path


def find(frame, view=None, classical_first=True, min_knobs=2):
    """The tier policy: classical when it is confident, ML to cover the rest.

    Deliberately not a merge of the two. The classical detector is the more
    precise of the pair when it works at all, and mixing sources would make
    knob names jump between frames, which the turn verification cannot
    tolerate. So one source wins per frame and says which it was.
    """
    if classical_first:
        import knobs2
        got = knobs2.find(frame, view=view)
        if len(got) >= min_knobs:
            return got
    return MLKnobs().find(frame, view=view)


if __name__ == '__main__':
    import json, pathlib, sys, cv2
    docs = pathlib.Path(__file__).resolve().parents[2] / 'docs' / 'bench_photos'
    if not (docs / 'manifest.json').exists():
        print('bench photos not present, nothing to test against')
        sys.exit(0)
    import knobs2
    man = json.load(open(docs / 'manifest.json'))
    tests = sorted(n for n, v in man.items()
                   if v['orientation'] == 'topdown' and v['knobs'] >= 2
                   and not v['overlay'])
    ml = MLKnobs()
    rows = []
    for n in tests:
        frame = cv2.imread(str(docs / n))
        c = len(knobs2.find(frame))
        m = len(ml.find(frame))
        rows.append((n, c, m))
        print(f'  {n[:22]}: classical {c}, ml {m}')
    covered = sum(1 for _, c, m in rows if max(c, m) >= 2)
    only_ml = [n[:22] for n, c, m in rows if c < 2 <= m]
    print(f'\n{covered}/{len(rows)} frames covered by at least one tier')
    if only_ml:
        print(f'ML rescued frames the classical tier missed: {only_ml}')
