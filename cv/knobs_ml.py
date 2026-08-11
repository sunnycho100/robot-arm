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

PROMPTS = ['knob']       # 'control knob' and 'dial' were no better, and slower
CONF = 0.05              # knobs are small and unusual; the useful range is low
MODEL = 'yolov8s-world.pt'


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
        """-> the same list of dicts knobs2.find returns, source='ml'."""
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            x0, y0, x1, y1 = box
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
            if not (0.5 < w / h < 2.0):        # a knob is roughly as wide as tall
                continue
            out.append(dict(cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                            r_px=float((w + h) / 4), conf=round(float(conf), 3),
                            source='ml'))
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
