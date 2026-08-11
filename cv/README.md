# Perception

Camera frame in, knob positions and pointer angles out. Every module runs
its own checks with `python <module>.py`.

```
frame ──► aruco_locate.locate()   tags of known size -> homography onto the
            │                     pedal plane, so everything downstream is
            │                     in millimetres and survives a camera move
            ▼
          knobs_ml.find()         classical first, ML only if it comes up short
            │  knobs2.find()      caps by saturation + dark-ring test
            │  knobs_ml.MLKnobs   YOLO-World, zero-shot, no training
            ▼
          pointer.angle()         white tab -> degrees
          pointer.turned()        before/after -> did it actually turn
```

| file | what it does |
|---|---|
| `compat.py` | one ArUco API across OpenCV 4.6 (Pi) and 5.x (Mac), plus the course camera intrinsics |
| `aruco_locate.py` | tags -> plane homography, `to_mm`, `rectify` |
| `knobs2.py` | classical knob finder |
| `knobs_ml.py` | YOLO-World tier and the two-tier policy |
| `pointer.py` | pointer angle and rotation check |
| `aruco.py` | the original course tracker node, kept for reference |

## Three things that are not obvious

**Saturation finds the cap, brightness does not.** Under a hot exposure the
orange pedal is brighter than the grey metal cap, so any "brightest N
percent" rule selects the pedal. The cap's real signature is that it is grey
(low saturation) while the pedal is vivid. Otsu then splits cap from body
among the low-saturation pixels only.

**An upper bound on box fill is what rejects ArUco tags.** A tag's white
cells are grey and bright, exactly like a cap. Shape separates them, but a
perimeter-based roundness test also rejects real knobs, because the white
pointer tab sticks out of the cap. Bounding-box fill is the clean cut: a
square fills its box (about 1.0), a disc fills about 0.785.

**Otsu has to be shown the right two populations.** Over a whole annulus it
cheerfully separates the dark knob body from the cap and never mentions the
pointer tab. Dropping everything at or below the median first leaves
cap-versus-tab as the only split available.

## Measured

- pointer angle: 0.58 deg mean, 1.32 deg worst over 100 synthetic knobs
- ArUco corners under warp, blur and noise: 0.65 px worst, 0.00 percent scale error
- classical knob finder on the bench photo set: 6 of 9 frames give 2+ knobs,
  16 detections, zero false positives. The 3 misses are frames where the
  pedal is blown out and the knobs sit in shadow, which is what the ML tier
  is for.

## Notes

- Model weights download on demand and are gitignored. `knobs_ml.py` imports
  torch lazily, so the Pi can run the classical tier without ultralytics.
- The ArUco checks are synthetic on purpose: the physical tag in the current
  bench photos has a torn border and detects in no dictionary at all. Print
  fresh tags before trusting any real-frame result.
