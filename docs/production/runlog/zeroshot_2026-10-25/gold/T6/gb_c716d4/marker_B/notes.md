# gb_c716d4 marker B

Box-like enclosure, t ~1.9 mm, full scan. 4 bends.

| line | patches | what |
|---|---|---|
| B1 | 2-3 | panel 1/2 (with window) to panel 3/4. Large-radius fold; thin strips 11-20 are the arc facets. margin 8 mm. Apex offset 6 mm, on the arc |
| B2 | 2-5 | panel 1/2 to frame wall 5/6 (5 is a ~13 mm rim around a big opening). Rounded fold |
| B3 | 3-7 | panel 3/4 to end wall 7/8. Rounded fold |
| B4 | 5-9 | frame wall rim 5/6 to flange 9/10 (268 mm). Rounded fold at f=0.2/0.8. Around mid-length a relief or notch interrupts it, and an edge there pulled the envelope apex off the skin, so I used apex_mode station_median (offset 1.66 vs 1.64 expected) |

How folds were told from seams: on 8 mm sections, the true folds show a rounded outer arc with apex offset ~1.5 mm (expected 1.6).
Rejected as corner seams/joints, not formed bends:
- 3-5 (137 mm)
- 1-7 (61 mm)
- 3-9 (25 mm)
These show zero outer radius, a sharp inner V and crossing skins (apex offsets 0.55 / 0.69 / 0.12 mm; inner-skin pairs at ~0 or negative). Each of these walls is already formed from another panel by a clear rounded fold.
Also rejected: 9-22, where 22 is a 2 mm strip, i.e. the plate thickness edge (only one plate in the sections); thin strips 21 and 24 (thickness edges); hole and window walls.
Trial runs are kept in trial1/ and trial2/ for the adjudicator.
Doubts: the seam/fold call on 3-5 and 5-9 is the main risk. If 5-9 were a seam instead, it would drop. I am fairly confident 5-9 is a fold (rounded arc away from the notch).
