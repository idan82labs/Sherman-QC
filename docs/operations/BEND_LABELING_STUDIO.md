# Bend Label Studio

The Bend Label Studio is a standalone, authenticated training surface at:

`/bend-labeling`

It intentionally does not show F1 predictions. Human labels must be created from the PLY geometry alone so the training truth is not biased by the detector.

## Start locally

From the project root:

```bash
PYTHON_BIN=/path/to/sherman/venv/bin/python scripts/run_bend_labeling_studio.sh
```

The default local URL is `http://127.0.0.1:3000/bend-labeling`. The launcher starts the API on port 9000 and Vite on port 3000. Both hosts and ports can be overridden with `BEND_LABELING_API_HOST`, `BEND_LABELING_API_PORT`, `BEND_LABELING_WEB_HOST`, and `BEND_LABELING_WEB_PORT`.

## Shareable labeling-only deployment

Build the restricted frontend and run the minimal server:

```bash
cd frontend/react
VITE_LABELING_ONLY=true npm run build -- --outDir ../dist-labeling
cd ../..
BEND_LABELING_USERNAME=sherman-labeler \
BEND_LABELING_PASSWORD='<strong generated password>' \
python -m uvicorn apps.labeling.main:app --host 127.0.0.1 --port 9100
```

This server exposes only authentication, the labeling API, PLY streaming, and the labeling-only frontend. It does not mount the QC dashboards or the rest of the Sherman API.

## Operator workflow

1. Sign in with an assigned Sherman QC account.
2. Confirm the annotator name. Do not share one account/name across labelers.
3. Select a PLY from the Human queue. Use All sources only when a lead has assigned a derived asset.
4. Inspect the part from isometric, front, right, and top views.
5. For every physical fold, click its start point and then its end point. The round marker is the start; the square marker is the end.
6. Use Move start or Move end if either endpoint is inaccurate. Delete the line and redraw it if its path is not centered on the fold.
7. Save draft while work is incomplete. Select No physical bends are present only after inspecting the complete part.
8. Select Complete only after every bend has been checked from at least two views.

Keyboard shortcuts:

- `Cmd/Ctrl+S`: save draft
- `Esc`: cancel a pending point or endpoint edit
- `Delete/Backspace`: delete the selected bend
- `Left/Right Arrow`: previous or next PLY in the filtered queue

## Data contract

Labels are stored atomically under:

`data/bend_corpus/annotations/human_bend_lines_v1/`

Each JSON file records:

- the stable PLY asset ID and source-relative path;
- the annotator and review status;
- each bend's start and end in `source_ply_raw` coordinates;
- a monotonically increasing revision;
- explicit confirmation for true zero-bend parts.

Concurrent stale saves return HTTP 409 instead of overwriting a newer revision. Back up the annotation directory as part of the normal corpus backup policy.

## Inventory scope

The API indexes PLY files from `source-assets`, `data`, `uploads`, `live_scans`, and `output`. The default Human queue excludes derived output and reference meshes, but All sources exposes the complete indexed inventory.
