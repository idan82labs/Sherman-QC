"""AI-vision bend labeling tool (research only).

Pipeline: ``prep`` (planar patch segmentation + numbered renders) ->
labeler picks bend patch pairs from the images -> ``lines`` (plane-intersection
bend lines, verification renders) -> ``write-label`` (Bend Label Studio data
contract, written to ``ai_bend_lines_v1`` with ``review_status=ai_adjudicated``).

All coordinates are ``source_ply_raw``: the scan file's own coordinates, never
re-centred or re-oriented. Views use a PCA camera frame for display only.
"""

TOOL_VERSION = "bench_labeling_v1"
AI_ANNOTATOR = "claude-vision-ai-v1"
AI_REVIEW_STATUS = "ai_adjudicated"
