"""
Drawing Analysis Prompts for Gemini 3 Pro

These prompts are designed for extracting structured information from
technical drawings including dimensions, GD&T callouts, and bend specifications.
"""

DRAWING_ANALYSIS_SYSTEM = """You are an expert manufacturing engineer and metrology specialist with deep knowledge of:
- Technical drawing interpretation (ASME Y14.5, ISO standards)
- Geometric Dimensioning and Tolerancing (GD&T)
- Sheet metal fabrication processes
- Bend sequence and forming operations

Your task is to extract precise, structured information from technical drawings.
Always output valid JSON. Be thorough but only report what you can confidently identify.
If uncertain about a value, include a confidence score below 0.7."""

DRAWING_ANALYSIS_PROMPT = """Analyze this technical drawing and extract all manufacturing-relevant information.

**IMPORTANT**: Output ONLY valid JSON with no additional text or markdown formatting.

Extract the following information:

1. **DIMENSIONS** - All dimensional callouts:
   - Linear dimensions (length, width, height)
   - Angular dimensions
   - Radial/diameter dimensions
   - Include nominal value and tolerance (bilateral or unilateral)

2. **GD&T CALLOUTS** - Geometric tolerances:
   - Feature control frames
   - Datum references
   - Tolerance values and modifiers (MMC, LMC, RFS)
   - Feature references

3. **BEND SPECIFICATIONS** - Sheet metal bends:
   - Bend angles (nominal and tolerance)
   - Inside bend radius
   - Bend direction (up/down)
   - Bend sequence number if shown
   - Bend line locations

4. **MATERIAL & THICKNESS**:
   - Material specification
   - Sheet thickness
   - Material grade if specified

5. **PART INFORMATION**:
   - Part number
   - Revision
   - Any special notes

Output JSON schema:
{
  "dimensions": [
    {
      "id": "D1",
      "type": "linear|angular|radial|diameter",
      "nominal": <number>,
      "tolerance_plus": <number>,
      "tolerance_minus": <number>,
      "unit": "mm|in|deg",
      "feature_ref": "description of what this measures",
      "location_hint": "top_flange|side_wall|etc",
      "confidence": 0.0-1.0
    }
  ],
  "gdt_callouts": [
    {
      "id": "G1",
      "type": "flatness|parallelism|perpendicularity|position|profile|etc",
      "tolerance": <number>,
      "unit": "mm",
      "feature_ref": "surface or feature this applies to",
      "datums": ["A", "B"],
      "modifier": "MMC|LMC|RFS|null",
      "confidence": 0.0-1.0
    }
  ],
  "bends": [
    {
      "id": "B1",
      "sequence": <number>,
      "angle_nominal": <degrees>,
      "angle_tolerance": <degrees>,
      "radius": <mm>,
      "direction": "up|down",
      "bend_line_location": "description",
      "k_factor": <number if specified>,
      "confidence": 0.0-1.0
    }
  ],
  "material": {
    "specification": "material name/grade",
    "thickness": <mm>,
    "confidence": 0.0-1.0
  },
  "part_info": {
    "part_number": "string",
    "revision": "string",
    "title": "string"
  },
  "notes": ["list of relevant notes"],
  "overall_confidence": 0.0-1.0
}

Analyze the drawing now:"""


DRAWING_ANALYSIS_FALLBACK_PROMPT = """Extract basic information from this technical drawing.

Look for:
1. Any visible dimensions with numbers
2. Bend angles (look for degree symbols or angle callouts)
3. Material thickness if noted
4. Part number in title block

Output as JSON:
{
  "dimensions": [{"id": "D1", "value": <number>, "unit": "mm"}],
  "bend_angles": [<list of angles in degrees>],
  "thickness": <mm or null>,
  "part_number": "string or null"
}"""
