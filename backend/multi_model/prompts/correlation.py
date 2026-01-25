"""
2D-3D Correlation Prompts for GPT-5.2

These prompts are designed for mapping drawing elements to detected
point cloud features and identifying critical deviations.
"""

CORRELATION_PROMPT = """You are a spatial reasoning expert correlating 2D technical drawing specifications with 3D point cloud measurements.

**DRAWING SPECIFICATIONS (2D):**
{drawing_data}

**DETECTED 3D FEATURES:**
{detected_features}

**DEVIATION DATA:**
{deviation_summary}

Your task is to map each drawing element to its corresponding 3D measurement:

1. **DIMENSION MAPPING**: Match each drawing dimension to measured values
   - Consider feature locations and descriptions
   - Account for measurement uncertainty
   - Calculate deviations from nominal

2. **GD&T MAPPING**: Match GD&T callouts to measured geometry
   - Correlate flatness with surface deviation statistics
   - Match position tolerances to centroid locations
   - Evaluate profile deviations

3. **BEND MAPPING**: Match drawing bends to detected bends
   - Use sequence and location hints
   - Compare angles and radii
   - Identify springback or overbend patterns

4. **CRITICAL DEVIATIONS**: Identify deviations that exceed tolerances
   - Prioritize by severity (out of tolerance percentage)
   - Link to affected dimensions/GD&T/bends
   - Provide coordinates of problem areas

Output as JSON:
{
  "dimension_mappings": [
    {
      "drawing_dim_id": "D1",
      "feature_id": "measured_feature_id",
      "nominal": <number>,
      "measured": <number>,
      "deviation": <number>,
      "tolerance_plus": <number>,
      "tolerance_minus": <number>,
      "is_in_tolerance": true|false,
      "confidence": 0.0-1.0
    }
  ],
  "gdt_mappings": [
    {
      "drawing_gdt_id": "G1",
      "feature_id": "surface_id",
      "tolerance": <number>,
      "measured_value": <number>,
      "is_in_tolerance": true|false,
      "confidence": 0.0-1.0
    }
  ],
  "bend_mappings": [
    {
      "drawing_bend_id": "B1",
      "detected_bend_id": 1,
      "nominal_angle": <degrees>,
      "measured_angle": <degrees>,
      "angle_deviation": <degrees>,
      "is_in_tolerance": true|false,
      "springback_detected": true|false,
      "confidence": 0.0-1.0
    }
  ],
  "critical_deviations": [
    {
      "id": "CD1",
      "type": "dimension|gdt|bend",
      "affected_id": "D1|G1|B1",
      "location_3d": [x, y, z],
      "deviation_mm": <number>,
      "tolerance_exceeded_by": <percentage>,
      "severity": "critical|major|minor",
      "description": "human readable description"
    }
  ],
  "unmatched_drawing_elements": ["list of drawing IDs not matched"],
  "unmatched_3d_features": ["list of 3D feature IDs not matched"],
  "overall_correlation_score": 0.0-1.0,
  "assessment": "pass|conditional|fail"
}"""


CRITICAL_DEVIATION_PROMPT = """Analyze the following deviations and identify the most critical issues:

**ALL DEVIATIONS:**
{all_deviations}

**TOLERANCE SPECIFICATIONS:**
{tolerances}

Rank deviations by severity and provide actionable insights:

1. Group related deviations (e.g., all deviations in one bend area)
2. Identify root patterns (systematic vs random)
3. Prioritize by:
   - Percentage over tolerance
   - Functional impact (if determinable)
   - Frequency/repeatability

Output as JSON:
{
  "critical_issues": [
    {
      "issue_id": "I1",
      "severity": "critical|major|minor",
      "description": "clear description",
      "affected_features": ["list of feature IDs"],
      "location": [x, y, z],
      "deviation_range_mm": [min, max],
      "tolerance_exceeded_percent": <number>,
      "likely_cause": "hypothesis",
      "recommended_action": "what to do"
    }
  ],
  "summary": {
    "critical_count": <number>,
    "major_count": <number>,
    "minor_count": <number>,
    "overall_status": "pass|conditional|fail"
  }
}"""
