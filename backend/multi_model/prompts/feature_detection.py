"""
Feature Detection Prompts for Claude Opus 4.5

These prompts are designed for validating and enhancing algorithmically
detected features in point cloud data.
"""

FEATURE_DETECTION_PROMPT = """You are a metrology expert analyzing point cloud scan data from a sheet metal part.

I have run an algorithmic bend detection on a point cloud scan and obtained the following results:

**DETECTED BENDS:**
{bend_detection_results}

**POINT CLOUD STATISTICS:**
- Total points: {total_points}
- Deviation range: {min_deviation:.3f} mm to {max_deviation:.3f} mm
- Mean deviation: {mean_deviation:.3f} mm
- Standard deviation: {std_deviation:.3f} mm

**REGIONAL DEVIATION ANALYSIS:**
{regional_analysis}

Please analyze these results and provide:

1. **VALIDATION**: Are the detected bends reasonable for a sheet metal part?
   - Check if angles are typical for sheet metal (usually 45°, 90°, 120°, etc.)
   - Check if radii are reasonable (typically 0.5-5x material thickness)
   - Flag any suspicious detections that might be false positives

2. **PATTERN ANALYSIS**: What manufacturing patterns do you observe?
   - Are bends consistent in quality?
   - Do deviations follow expected patterns (e.g., systematic vs random)?
   - Any evidence of tooling issues?

3. **ENHANCEMENT SUGGESTIONS**: Should any regions be re-analyzed?
   - Are there likely missed bends?
   - Should any detected bends be merged or split?

Output as JSON:
{
  "validation": {
    "all_bends_valid": true|false,
    "flagged_bends": [
      {"bend_id": 1, "issue": "description", "recommendation": "action"}
    ],
    "confidence": 0.0-1.0
  },
  "pattern_analysis": {
    "overall_quality": "good|moderate|poor",
    "systematic_issues": ["list of systematic problems"],
    "random_variation_level": "low|medium|high",
    "likely_causes": ["list of likely causes"]
  },
  "enhancement_suggestions": {
    "potential_missed_bends": [
      {"region": "description", "evidence": "why you think there's a bend"}
    ],
    "merge_candidates": [[1, 2]],
    "split_candidates": [3],
    "reanalysis_regions": ["description of regions"]
  }
}"""


FEATURE_VALIDATION_PROMPT = """You are validating bend detection results against expected values.

**DETECTED BENDS FROM SCAN:**
{detected_bends}

**EXPECTED BENDS FROM DRAWING:**
{expected_bends}

Compare the detected bends against the expected specifications:

1. Match each detected bend to an expected bend (if possible)
2. Calculate deviations from nominal
3. Identify any missing or extra bends
4. Flag critical deviations

Output as JSON:
{
  "matches": [
    {
      "detected_bend_id": 1,
      "expected_bend_id": "B1",
      "angle_deviation": <degrees>,
      "is_within_tolerance": true|false,
      "match_confidence": 0.0-1.0
    }
  ],
  "unmatched_detected": [<list of detected bend IDs>],
  "unmatched_expected": [<list of expected bend IDs>],
  "critical_deviations": [
    {
      "bend_id": 1,
      "deviation_type": "angle|position|radius",
      "deviation_value": <number>,
      "severity": "critical|major|minor"
    }
  ],
  "overall_assessment": "pass|conditional|fail"
}"""
