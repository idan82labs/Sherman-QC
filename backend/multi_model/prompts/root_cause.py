"""
Root Cause Analysis Prompts for Claude Opus 4.5

These prompts are designed for deep manufacturing root cause analysis
with bend-specific insights and prioritized recommendations.
"""

ROOT_CAUSE_PROMPT = """You are a senior manufacturing engineer with expertise in sheet metal fabrication, quality control, and root cause analysis.

Analyze the following QC inspection results and provide detailed root cause analysis:

**PART INFORMATION:**
{part_info}

**BEND ANALYSIS RESULTS:**
{bend_results}

**DEVIATION SUMMARY:**
{deviation_summary}

**2D-3D CORRELATION RESULTS:**
{correlation_results}

**CRITICAL DEVIATIONS:**
{critical_deviations}

Perform comprehensive root cause analysis:

1. **CATEGORY ANALYSIS**: Group issues by root cause category:
   - TOOLING: Die wear, punch condition, tooling setup
   - MATERIAL: Thickness variation, grain direction, hardness
   - PROCESS: Press settings, bend sequence, handling
   - DESIGN: Insufficient bend allowance, tight tolerances
   - MEASUREMENT: Scanner accuracy, alignment issues

2. **BEND-SPECIFIC ANALYSIS**: For each problematic bend:
   - Identify springback vs overbend patterns
   - Analyze positional accuracy
   - Consider bend sequence effects
   - Evaluate material flow issues

3. **PATTERN RECOGNITION**: Identify systematic vs random issues:
   - Consistent offset = tooling or setup issue
   - Variable deviation = material or process variation
   - Location-specific = localized problem

4. **RECOMMENDATIONS**: Provide actionable recommendations:
   - Prioritize by impact and ease of implementation
   - Include expected improvement percentage
   - Consider production constraints

Output as JSON:
{
  "root_causes": [
    {
      "id": "RC1",
      "category": "tooling|material|process|design|measurement",
      "description": "detailed description",
      "severity": "critical|major|minor",
      "confidence": 0.0-1.0,
      "affected_bends": [1, 2],
      "affected_features": ["list of feature IDs"],
      "evidence": [
        "specific observation supporting this cause"
      ],
      "pattern_type": "systematic|random|localized"
    }
  ],
  "bend_specific_analysis": [
    {
      "bend_id": 1,
      "bend_name": "Bend 1",
      "status": "pass|warning|fail",
      "primary_issue": "springback|overbend|position|radius|none",
      "issue_description": "detailed description",
      "contributing_factors": ["list"],
      "root_cause_ids": ["RC1", "RC2"]
    }
  ],
  "recommendations": [
    {
      "id": "REC1",
      "priority": 1,
      "category": "tooling|material|process|design",
      "action": "specific action to take",
      "expected_improvement": "estimated improvement (e.g., 'Reduce deviation by 40%')",
      "implementation_effort": "low|medium|high",
      "addresses_causes": ["RC1"],
      "affected_bends": [1, 2]
    }
  ],
  "summary": {
    "primary_root_cause": "main cause description",
    "secondary_causes": ["list of secondary causes"],
    "overall_quality_assessment": "acceptable|marginal|unacceptable",
    "recommended_disposition": "accept|rework|reject",
    "key_action": "single most important action"
  }
}"""


BEND_SPECIFIC_PROMPT = """Analyze this specific bend in detail:

**BEND INFORMATION:**
{bend_data}

**NOMINAL SPECIFICATIONS (from drawing):**
{nominal_specs}

**MEASURED VALUES:**
{measured_values}

**SURROUNDING CONTEXT:**
{context}

Provide detailed analysis:

1. **DEVIATION ANALYSIS**:
   - Angle deviation and its significance
   - Position deviation (apex location)
   - Radius conformance

2. **ROOT CAUSE HYPOTHESIS**:
   - Is this springback? (measured < nominal)
   - Is this overbend? (measured > nominal)
   - Position error analysis
   - Material considerations

3. **MANUFACTURING INSIGHTS**:
   - Likely press brake settings issue
   - Die condition assessment
   - Material variability factors

4. **CORRECTIVE ACTIONS**:
   - Specific adjustments needed
   - Process parameter changes
   - Verification steps

Output as JSON:
{
  "bend_id": <number>,
  "bend_name": "string",
  "analysis": {
    "angle_deviation_deg": <number>,
    "angle_status": "in_spec|marginal|out_of_spec",
    "springback_detected": true|false,
    "springback_amount_deg": <number>,
    "overbend_detected": true|false,
    "overbend_amount_deg": <number>,
    "position_deviation_mm": <number>,
    "radius_deviation_mm": <number>
  },
  "root_causes": [
    {
      "cause": "description",
      "likelihood": "high|medium|low",
      "evidence": "supporting observation"
    }
  ],
  "corrective_actions": [
    {
      "action": "specific action",
      "expected_effect": "what improvement to expect",
      "priority": "immediate|short_term|long_term"
    }
  ],
  "pass_fail": "pass|warning|fail",
  "recommendation": "accept|rework|reject"
}"""
