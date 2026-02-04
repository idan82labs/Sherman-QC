# Sherman QC - AI-Powered Quality Control System

## Project Overview

Sherman QC is a comprehensive quality control system for sheet metal manufacturing that combines 3D point cloud analysis with AI-powered insights.

### Key Features
- **3D Scan Analysis**: ICP alignment of LiDAR scans to CAD reference
- **Deviation Heatmaps**: Visual representation of manufacturing deviations
- **Bend Detection**: Automatic detection and measurement of bend angles
- **Dimension Analysis**: XLSX specification comparison with pass/fail status
- **AI Analysis**: Multi-model AI pipeline (Claude, Gemini, GPT) for root cause analysis
- **PDF Reports**: Comprehensive QC reports with visualizations

## Architecture

```
frontend/react/          # React + TypeScript + Vite
├── src/components/      # UI components (ThreeViewer, DimensionAnalysisPanel, etc.)
├── src/pages/           # Page components (Upload, JobDetail, etc.)
└── src/types/           # TypeScript interfaces

backend/                 # Python FastAPI
├── server.py           # API endpoints
├── qc_engine.py        # Core QC analysis engine
├── bend_detector.py    # Bend detection algorithm
├── dimension_parser.py # XLSX specification parser
├── bend_matcher.py     # XLSX to CAD bend matching
├── ai_analyzer.py      # AI analysis integration
├── pdf_generator.py    # Report generation
└── multi_model/        # Multi-model AI orchestration
```

## Key Algorithms

### ICP Alignment
- Iterative Closest Point with auto-scaling
- Fitness threshold: 0.3 (30% overlap required)
- RMSE calculation for alignment quality

### Bend Detection
1. Primary: Curvature-based detection using surface normals
2. Fallback: KMeans clustering on surface normals

### Deviation Calculation
- Point-to-mesh distance using KDTree
- Signed distance for direction indication
- Statistical measures: mean, std, min, max, percentiles

## Custom Commands

Available via `/critique`:
- Full system critique from 3 expert perspectives
- Mathematics/Statistics review
- Systems Engineering review
- Metrology/Manufacturing QC review

## Tech Stack

**Frontend:**
- React 18 + TypeScript
- React Three Fiber (Three.js)
- TanStack Query
- Tailwind CSS

**Backend:**
- Python 3.11+
- FastAPI
- Open3D (point cloud processing)
- NumPy/SciPy (numerical computing)
- ReportLab (PDF generation)

**AI Integration:**
- Anthropic Claude API
- Google Gemini API
- OpenAI GPT API
