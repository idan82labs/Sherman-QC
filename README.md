# Scan QC System - Full Application

**AI-Powered Quality Control for CNC-machined and Bent Sheet Metal Parts**

A complete web application for automated QC inspection with PDF report generation.

## 🎯 Features

- ✅ **File Upload UI** - Drag & drop STL/PLY files
- ✅ **Real-time Progress** - Live progress bar during analysis
- ✅ **AI Analysis** - Intelligent interpretation of deviations
- ✅ **PDF Reports** - Professional downloadable reports
- ✅ **Drawing Context** - Upload technical drawings for AI context
- ✅ **Regional Analysis** - Breakdown by part regions
- ✅ **Root Cause Analysis** - Manufacturing issue identification

## 📁 Project Structure

```
scan_qc_app/
├── backend/
│   ├── server.py         # FastAPI server with endpoints
│   ├── qc_engine.py      # Core QC analysis engine
│   └── pdf_generator.py  # PDF report generation
├── frontend/
│   └── index.html        # Web UI (single-page app)
├── uploads/              # Temporary upload storage
├── output/               # Generated reports
├── requirements.txt      # Python dependencies
└── run.py               # Startup script
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python run.py
```

### 3. Open Browser
Navigate to: `http://localhost:8080`

### 4. Upload Files
- **Reference Model**: STL file (CAD reference)
- **Scan File**: STL or PLY from LiDAR/scanner
- **Drawing PDF**: Optional technical drawing for AI context

### 5. Get Results
- View pass/fail verdict
- Download PDF report
- Review AI analysis

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Start analysis job |
| `/api/progress/{id}` | GET | Get job progress |
| `/api/result/{id}` | GET | Get analysis results |
| `/api/download/{id}/pdf` | GET | Download PDF report |
| `/api/download/{id}/json` | GET | Download JSON data |

### Example API Usage
```bash
curl -X POST http://localhost:8080/api/analyze \
  -F "reference_file=@reference.stl" \
  -F "scan_file=@scan.ply" \
  -F "drawing_file=@drawing.pdf" \
  -F "part_id=9353735-01" \
  -F "part_name=SPS FLEX GUIDE" \
  -F "material=Al-5053-H32" \
  -F "tolerance=0.1"
```

## 📊 Report Contents

### PDF Report Includes:
- Part identification & specifications
- Pass/Fail verdict with quality score
- Statistical summary (points analyzed, pass rate, deviations)
- Regional analysis table
- AI-generated root cause analysis
- Actionable recommendations
- Detailed AI interpretation

### Quality Score Calculation:
- **Pass Rate Score** (0-60 points): % of points within tolerance
- **Max Deviation Score** (0-20 points): Penalty for extreme deviations
- **Issue Score** (0-20 points): Penalty for identified problems

## 🔧 Configuration

### Tolerance Settings
Default tolerances can be modified in the API call:
- General: ±0.1mm
- Angular: ±0.3°
- Holes: ±0.1mm

### Supported File Formats

**Reference Model:**
- STL (recommended)
- OBJ
- PLY

**Scan Data:**
- STL
- PLY (recommended)
- OBJ
- PCD

**Technical Drawing:**
- PDF (text extracted for AI context)

## 🖥️ Frontend Features

- Modern dark theme UI
- Drag & drop file upload
- Real-time progress tracking
- Color-coded verdict display
- Regional analysis table
- Root cause cards
- One-click PDF download

## 🔬 Analysis Pipeline

```
1. Load Reference    ████░░░░░░░░░░░░░░░░  15%
2. Load Scan         ████████░░░░░░░░░░░░  30%
3. Preprocess        ██████████░░░░░░░░░░  45%
4. ICP Alignment     ██████████████░░░░░░  70%
5. Deviation Calc    ████████████████░░░░  85%
6. AI Analysis       ██████████████████░░  92%
7. Generate Report   ████████████████████ 100%
```

## 🤖 AI Analysis Features

### Regional Interpretation
Each region gets AI-generated explanation:
- "Top surface shows significant outward deviation. This may indicate springback after forming."
- "Curved section shows classic springback pattern - elastic recovery after bend release."

### Root Cause Identification
- Springback detection
- Systematic offset analysis
- Edge deformation patterns
- Material/tooling issues

### Recommendations
Priority-ranked actions with expected improvement estimates.

## 📝 Example Output

```
VERDICT: ❌ FAIL
Quality Score: 82.5/100
Pass Rate: 95.9%

Root Causes Identified:
1. [HIGH] Springback on curved section: 0.08mm mean outward deviation
   - Cause: Elastic recovery after bending
   - Fix: Increase overbend by 2-4°

2. [MEDIUM] Edge deviation at front edge
   - Cause: Insufficient support during forming
   - Fix: Add edge support fixtures
```

## 🏭 Production Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "run.py"]
```

### Environment Variables
- `PORT`: Server port (default: 8080)
- `HOST`: Server host (default: 0.0.0.0)

## 📞 Support

Built for Sherman - Tailoring Integrated Solutions
Braude College - Mechanical Engineering Department
Final Project 2026
