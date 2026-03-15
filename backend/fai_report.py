"""
FAI/PPAP Report Generator (AS9102 Compliant)

Generates First Article Inspection (FAI) reports per AS9102 standard.
Also supports PPAP (Production Part Approval Process) for automotive.

AS9102 Forms:
- Form 1: Part Number Accountability
- Form 2: Product Accountability (Raw Material, Special Processes, Functional Testing)
- Form 3: Characteristic Accountability (Dimensional and Other Characteristics)
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FAIType(Enum):
    """FAI submission types per AS9102"""
    FULL = "full"  # Complete FAI - new part
    PARTIAL = "partial"  # Partial FAI - some changes
    DELTA = "delta"  # Delta FAI - only affected characteristics


class CharacteristicType(Enum):
    """Types of characteristics per AS9102"""
    DIMENSIONAL = "dimensional"
    MATERIAL = "material"
    PROCESS = "process"
    FUNCTIONAL = "functional"
    VISUAL = "visual"
    OTHER = "other"


class ResultCode(Enum):
    """Result codes for characteristics"""
    PASS = "C"  # Conforming
    FAIL = "NC"  # Non-conforming
    WAIVER = "W"  # Waiver/Deviation accepted
    NA = "N/A"  # Not Applicable


@dataclass
class PartInfo:
    """Part identification information for Form 1"""
    part_number: str
    part_name: str
    revision: str = "A"
    drawing_number: str = ""
    drawing_revision: str = "A"
    serial_number: str = ""
    lot_number: str = ""

    organization_name: str = ""
    organization_cage_code: str = ""

    purchase_order: str = ""
    customer_name: str = ""

    fai_type: FAIType = FAIType.FULL
    fai_reason: str = "New Part"

    def to_dict(self) -> Dict:
        return {
            "part_number": self.part_number,
            "part_name": self.part_name,
            "revision": self.revision,
            "drawing_number": self.drawing_number or self.part_number,
            "drawing_revision": self.drawing_revision,
            "serial_number": self.serial_number,
            "lot_number": self.lot_number,
            "organization_name": self.organization_name,
            "organization_cage_code": self.organization_cage_code,
            "purchase_order": self.purchase_order,
            "customer_name": self.customer_name,
            "fai_type": self.fai_type.value,
            "fai_reason": self.fai_reason,
        }


@dataclass
class MaterialInfo:
    """Material/process information for Form 2"""
    material_name: str
    specification: str
    supplier: str = ""
    lot_number: str = ""
    cert_number: str = ""
    result: ResultCode = ResultCode.PASS

    def to_dict(self) -> Dict:
        return {
            "material_name": self.material_name,
            "specification": self.specification,
            "supplier": self.supplier,
            "lot_number": self.lot_number,
            "cert_number": self.cert_number,
            "result": self.result.value,
        }


@dataclass
class SpecialProcess:
    """Special process information for Form 2"""
    process_name: str
    specification: str
    supplier: str = ""
    cert_number: str = ""
    result: ResultCode = ResultCode.PASS

    def to_dict(self) -> Dict:
        return {
            "process_name": self.process_name,
            "specification": self.specification,
            "supplier": self.supplier,
            "cert_number": self.cert_number,
            "result": self.result.value,
        }


@dataclass
class FunctionalTest:
    """Functional test information for Form 2"""
    test_name: str
    specification: str
    actual_result: str = ""
    result: ResultCode = ResultCode.PASS

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "specification": self.specification,
            "actual_result": self.actual_result,
            "result": self.result.value,
        }


@dataclass
class Characteristic:
    """Dimensional/characteristic data for Form 3"""
    char_number: int
    char_designator: str  # Drawing reference
    char_type: CharacteristicType = CharacteristicType.DIMENSIONAL

    nominal: float = 0.0
    tolerance_plus: float = 0.0
    tolerance_minus: float = 0.0

    measured_value: float = 0.0
    deviation: float = 0.0

    unit: str = "mm"
    tool_used: str = ""

    result: ResultCode = ResultCode.PASS
    notes: str = ""

    # GD&T specific
    gdt_symbol: str = ""  # e.g., "⌖" for position
    datum_reference: str = ""

    def to_dict(self) -> Dict:
        return {
            "char_number": self.char_number,
            "char_designator": self.char_designator,
            "char_type": self.char_type.value,
            "nominal": self.nominal,
            "tolerance_plus": self.tolerance_plus,
            "tolerance_minus": self.tolerance_minus,
            "measured_value": round(self.measured_value, 4),
            "deviation": round(self.deviation, 4),
            "unit": self.unit,
            "tool_used": self.tool_used,
            "result": self.result.value,
            "notes": self.notes,
            "gdt_symbol": self.gdt_symbol,
            "datum_reference": self.datum_reference,
        }


@dataclass
class FAIReport:
    """Complete FAI Report per AS9102"""
    # Form 1 - Part Number Accountability
    part_info: PartInfo

    # Form 2 - Product Accountability
    materials: List[MaterialInfo] = field(default_factory=list)
    special_processes: List[SpecialProcess] = field(default_factory=list)
    functional_tests: List[FunctionalTest] = field(default_factory=list)

    # Form 3 - Characteristic Accountability
    characteristics: List[Characteristic] = field(default_factory=list)

    # Report metadata
    report_number: str = ""
    report_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    inspector_name: str = ""
    inspector_signature: str = ""

    # Summary
    overall_result: ResultCode = ResultCode.PASS
    total_characteristics: int = 0
    conforming_count: int = 0
    nonconforming_count: int = 0

    def calculate_summary(self):
        """Calculate summary statistics"""
        self.total_characteristics = len(self.characteristics)
        self.conforming_count = sum(1 for c in self.characteristics if c.result == ResultCode.PASS)
        self.nonconforming_count = sum(1 for c in self.characteristics if c.result == ResultCode.FAIL)

        # Check materials and processes
        material_pass = all(m.result == ResultCode.PASS for m in self.materials)
        process_pass = all(p.result == ResultCode.PASS for p in self.special_processes)
        test_pass = all(t.result == ResultCode.PASS for t in self.functional_tests)
        char_pass = self.nonconforming_count == 0

        if material_pass and process_pass and test_pass and char_pass:
            self.overall_result = ResultCode.PASS
        else:
            self.overall_result = ResultCode.FAIL

    def to_dict(self) -> Dict:
        self.calculate_summary()
        return {
            "report_number": self.report_number,
            "report_date": self.report_date,
            "inspector_name": self.inspector_name,
            "overall_result": self.overall_result.value,
            "summary": {
                "total_characteristics": self.total_characteristics,
                "conforming": self.conforming_count,
                "nonconforming": self.nonconforming_count,
                "pass_rate": round(100 * self.conforming_count / max(1, self.total_characteristics), 1),
            },
            "form1_part_info": self.part_info.to_dict(),
            "form2_materials": [m.to_dict() for m in self.materials],
            "form2_special_processes": [p.to_dict() for p in self.special_processes],
            "form2_functional_tests": [t.to_dict() for t in self.functional_tests],
            "form3_characteristics": [c.to_dict() for c in self.characteristics],
        }


class FAIReportGenerator:
    """
    Generator for AS9102 FAI reports and PPAP documentation.

    Usage:
        generator = FAIReportGenerator()
        report = generator.create_report_from_qc_result(qc_result, part_info)
        generator.save_json(report, "fai_report.json")
        generator.generate_pdf(report, "fai_report.pdf")
    """

    GDT_SYMBOLS = {
        "flatness": "⏥",
        "straightness": "⏤",
        "circularity": "○",
        "cylindricity": "⌭",
        "parallelism": "∥",
        "perpendicularity": "⊥",
        "angularity": "∠",
        "position": "⌖",
        "concentricity": "◎",
        "symmetry": "⌯",
        "profile_of_a_line": "⌒",
        "profile_of_a_surface": "⌓",
        "circular_runout": "↗",
        "total_runout": "↗↗",
    }

    def __init__(self, organization_name: str = "", cage_code: str = ""):
        self.organization_name = organization_name
        self.cage_code = cage_code

    def create_report_from_qc_result(
        self,
        qc_result: Dict,
        part_info: PartInfo,
        material_info: Optional[List[MaterialInfo]] = None,
        special_processes: Optional[List[SpecialProcess]] = None,
        functional_tests: Optional[List[FunctionalTest]] = None,
    ) -> FAIReport:
        """
        Create an FAI report from QC analysis results.

        Args:
            qc_result: Results from QC analysis
            part_info: Part identification information
            material_info: Optional list of materials
            special_processes: Optional list of special processes
            functional_tests: Optional list of functional tests

        Returns:
            FAIReport ready for export
        """
        # Set organization info
        if not part_info.organization_name:
            part_info.organization_name = self.organization_name
        if not part_info.organization_cage_code:
            part_info.organization_cage_code = self.cage_code

        report = FAIReport(
            part_info=part_info,
            report_number=self._generate_report_number(part_info.part_number),
            materials=material_info or [],
            special_processes=special_processes or [],
            functional_tests=functional_tests or [],
        )

        # Add default material based on QC result
        if not report.materials and "material" in qc_result:
            report.materials.append(MaterialInfo(
                material_name=qc_result.get("material", "Unknown"),
                specification="Per drawing",
                result=ResultCode.PASS
            ))

        # Convert QC measurements to Form 3 characteristics
        char_number = 1

        # Add GD&T results as characteristics
        if "gdt_results" in qc_result and qc_result["gdt_results"]:
            for gdt_type, gdt_data in qc_result["gdt_results"].items():
                if isinstance(gdt_data, dict):
                    measured = gdt_data.get("measured_value_mm", gdt_data.get("measured_value", 0))
                    tolerance = gdt_data.get("tolerance_mm", gdt_data.get("tolerance", 0.1))
                    conformance = gdt_data.get("conformance", "PASS")
                else:
                    measured = float(gdt_data) if gdt_data else 0
                    tolerance = qc_result.get("tolerance", 0.1)
                    conformance = "PASS" if measured <= tolerance else "FAIL"

                char = Characteristic(
                    char_number=char_number,
                    char_designator=f"GD&T-{char_number}",
                    char_type=CharacteristicType.DIMENSIONAL,
                    nominal=0,
                    tolerance_plus=tolerance,
                    tolerance_minus=tolerance,
                    measured_value=measured,
                    deviation=measured,
                    gdt_symbol=self.GDT_SYMBOLS.get(gdt_type, ""),
                    result=ResultCode.PASS if conformance == "PASS" else ResultCode.FAIL,
                    notes=f"{gdt_type.replace('_', ' ').title()}"
                )
                report.characteristics.append(char)
                char_number += 1

        # Add deviation measurements as characteristics
        if "statistics" in qc_result:
            stats = qc_result["statistics"]
            tolerance = qc_result.get("tolerance", 0.1)

            # Max deviation characteristic
            max_dev = stats.get("max_deviation", 0)
            report.characteristics.append(Characteristic(
                char_number=char_number,
                char_designator=f"DEV-{char_number}",
                char_type=CharacteristicType.DIMENSIONAL,
                nominal=0,
                tolerance_plus=tolerance,
                tolerance_minus=tolerance,
                measured_value=max_dev,
                deviation=max_dev,
                result=ResultCode.PASS if abs(max_dev) <= tolerance else ResultCode.FAIL,
                notes="Maximum surface deviation"
            ))
            char_number += 1

            # Mean deviation
            mean_dev = stats.get("mean_deviation", 0)
            report.characteristics.append(Characteristic(
                char_number=char_number,
                char_designator=f"DEV-{char_number}",
                char_type=CharacteristicType.DIMENSIONAL,
                nominal=0,
                tolerance_plus=tolerance,
                tolerance_minus=tolerance,
                measured_value=mean_dev,
                deviation=mean_dev,
                result=ResultCode.PASS if abs(mean_dev) <= tolerance else ResultCode.FAIL,
                notes="Mean surface deviation"
            ))
            char_number += 1

        # Add region-specific measurements
        if "regions" in qc_result:
            for region in qc_result["regions"]:
                region_name = region.get("name", f"Region {char_number}")
                mean_dev = region.get("mean_deviation", 0)
                status = region.get("status", "OK")
                tolerance = qc_result.get("tolerance", 0.1)

                report.characteristics.append(Characteristic(
                    char_number=char_number,
                    char_designator=f"REG-{char_number}",
                    char_type=CharacteristicType.DIMENSIONAL,
                    nominal=0,
                    tolerance_plus=tolerance,
                    tolerance_minus=tolerance,
                    measured_value=mean_dev,
                    deviation=mean_dev,
                    result=ResultCode.PASS if status == "OK" else ResultCode.FAIL,
                    notes=f"Region: {region_name}"
                ))
                char_number += 1

        report.calculate_summary()
        return report

    def _generate_report_number(self, part_number: str) -> str:
        """Generate a unique FAI report number"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"FAI-{part_number}-{timestamp}"

    def save_json(self, report: FAIReport, filepath: str) -> str:
        """Save report as JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        return filepath

    def generate_html(self, report: FAIReport) -> str:
        """Generate HTML representation of the FAI report"""
        report.calculate_summary()
        data = report.to_dict()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>First Article Inspection Report - {data['form1_part_info']['part_number']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; font-size: 12px; }}
        h1 {{ color: #333; font-size: 18px; }}
        h2 {{ color: #555; font-size: 14px; border-bottom: 2px solid #333; padding-bottom: 5px; }}
        h3 {{ color: #666; font-size: 12px; margin-top: 15px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
        th {{ background-color: #f0f0f0; font-weight: bold; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .header-info {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
        .header-box {{ border: 1px solid #333; padding: 10px; width: 45%; }}
        .form-title {{ background-color: #333; color: white; padding: 5px 10px; margin: 20px 0 10px 0; }}
        .summary-box {{ background-color: #f5f5f5; padding: 15px; margin: 15px 0; border: 1px solid #ccc; }}
        @media print {{
            .page-break {{ page-break-before: always; }}
        }}
    </style>
</head>
<body>
    <h1>FIRST ARTICLE INSPECTION REPORT</h1>
    <h2>AS9102 Compliant</h2>

    <div class="header-info">
        <div class="header-box">
            <strong>Report Number:</strong> {data['report_number']}<br>
            <strong>Report Date:</strong> {data['report_date']}<br>
            <strong>FAI Type:</strong> {data['form1_part_info']['fai_type'].upper()}<br>
            <strong>Overall Result:</strong> <span class="{'pass' if data['overall_result'] == 'C' else 'fail'}">{data['overall_result']}</span>
        </div>
        <div class="header-box">
            <strong>Organization:</strong> {data['form1_part_info']['organization_name']}<br>
            <strong>CAGE Code:</strong> {data['form1_part_info']['organization_cage_code']}<br>
            <strong>Customer:</strong> {data['form1_part_info']['customer_name']}<br>
            <strong>PO Number:</strong> {data['form1_part_info']['purchase_order']}
        </div>
    </div>

    <div class="summary-box">
        <strong>Summary:</strong><br>
        Total Characteristics: {data['summary']['total_characteristics']} |
        Conforming: {data['summary']['conforming']} |
        Non-Conforming: {data['summary']['nonconforming']} |
        Pass Rate: {data['summary']['pass_rate']}%
    </div>

    <div class="form-title">FORM 1 - PART NUMBER ACCOUNTABILITY</div>
    <table>
        <tr><th>Field</th><th>Value</th><th>Field</th><th>Value</th></tr>
        <tr>
            <td>Part Number</td><td><strong>{data['form1_part_info']['part_number']}</strong></td>
            <td>Part Name</td><td>{data['form1_part_info']['part_name']}</td>
        </tr>
        <tr>
            <td>Part Revision</td><td>{data['form1_part_info']['revision']}</td>
            <td>Drawing Number</td><td>{data['form1_part_info']['drawing_number']}</td>
        </tr>
        <tr>
            <td>Drawing Revision</td><td>{data['form1_part_info']['drawing_revision']}</td>
            <td>Serial Number</td><td>{data['form1_part_info']['serial_number']}</td>
        </tr>
        <tr>
            <td>Lot Number</td><td>{data['form1_part_info']['lot_number']}</td>
            <td>FAI Reason</td><td>{data['form1_part_info']['fai_reason']}</td>
        </tr>
    </table>
"""

        # Form 2 - Materials
        if data['form2_materials']:
            html += """
    <div class="form-title">FORM 2 - PRODUCT ACCOUNTABILITY: RAW MATERIALS</div>
    <table>
        <tr>
            <th>Material</th><th>Specification</th><th>Supplier</th>
            <th>Lot Number</th><th>Cert Number</th><th>Result</th>
        </tr>
"""
            for mat in data['form2_materials']:
                result_class = 'pass' if mat['result'] == 'C' else 'fail'
                html += f"""
        <tr>
            <td>{mat['material_name']}</td>
            <td>{mat['specification']}</td>
            <td>{mat['supplier']}</td>
            <td>{mat['lot_number']}</td>
            <td>{mat['cert_number']}</td>
            <td class="{result_class}">{mat['result']}</td>
        </tr>
"""
            html += "    </table>\n"

        # Form 2 - Special Processes
        if data['form2_special_processes']:
            html += """
    <div class="form-title">FORM 2 - PRODUCT ACCOUNTABILITY: SPECIAL PROCESSES</div>
    <table>
        <tr>
            <th>Process</th><th>Specification</th><th>Supplier</th>
            <th>Cert Number</th><th>Result</th>
        </tr>
"""
            for proc in data['form2_special_processes']:
                result_class = 'pass' if proc['result'] == 'C' else 'fail'
                html += f"""
        <tr>
            <td>{proc['process_name']}</td>
            <td>{proc['specification']}</td>
            <td>{proc['supplier']}</td>
            <td>{proc['cert_number']}</td>
            <td class="{result_class}">{proc['result']}</td>
        </tr>
"""
            html += "    </table>\n"

        # Form 2 - Functional Tests
        if data['form2_functional_tests']:
            html += """
    <div class="form-title">FORM 2 - PRODUCT ACCOUNTABILITY: FUNCTIONAL TESTS</div>
    <table>
        <tr>
            <th>Test</th><th>Specification</th><th>Actual Result</th><th>Result</th>
        </tr>
"""
            for test in data['form2_functional_tests']:
                result_class = 'pass' if test['result'] == 'C' else 'fail'
                html += f"""
        <tr>
            <td>{test['test_name']}</td>
            <td>{test['specification']}</td>
            <td>{test['actual_result']}</td>
            <td class="{result_class}">{test['result']}</td>
        </tr>
"""
            html += "    </table>\n"

        # Form 3 - Characteristics
        html += """
    <div class="page-break"></div>
    <div class="form-title">FORM 3 - CHARACTERISTIC ACCOUNTABILITY</div>
    <table>
        <tr>
            <th>#</th><th>Char</th><th>GD&T</th><th>Nominal</th>
            <th>Tol +</th><th>Tol -</th><th>Measured</th>
            <th>Dev</th><th>Result</th><th>Notes</th>
        </tr>
"""
        for char in data['form3_characteristics']:
            result_class = 'pass' if char['result'] == 'C' else 'fail'
            html += f"""
        <tr>
            <td>{char['char_number']}</td>
            <td>{char['char_designator']}</td>
            <td>{char['gdt_symbol']}</td>
            <td>{char['nominal']}</td>
            <td>+{char['tolerance_plus']}</td>
            <td>-{char['tolerance_minus']}</td>
            <td>{char['measured_value']}</td>
            <td>{char['deviation']}</td>
            <td class="{result_class}">{char['result']}</td>
            <td>{char['notes']}</td>
        </tr>
"""

        html += """
    </table>

    <div style="margin-top: 30px; border-top: 2px solid #333; padding-top: 15px;">
        <table style="width: 100%; border: none;">
            <tr style="border: none;">
                <td style="border: none; width: 50%;">
                    <strong>Inspector:</strong> ___________________________<br><br>
                    <strong>Date:</strong> ___________________________
                </td>
                <td style="border: none; width: 50%;">
                    <strong>Quality Approval:</strong> ___________________________<br><br>
                    <strong>Date:</strong> ___________________________
                </td>
            </tr>
        </table>
    </div>

    <div style="margin-top: 20px; font-size: 10px; color: #666;">
        Generated by Sherman QC - AI-Powered Quality Control System<br>
        Report compliant with AS9102 Rev C standard
    </div>
</body>
</html>
"""
        return html

    def save_html(self, report: FAIReport, filepath: str) -> str:
        """Save report as HTML file"""
        html = self.generate_html(report)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(html)
        return filepath


# PPAP-specific report generator
class PPAPLevel(Enum):
    """PPAP submission levels"""
    LEVEL_1 = 1  # Part Submission Warrant only
    LEVEL_2 = 2  # PSW with product samples and limited data
    LEVEL_3 = 3  # PSW with product samples and complete data
    LEVEL_4 = 4  # PSW with complete data, no samples
    LEVEL_5 = 5  # PSW with complete data and samples at supplier


@dataclass
class PPAPReport:
    """PPAP (Production Part Approval Process) Report"""
    # Part information
    part_number: str
    part_name: str
    customer: str

    # Submission info
    level: PPAPLevel = PPAPLevel.LEVEL_3
    reason_for_submission: str = "Initial submission"

    # Required elements
    design_records: bool = False
    authorized_eng_change_docs: bool = False
    customer_eng_approval: bool = False
    dfmea: bool = False
    process_flow_diagram: bool = False
    pfmea: bool = False
    control_plan: bool = False
    msa_studies: bool = False
    dimensional_results: bool = False
    material_test_results: bool = False
    initial_process_studies: bool = False
    qualified_lab_docs: bool = False
    appearance_approval: bool = False
    sample_parts: bool = False
    master_sample: bool = False
    checking_aids: bool = False
    customer_specific_requirements: bool = False
    part_submission_warrant: bool = False

    # Dimensional results from QC
    fai_report: Optional[FAIReport] = None

    def to_dict(self) -> Dict:
        elements = {
            "design_records": self.design_records,
            "authorized_eng_change_docs": self.authorized_eng_change_docs,
            "customer_eng_approval": self.customer_eng_approval,
            "dfmea": self.dfmea,
            "process_flow_diagram": self.process_flow_diagram,
            "pfmea": self.pfmea,
            "control_plan": self.control_plan,
            "msa_studies": self.msa_studies,
            "dimensional_results": self.dimensional_results,
            "material_test_results": self.material_test_results,
            "initial_process_studies": self.initial_process_studies,
            "qualified_lab_docs": self.qualified_lab_docs,
            "appearance_approval": self.appearance_approval,
            "sample_parts": self.sample_parts,
            "master_sample": self.master_sample,
            "checking_aids": self.checking_aids,
            "customer_specific_requirements": self.customer_specific_requirements,
            "part_submission_warrant": self.part_submission_warrant,
        }

        return {
            "part_number": self.part_number,
            "part_name": self.part_name,
            "customer": self.customer,
            "level": self.level.value,
            "reason_for_submission": self.reason_for_submission,
            "elements": elements,
            "fai_report": self.fai_report.to_dict() if self.fai_report else None,
        }


def create_fai_generator(
    organization_name: str = "",
    cage_code: str = ""
) -> FAIReportGenerator:
    """Create an FAI report generator instance"""
    return FAIReportGenerator(organization_name, cage_code)
