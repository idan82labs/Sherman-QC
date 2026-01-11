"""
PDF Report Generator for QC Analysis
Generates professional PDF reports from QC analysis results.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from io import BytesIO
from pathlib import Path
from typing import Dict, Any
import datetime


class PDFReportGenerator:
    """Generate professional PDF reports from QC analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#64748b')
        ))
        
        self.styles.add(ParagraphStyle(
            name='QCSectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#1a1a2e'),
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='VerdictPass',
            parent=self.styles['Heading1'],
            fontSize=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#16a34a')
        ))
        
        self.styles.add(ParagraphStyle(
            name='VerdictFail',
            parent=self.styles['Heading1'],
            fontSize=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#dc2626')
        ))
        
        self.styles.add(ParagraphStyle(
            name='VerdictWarning',
            parent=self.styles['Heading1'],
            fontSize=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#d97706')
        ))
        
        self.styles.add(ParagraphStyle(
            name='QCBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='QCSmallText',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#64748b')
        ))
    
    def generate(self, report_data: Dict[str, Any], output_path: str) -> str:
        """
        Generate PDF report from QC analysis data.
        
        Args:
            report_data: Dictionary containing QC analysis results
            output_path: Path to save the PDF
            
        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )
        
        story = []
        
        # Title Page
        story.extend(self._create_title_section(report_data))
        
        # Verdict Section
        story.extend(self._create_verdict_section(report_data))
        
        # Statistics Section
        story.extend(self._create_statistics_section(report_data))
        
        # Regional Analysis
        story.extend(self._create_regional_section(report_data))
        
        # Root Cause Analysis
        story.extend(self._create_root_cause_section(report_data))
        
        # Recommendations
        story.extend(self._create_recommendations_section(report_data))
        
        # AI Analysis
        story.extend(self._create_ai_analysis_section(report_data))
        
        # Footer
        story.extend(self._create_footer(report_data))
        
        doc.build(story)
        return output_path
    
    def _create_title_section(self, data: Dict) -> list:
        """Create report title section"""
        elements = []
        
        # Logo/Header area
        elements.append(Spacer(1, 10*mm))
        elements.append(Paragraph("QUALITY CONTROL ANALYSIS REPORT", self.styles['ReportTitle']))
        elements.append(Paragraph("AI-Powered Dimensional Inspection System", self.styles['ReportSubtitle']))
        
        elements.append(Spacer(1, 10*mm))
        
        # Part Info Table
        part_info = [
            ['Part Number:', data.get('part_id', 'N/A'), 'Material:', data.get('material', 'N/A')],
            ['Part Name:', data.get('part_name', 'N/A'), 'Tolerance:', f"±{data.get('tolerance', 0.1)}mm"],
            ['Inspection Date:', data.get('timestamp', '')[:19].replace('T', ' '), '', ''],
        ]
        
        info_table = Table(part_info, colWidths=[80, 140, 80, 140])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#64748b')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 15*mm))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        
        return elements
    
    def _create_verdict_section(self, data: Dict) -> list:
        """Create verdict section with large pass/fail indicator"""
        elements = []
        
        verdict = data.get('overall_result', 'FAIL')
        score = data.get('quality_score', 0)
        confidence = data.get('confidence', 0)
        
        # Verdict styling
        if verdict == 'PASS':
            verdict_style = self.styles['VerdictPass']
            verdict_icon = '✓'
            bg_color = colors.HexColor('#dcfce7')
        elif verdict == 'WARNING':
            verdict_style = self.styles['VerdictWarning']
            verdict_icon = '⚠'
            bg_color = colors.HexColor('#fef3c7')
        else:
            verdict_style = self.styles['VerdictFail']
            verdict_icon = '✗'
            bg_color = colors.HexColor('#fee2e2')
        
        elements.append(Spacer(1, 10*mm))
        
        # Verdict box
        verdict_table = Table(
            [[f"{verdict_icon} {verdict}"]],
            colWidths=[440]
        )
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 24),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('ROUNDEDCORNERS', [8, 8, 8, 8]),
        ]))
        
        elements.append(verdict_table)
        
        # Score and confidence
        score_text = f"Quality Score: {score:.1f}/100  |  Confidence: {confidence:.0f}%"
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph(score_text, ParagraphStyle(
            'ScoreText',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#64748b')
        )))
        
        # AI Summary
        if data.get('ai_summary'):
            elements.append(Spacer(1, 8*mm))
            elements.append(Paragraph(data['ai_summary'], self.styles['QCBodyText']))
        
        elements.append(Spacer(1, 10*mm))
        
        return elements
    
    def _create_statistics_section(self, data: Dict) -> list:
        """Create statistics section"""
        elements = []
        
        elements.append(Paragraph("Statistical Summary", self.styles['QCSectionHeader']))
        
        stats = data.get('statistics', {})
        
        stats_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Total Points', f"{stats.get('total_points', 0):,}", 
             'Pass Rate', f"{stats.get('pass_rate', 0):.1f}%"],
            ['Points In Tolerance', f"{stats.get('points_in_tolerance', 0):,}",
             'Points Out', f"{stats.get('points_out_of_tolerance', 0):,}"],
            ['Mean Deviation', f"{stats.get('mean_deviation_mm', 0):.4f} mm",
             'Std Deviation', f"{stats.get('std_deviation_mm', 0):.4f} mm"],
            ['Max Deviation', f"{stats.get('max_deviation_mm', 0):.4f} mm",
             'Min Deviation', f"{stats.get('min_deviation_mm', 0):.4f} mm"],
        ]
        
        stats_table = Table(stats_data, colWidths=[100, 110, 100, 110])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#64748b')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 10*mm))
        
        # Alignment info
        alignment = data.get('alignment', {})
        align_text = f"Alignment Quality: {alignment.get('fitness', 0):.1f}% fitness, RMSE: {alignment.get('rmse_mm', 0):.4f}mm"
        elements.append(Paragraph(align_text, self.styles['QCSmallText']))
        
        return elements
    
    def _create_regional_section(self, data: Dict) -> list:
        """Create regional analysis section"""
        elements = []
        
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph("Regional Analysis", self.styles['QCSectionHeader']))
        
        regions = data.get('regions', [])
        if not regions:
            elements.append(Paragraph("No regional data available.", self.styles['QCBodyText']))
            return elements
        
        # Region table
        region_header = ['Region', 'Points', 'Mean Dev.', 'Max Dev.', 'Pass Rate', 'Status']
        region_rows = [region_header]
        
        for r in regions:
            status_text = '✓ OK' if r.get('status') == 'OK' else '⚠ ATTENTION'
            region_rows.append([
                r.get('name', ''),
                f"{r.get('point_count', 0):,}",
                f"{r.get('mean_deviation_mm', 0):.3f}mm",
                f"{r.get('max_deviation_mm', 0):.3f}mm",
                f"{r.get('pass_rate', 0):.1f}%",
                status_text
            ])
        
        region_table = Table(region_rows, colWidths=[90, 60, 70, 70, 60, 70])
        
        # Styling
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]
        
        # Color rows based on status
        for i, r in enumerate(regions, start=1):
            if r.get('status') != 'OK':
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fef2f2')))
        
        region_table.setStyle(TableStyle(table_style))
        elements.append(region_table)
        
        # AI interpretations
        elements.append(Spacer(1, 5*mm))
        for r in regions:
            if r.get('ai_interpretation') and r.get('status') != 'OK':
                elements.append(Paragraph(
                    f"<b>{r.get('name')}:</b> {r.get('ai_interpretation')}",
                    self.styles['QCSmallText']
                ))
                elements.append(Spacer(1, 2*mm))
        
        return elements
    
    def _create_root_cause_section(self, data: Dict) -> list:
        """Create root cause analysis section"""
        elements = []
        
        root_causes = data.get('root_causes', [])
        if not root_causes:
            return elements
        
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph("Root Cause Analysis", self.styles['QCSectionHeader']))
        
        for i, rc in enumerate(root_causes, 1):
            priority = rc.get('priority', 'MEDIUM')
            if priority == 'CRITICAL':
                bg_color = colors.HexColor('#fee2e2')
                priority_color = colors.HexColor('#dc2626')
            elif priority == 'HIGH':
                bg_color = colors.HexColor('#fef3c7')
                priority_color = colors.HexColor('#d97706')
            else:
                bg_color = colors.HexColor('#f0fdf4')
                priority_color = colors.HexColor('#16a34a')
            
            rc_table = Table([
                [f"[{priority}] Issue #{i}: {rc.get('issue', '')}"],
                [f"Likely Cause: {rc.get('likely_cause', '')}"],
                [rc.get('explanation', '')],
                [f"Confidence: {rc.get('confidence', 0):.0f}%"]
            ], colWidths=[420])
            
            rc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TEXTCOLOR', (0, 3), (0, 3), colors.HexColor('#64748b')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ]))
            
            elements.append(rc_table)
            elements.append(Spacer(1, 3*mm))
        
        return elements
    
    def _create_recommendations_section(self, data: Dict) -> list:
        """Create recommendations section"""
        elements = []
        
        recommendations = data.get('recommendations', [])
        if not recommendations:
            return elements
        
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph("Recommended Actions", self.styles['QCSectionHeader']))
        
        for i, rec in enumerate(recommendations, 1):
            priority = rec.get('priority', 'MEDIUM')
            icon = '🔴' if priority in ['CRITICAL', 'HIGH'] else '🟡'
            
            rec_text = f"<b>{i}. [{priority}]</b> {rec.get('action', '')}<br/>"
            rec_text += f"<i>Expected: {rec.get('expected_improvement', '')}</i>"
            
            elements.append(Paragraph(rec_text, self.styles['QCBodyText']))
            elements.append(Spacer(1, 2*mm))
        
        return elements
    
    def _create_ai_analysis_section(self, data: Dict) -> list:
        """Create detailed AI analysis section"""
        elements = []
        
        detailed = data.get('ai_detailed_analysis', '')
        if not detailed:
            return elements
        
        elements.append(PageBreak())
        elements.append(Paragraph("Detailed AI Analysis", self.styles['QCSectionHeader']))
        
        # Convert markdown-like formatting - safely handle text
        lines = detailed.split('\n')
        for line in lines:
            try:
                # Clean the line of problematic characters
                clean_line = line.replace('✓', '[OK]').replace('⚠️', '[!]').replace('✗', '[X]')
                clean_line = clean_line.replace('<', '&lt;').replace('>', '&gt;')
                
                if line.startswith('## '):
                    text = clean_line[3:].strip()
                    if text:
                        elements.append(Spacer(1, 5*mm))
                        elements.append(Paragraph(text, self.styles['QCSectionHeader']))
                elif line.startswith('### '):
                    text = clean_line[4:].strip()
                    if text:
                        elements.append(Spacer(1, 3*mm))
                        elements.append(Paragraph(text, self.styles['QCBodyText']))
                elif line.startswith('- '):
                    text = clean_line[2:].strip()
                    if text:
                        elements.append(Paragraph(f"  - {text}", self.styles['QCSmallText']))
                elif clean_line.strip():
                    # Remove markdown bold markers entirely for PDF
                    text = clean_line.replace('**', '').strip()
                    if text:
                        elements.append(Paragraph(text, self.styles['QCBodyText']))
            except Exception as e:
                # Skip problematic lines
                continue
        
        return elements
    
    def _create_footer(self, data: Dict) -> list:
        """Create report footer"""
        elements = []
        
        elements.append(Spacer(1, 15*mm))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 5*mm))
        
        footer_text = f"Generated by Sherman Scan QC System | AI-Powered Quality Control<br/>"
        footer_text += f"Report Date: {data.get('timestamp', '')[:19].replace('T', ' ')}"
        
        elements.append(Paragraph(footer_text, ParagraphStyle(
            'Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#94a3b8')
        )))
        
        return elements


def generate_pdf_report(report_data: Dict, output_path: str) -> str:
    """Convenience function to generate PDF report"""
    generator = PDFReportGenerator()
    return generator.generate(report_data, output_path)
