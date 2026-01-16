from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from data.models import Analysis, Severity


class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
        self.styles.add(ParagraphStyle(
            name='Title2',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        ))
    
    def generate(
        self,
        analysis: Analysis,
        output_path: Path,
        image_path: Path = None
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )
        
        story = []
        
        story.append(Paragraph("ConcreteSpot Analysis Report", self.styles['Title2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Analysis Information", self.styles['SubHeading']))
        
        info_data = [
            ["Analysis ID:", str(analysis.id) if analysis.id else "N/A"],
            ["Date/Time:", analysis.timestamp.strftime("%Y-%m-%d %H:%M:%S")],
            ["Source Type:", analysis.source_type.value.title()],
            ["Source Path:", str(Path(analysis.source_path).name)],
        ]
        
        info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        if image_path and image_path.exists():
            story.append(Paragraph("Analyzed Image", self.styles['SubHeading']))
            
            img = Image(str(image_path))
            img_width = 5 * inch
            aspect = img.imageHeight / img.imageWidth
            img_height = img_width * aspect
            
            if img_height > 4 * inch:
                img_height = 4 * inch
                img_width = img_height / aspect
            
            img.drawWidth = img_width
            img.drawHeight = img_height
            
            story.append(img)
            story.append(Spacer(1, 20))
        
        story.append(Paragraph("Detection Summary", self.styles['SubHeading']))
        
        summary_data = [
            ["Metric", "Value"],
            ["Total Detections", str(analysis.total_detections)],
            ["Cracks Detected", str(analysis.cracks_count)],
            ["Spalling Detected", str(analysis.spalling_count)],
        ]
        
        summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        if analysis.detections:
            story.append(Paragraph("Detailed Detections", self.styles['SubHeading']))
            
            det_data = [["#", "Type", "Severity", "Confidence", "Location"]]
            
            for i, det in enumerate(analysis.detections, 1):
                det_data.append([
                    str(i),
                    det.damage_type.value.title(),
                    det.severity.value.title(),
                    f"{det.confidence:.1%}",
                    f"({det.bbox.x}, {det.bbox.y})"
                ])
            
            det_table = Table(det_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.5*inch])
            
            style_commands = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]
            
            for i, det in enumerate(analysis.detections, 1):
                if det.severity == Severity.SEVERE:
                    style_commands.append(('BACKGROUND', (2, i), (2, i), colors.pink))
                elif det.severity == Severity.MODERATE:
                    style_commands.append(('BACKGROUND', (2, i), (2, i), colors.lightyellow))
                else:
                    style_commands.append(('BACKGROUND', (2, i), (2, i), colors.lightgreen))
            
            det_table.setStyle(TableStyle(style_commands))
            story.append(det_table)
        
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"Generated by ConcreteSpot v1.0.0 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        ))
        
        doc.build(story)
