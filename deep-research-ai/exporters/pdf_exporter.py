from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

def export_pdf(report: str):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, rightMargin=0.5*inch, leftMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Create a custom style for body text that wraps
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=0,
    )

    # Split report into paragraphs and create elements
    elements = []
    for para in report.split('\n'):
        if para.strip():
            elements.append(Paragraph(para, body_style))
        else:
            elements.append(Spacer(1, 0.2*inch))

    doc.build(elements)

    buffer.seek(0)
    return buffer.getvalue()