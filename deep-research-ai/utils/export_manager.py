# utils/export_manager.py
from exporters.markdown_exporter import export_markdown
from exporters.pdf_exporter import export_pdf
from exporters.docx_exporter import export_docx


def generate_export(report: str, export_type: str):
    if export_type == "md":
        return export_markdown(report)

    elif export_type == "pdf":
        return export_pdf(report)

    elif export_type == "docx":
        return export_docx(report)

    raise ValueError("Unsupported export type")