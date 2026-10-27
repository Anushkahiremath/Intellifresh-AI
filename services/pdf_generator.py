"""Audit-grade PDF Inspection Report Generator for IntelliFresh AI.

Produces a publication-ready, multi-section quality & toxicity assessment report
with high-resolution produce imagery, Grad-CAM attention visual evidence,
chemical/biological reaction analysis, and cold-chain compliance.
"""
from __future__ import annotations

import datetime
import io
import re
import uuid
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 762, "IntelliFresh AI — Produce Quality & Toxicological Safety Audit Report")
            self.drawRightString(576, 762, f"Inspection ID: {getattr(self, 'report_id', 'INF-AUDIT')}")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)
        self.drawString(36, 30, "Confidential | IntelliFresh Multimodal AI Quality Inspection System | Final Year Capstone Project")
        self.drawRightString(576, 30, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def _make_safe_image_flowable(image_bytes: bytes | None, max_w: float = 230, max_h: float = 160) -> RLImage | Paragraph:
    if not image_bytes:
        return Paragraph("<font color='#94A3B8'><i>Visual evidence unavailable</i></font>", ParagraphStyle("NA", fontName="Helvetica", fontSize=9))
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size
        ratio = min(max_w / w, max_h / h)
        out_w, out_h = w * ratio, h * ratio
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return RLImage(buf, width=out_w, height=out_h)
    except Exception:
        return Paragraph("<font color='#94A3B8'><i>Image rendering error</i></font>", ParagraphStyle("NA", fontName="Helvetica", fontSize=9))


def generate_pdf_report(result: dict, gemini_data: dict, temperature: float, humidity: float) -> bytes:
    """Generate a comprehensive multi-section PDF report."""
    report_id = f"INF-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#064E3B"),
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#047857"),
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "DocBody",
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
    )
    body_bold = ParagraphStyle(
        "DocBodyBold",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#0F172A"),
    )
    callout_text = ParagraphStyle(
        "Callout",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1E293B"),
    )

    status_str = str(result.get("status", "Unknown")).strip()
    is_fresh = status_str.lower() == "fresh"
    badge_bg = colors.HexColor("#DCFCE7") if is_fresh else colors.HexColor("#FEE2E2")
    badge_fg = colors.HexColor("#166534") if is_fresh else colors.HexColor("#991B1B")

    story = []

    # 1. Header Banner & Audit Metadata
    header_table = Table(
        [
            [
                Paragraph("<b>INTELLIFRESH AI — QUALITY & TOXICITY AUDIT REPORT</b>", title_style),
                Paragraph(f"<b>REPORT ID:</b><br/>{report_id}<br/><b>DATE:</b> {timestamp}", ParagraphStyle("Meta", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=2))
            ],
            [
                Paragraph("Automated Multimodal Post-Harvest Produce Inspection | Final Year Capstone Project", subtitle_style),
                Paragraph(f"<b>INSPECTION MODE:</b> {result.get('model_mode', 'Multimodal Deep Learning')}", ParagraphStyle("Meta2", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#475569"), alignment=2))
            ]
        ],
        colWidths=[380, 160]
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#059669"), spaceBefore=2, spaceAfter=8))

    # 2. Executive Status Callout Box
    verdict_text = gemini_data.get("verdict", f"{status_str.upper()} — Quality Assessment Complete")
    severity = gemini_data.get("risk_severity", "LOW" if is_fresh else "HIGH")
    verdict_para = Paragraph(f"<b>STATUS VERDICT:</b> {verdict_text} &nbsp;|&nbsp; <b>RISK SEVERITY:</b> {severity}", ParagraphStyle("Vrd", fontName="Helvetica-Bold", fontSize=10, textColor=badge_fg))
    verdict_box = Table([[verdict_para]], colWidths=[540])
    verdict_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
        ("BOX", (0, 0), (-1, -1), 1.5, badge_fg),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(verdict_box)
    story.append(Spacer(1, 8))

    # 3. Core Produce & Sensor Metrics Table
    produce_name = result.get("fruit", result.get("produce_name", "Produce Item"))
    confidence = result.get("confidence", result.get("produce_confidence", 0.0))
    freshness_score = result.get("freshness_score", 0.0)
    shelf_life = result.get("shelf_life_days")
    shelf_life_str = f"{shelf_life:.1f} Days Remaining" if shelf_life is not None else "0 Days (Discard)"

    metrics_data = [
        [
            Paragraph("<b>Target Produce</b>", body_bold),
            Paragraph(f"{produce_name} <i>({gemini_data.get('produce_scientific', 'Botanical specimen')})</i>", body_style),
            Paragraph("<b>Condition Grade</b>", body_bold),
            Paragraph(f"{status_str.upper()} (Score: {freshness_score:.1f}/100)", body_style),
        ],
        [
            Paragraph("<b>Model Confidence</b>", body_bold),
            Paragraph(f"{confidence:.1f}%", body_style),
            Paragraph("<b>Usable Shelf Life</b>", body_bold),
            Paragraph(shelf_life_str, body_style),
        ],
        [
            Paragraph("<b>Storage Temp</b>", body_bold),
            Paragraph(f"{temperature:.1f} °C", body_style),
            Paragraph("<b>Storage Humidity</b>", body_bold),
            Paragraph(f"{humidity:.1f} % RH", body_style),
        ],
        [
            Paragraph("<b>Safe to Consume?</b>", body_bold),
            Paragraph(f"<b>{gemini_data.get('safe_to_use', 'YES' if is_fresh else 'NO')}</b>", ParagraphStyle("SafeP", fontName="Helvetica-Bold", fontSize=9, textColor=badge_fg)),
            Paragraph("<b>Assessment Engine</b>", body_bold),
            Paragraph(gemini_data.get("source", "IntelliFresh Multimodal AI"), body_style),
        ],
    ]
    metrics_table = Table(metrics_data, colWidths=[110, 160, 110, 160])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 10))

    # 4. Visual Evidence & Grad-CAM Explainable AI Attention Map
    story.append(Paragraph("<b>Visual Evidence & Neural Attention Analysis (Grad-CAM)</b>", h2_style))
    img_flowable = _make_safe_image_flowable(result.get("image_bytes"))
    cam_flowable = _make_safe_image_flowable(result.get("gradcam_bytes"))

    img_table = Table(
        [
            [img_flowable, cam_flowable],
            [
                Paragraph("<b>Figure 1:</b> Uploaded Inspection Specimen", ParagraphStyle("Cap1", fontName="Helvetica-Oblique", fontSize=8, alignment=1, textColor=colors.HexColor("#475569"))),
                Paragraph("<b>Figure 2:</b> Grad-CAM Attention Heatmap (Discoloration/Pathology)", ParagraphStyle("Cap2", fontName="Helvetica-Oblique", fontSize=8, alignment=1, textColor=colors.HexColor("#475569")))
            ]
        ],
        colWidths=[265, 265]
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # 5. Gemini Detailed Assessment: Shelf-life & Consumption Guidance
    story.append(Paragraph("<b>1. Detailed Usability & Remaining Shelf-Life Window</b>", h2_style))
    usable_summary = gemini_data.get("usable_days_summary", "Review storage conditions to maximize produce longevity.")
    story.append(Paragraph(usable_summary, body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>2. Handling & Culinary Action Protocol</b>", h2_style))
    culinary_action = gemini_data.get("culinary_action", "Ensure proper sanitation before consuming.")
    story.append(Paragraph(culinary_action, body_style))
    story.append(Spacer(1, 8))

    # 6. Potential Health Hazards, Microbial Growth & Toxicological Reactions
    story.append(KeepTogether([
        Paragraph("<b>3. Biological Reactions & Toxicological Hazards if Consumed</b>", h2_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceBefore=1, spaceAfter=4)
    ]))
    reactions = gemini_data.get("biological_reactions_if_consumed", "")
    for paragraph_chunk in reactions.split("\n\n"):
        clean_chunk = paragraph_chunk.strip().replace("**", "<b>").replace("<b>", "<b>", 1)
        # Convert markdown bold to reportlab bold
        clean_chunk = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", paragraph_chunk.strip())
        if clean_chunk:
            story.append(Paragraph(clean_chunk, body_style))
            story.append(Spacer(1, 4))

    # 7. Post-Harvest Pathogens & Storage Protocol
    pathogens = gemini_data.get("spoilage_pathogens", "Common fungal/bacterial molds")
    storage_proto = gemini_data.get("storage_protocol", "Store in clean cold environment.")

    recs_box_data = [
        [
            Paragraph("<b>Associated Spoilage Pathogens:</b>", body_bold),
            Paragraph(pathogens, callout_text)
        ],
        [
            Paragraph("<b>Optimal Storage Protocol:</b>", body_bold),
            Paragraph(storage_proto, callout_text)
        ]
    ]
    recs_box = Table(recs_box_data, colWidths=[160, 380])
    recs_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 4))
    story.append(recs_box)
    story.append(Spacer(1, 12))

    # 8. Sign-off / Capstone Certification Block
    signoff_table = Table(
        [
            [
                Paragraph("<b>Automated Quality System:</b><br/>IntelliFresh Multimodal ResNet50 Engine<br/>Verified by Explainable AI (Grad-CAM)", ParagraphStyle("Sign1", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#475569"))),
                Paragraph("<b>Capstone Academic Review:</b><br/>Project Examination Board<br/>Department of Computer Science & Engineering", ParagraphStyle("Sign2", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=2))
            ]
        ],
        colWidths=[270, 270]
    )
    signoff_table.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(signoff_table)

    # Build document
    canvas_maker = lambda *args, **kwargs: NumberedCanvas(*args, report_id=report_id, **kwargs)
    doc.build(story, canvasmaker=canvas_maker)
    return buffer.getvalue()
