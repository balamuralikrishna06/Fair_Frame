# backend/models/report_generator.py
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.io as pio
import base64
from io import BytesIO
import os

class ReportGenerator:
    def __init__(self):
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        print("📄 Report Generator initialized")
    
    def generate_pdf_report(self, analysis_results: Dict, job_id: str, 
                           media_type: str = "video") -> str:
        """Generate comprehensive PDF report"""
        try:
            # Create PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Add first page
            pdf.add_page()
            self._add_header(pdf, job_id, media_type)
            
            # Add summary section
            self._add_summary_section(pdf, analysis_results)
            
            # Add detailed analysis
            self._add_detailed_analysis(pdf, analysis_results)
            
            # Add recommendations
            self._add_recommendations(pdf, analysis_results)
            
            # Add timeline analysis
            if media_type == "video":
                self._add_timeline_section(pdf, analysis_results)
            
            # Add footer
            self._add_footer(pdf)
            
            # Save PDF
            report_path = self.report_dir / f"fairframe_report_{job_id}.pdf"
            pdf.output(str(report_path))
            
            print(f"✅ PDF report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            # Create a simple text report as fallback
            return self._generate_text_report(analysis_results, job_id, media_type)
    
    def _add_header(self, pdf: FPDF, job_id: str, media_type: str):
        """Add report header"""
        # Title
        pdf.set_font("Arial", "B", 24)
        pdf.set_text_color(0, 102, 204)  # Blue color
        pdf.cell(0, 15, "FAIR FRAME - BIAS ANALYSIS REPORT", ln=True, align="C")
        
        # Subtitle
        pdf.set_font("Arial", "I", 14)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, "Multimodal AI Bias Detection System", ln=True, align="C")
        
        # Separator
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
        # Report info
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(0, 0, 0)
        
        info_data = [
            ("Report ID:", job_id),
            ("Media Type:", media_type.upper()),
            ("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Analysis Type:", "Comprehensive Bias Detection")
        ]
        
        for label, value in info_data:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(40, 8, label)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, value, ln=True)
        
        pdf.ln(10)
    
    def _add_summary_section(self, pdf: FPDF, analysis_results: Dict):
        """Add summary section"""
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "EXECUTIVE SUMMARY", ln=True)
        
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Get summary data
        summary = analysis_results.get("analysis_summary", {})
        overall_score = summary.get("overall_bias_score", 0)
        risk_level = summary.get("risk_level", "UNKNOWN")
        
        # Risk indicator
        pdf.set_font("Arial", "B", 14)
        if risk_level == "HIGH":
            pdf.set_text_color(220, 53, 69)  # Red
            risk_text = "⚠️ HIGH BIAS RISK"
        elif risk_level == "MEDIUM":
            pdf.set_text_color(255, 193, 7)  # Yellow/Orange
            risk_text = "⚠️ MEDIUM BIAS RISK"
        else:
            pdf.set_text_color(40, 167, 69)  # Green
            risk_text = "✅ LOW BIAS RISK"
        
        pdf.cell(0, 10, risk_text, ln=True)
        
        # Score
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f"Overall Bias Score: {overall_score:.3f}/1.0", ln=True)
        
        # Key metrics
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Key Metrics:", ln=True)
        pdf.set_font("Arial", "", 11)
        
        metrics = []
        if "total_faces" in summary:
            metrics.append(f"• Faces Detected: {summary['total_faces']}")
        if "speech_duration" in summary:
            metrics.append(f"• Speech Duration: {summary['speech_duration']:.1f}s")
        if "word_count" in summary.get("speech_summary", {}):
            metrics.append(f"• Words Analyzed: {summary['speech_summary']['word_count']}")
        
        for metric in metrics:
            pdf.cell(0, 6, metric, ln=True)
        
        pdf.ln(10)
    
    def _add_detailed_analysis(self, pdf: FPDF, analysis_results: Dict):
        """Add detailed analysis section"""
        pdf.add_page()
        
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "DETAILED ANALYSIS", ln=True)
        
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
        detailed = analysis_results.get("detailed_results", {})
        
        # 1. Face Analysis
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 8, "1. Visual Analysis (Face Detection)", ln=True)
        pdf.ln(2)
        
        face_analysis = detailed.get("face_analysis", {})
        face_stats = face_analysis.get("face_statistics", {})
        
        if face_stats:
            pdf.set_font("Arial", "", 11)
            
            # Gender distribution
            gender_data = face_stats.get("gender", {})
            if gender_data:
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, "Gender Distribution:", ln=True)
                pdf.set_font("Arial", "", 11)
                
                for gender, data in gender_data.items():
                    if isinstance(data, dict):
                        pct = data.get("percentage", 0)
                        count = data.get("count", 0)
                        pdf.cell(0, 6, f"  • {gender.title()}: {count} faces ({pct:.1f}%)", ln=True)
            
            # Racial distribution
            race_data = face_stats.get("race", {})
            if race_data:
                pdf.ln(2)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, "Racial/Ethnic Distribution:", ln=True)
                pdf.set_font("Arial", "", 11)
                
                for race, data in race_data.items():
                    if isinstance(data, dict) and race != "unknown":
                        pct = data.get("percentage", 0)
                        count = data.get("count", 0)
                        pdf.cell(0, 6, f"  • {race.title()}: {count} faces ({pct:.1f}%)", ln=True)
            
            # Diversity score
            demo_summary = face_analysis.get("demographic_summary", {})
            diversity_score = demo_summary.get("diversity_score", 0)
            pdf.ln(2)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, f"Diversity Score: {diversity_score:.1f}/100", ln=True)
        
        else:
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 6, "No faces detected in the content", ln=True)
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(10)
        
        # 2. Audio/Speech Analysis
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 8, "2. Audio Analysis (Speech Detection)", ln=True)
        pdf.ln(2)
        
        audio_analysis = detailed.get("audio_analysis", {})
        bias_detection = audio_analysis.get("bias_detection", {})
        
        if bias_detection:
            pdf.set_font("Arial", "", 11)
            
            # Biased phrases
            biased_phrases = bias_detection.get("biased_phrases_found", [])
            if biased_phrases:
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, "Potentially Biased Phrases Found:", ln=True)
                pdf.set_font("Arial", "", 10)
                
                for i, phrase in enumerate(biased_phrases[:5]):  # Show first 5
                    pdf.multi_cell(0, 5, f"  {i+1}. \"{phrase}\"")
            
            # Pronoun analysis
            pronoun_counts = bias_detection.get("pronoun_counts", {})
            if pronoun_counts:
                pdf.ln(2)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, "Pronoun Usage:", ln=True)
                pdf.set_font("Arial", "", 11)
                
                male = pronoun_counts.get("male", 0)
                female = pronoun_counts.get("female", 0)
                total = male + female
                
                if total > 0:
                    male_pct = (male / total) * 100
                    female_pct = (female / total) * 100
                    
                    pdf.cell(0, 6, f"  • Male pronouns (he/him/his): {male} ({male_pct:.1f}%)", ln=True)
                    pdf.cell(0, 6, f"  • Female pronouns (she/her/hers): {female} ({female_pct:.1f}%)", ln=True)
                    
                    imbalance = abs(male_pct - female_pct)
                    if imbalance > 60:
                        pdf.set_text_color(220, 53, 69)
                        pdf.cell(0, 6, f"  ⚠️ High pronoun imbalance detected ({imbalance:.1f}% difference)", ln=True)
                        pdf.set_text_color(0, 0, 0)
            
            # Speaking statistics
            speaking_stats = audio_analysis.get("speaking_statistics", {})
            if speaking_stats:
                pdf.ln(2)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, "Speaking Patterns:", ln=True)
                pdf.set_font("Arial", "", 11)
                
                speaking_pct = speaking_stats.get("speaking_percentage", 0)
                avg_segment = speaking_stats.get("average_segment_duration", 0)
                
                pdf.cell(0, 6, f"  • Speech covers {speaking_pct:.1f}% of content duration", ln=True)
                pdf.cell(0, 6, f"  • Average speaking segment: {avg_segment:.1f} seconds", ln=True)
        
        else:
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 6, "No speech detected or audio analysis unavailable", ln=True)
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(10)
        
        # 3. Component Scores
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 8, "3. Component Bias Scores", ln=True)
        pdf.ln(2)
        
        combined = detailed.get("combined_analysis", {})
        component_scores = combined.get("component_scores", {})
        
        if component_scores:
            pdf.set_font("Arial", "", 11)
            
            scores = [
                ("Visual/Face Bias", component_scores.get("face_bias", 0)),
                ("Speech/Audio Bias", component_scores.get("speech_bias", 0)),
                ("Presence-Speech Discrepancy", component_scores.get("presence_speech_bias", 0))
            ]
            
            for name, score in scores:
                # Color code based on score
                if score > 0.7:
                    color = (220, 53, 69)  # Red
                    risk = "HIGH"
                elif score > 0.4:
                    color = (255, 193, 7)   # Yellow
                    risk = "MEDIUM"
                else:
                    color = (40, 167, 69)   # Green
                    risk = "LOW"
                
                pdf.set_text_color(*color)
                pdf.cell(0, 6, f"  • {name}: {score:.3f} ({risk} risk)", ln=True)
            
            pdf.set_text_color(0, 0, 0)
    
    def _add_timeline_section(self, pdf: FPDF, analysis_results: Dict):
        """Add timeline analysis section (for videos)"""
        pdf.add_page()
        
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "TIMELINE ANALYSIS", ln=True)
        
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
        timeline = analysis_results.get("timeline_analysis", [])
        
        if timeline:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Key Time Segments with Bias Events:", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 10)
            
            # Create table header
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(30, 8, "Time", border=1, fill=True)
            pdf.cell(20, 8, "Faces", border=1, fill=True)
            pdf.cell(25, 8, "Speech (s)", border=1, fill=True)
            pdf.cell(20, 8, "Score", border=1, fill=True)
            pdf.cell(20, 8, "Risk", border=1, fill=True)
            pdf.cell(0, 8, "Key Events", border=1, fill=True, ln=True)
            
            pdf.set_font("Arial", "", 9)
            
            # Add table rows
            for i, segment in enumerate(timeline[:20]):  # Show first 20 segments
                if i % 2 == 0:
                    pdf.set_fill_color(250, 250, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)
                
                # Determine risk color
                risk = segment.get("risk_level", "LOW")
                if risk == "HIGH":
                    risk_color = (220, 53, 69)
                elif risk == "MEDIUM":
                    risk_color = (255, 193, 7)
                else:
                    risk_color = (40, 167, 69)
                
                # Time
                pdf.cell(30, 8, segment.get("time_range", ""), border=1, fill=True)
                
                # Faces
                pdf.cell(20, 8, str(segment.get("face_count", 0)), border=1, fill=True)
                
                # Speech
                pdf.cell(25, 8, f"{segment.get('speech_duration', 0):.1f}", border=1, fill=True)
                
                # Score
                score = segment.get("bias_score", 0)
                pdf.cell(20, 8, f"{score:.3f}", border=1, fill=True)
                
                # Risk with color
                pdf.set_text_color(*risk_color)
                pdf.cell(20, 8, risk, border=1, fill=True)
                pdf.set_text_color(0, 0, 0)
                
                # Key events
                events = segment.get("key_events", [])
                events_text = "; ".join(events[:2]) if events else "None"
                pdf.multi_cell(0, 8, events_text, border=1, fill=True)
            
            pdf.ln(10)
            
            # Timeline summary
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "Timeline Insights:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            # Find highest bias segment
            if timeline:
                max_segment = max(timeline, key=lambda x: x.get("bias_score", 0))
                max_time = max_segment.get("time_range", "Unknown")
                max_score = max_segment.get("bias_score", 0)
                
                pdf.cell(0, 6, f"• Highest bias segment: {max_time} (score: {max_score:.3f})", ln=True)
                
                # Count risk segments
                high_risk = sum(1 for s in timeline if s.get("risk_level") == "HIGH")
                medium_risk = sum(1 for s in timeline if s.get("risk_level") == "MEDIUM")
                
                if high_risk > 0:
                    pdf.set_text_color(220, 53, 69)
                    pdf.cell(0, 6, f"• ⚠️ {high_risk} time segments with HIGH bias risk", ln=True)
                    pdf.set_text_color(0, 0, 0)
                
                if medium_risk > 0:
                    pdf.set_text_color(255, 193, 7)
                    pdf.cell(0, 6, f"• ⚠️ {medium_risk} time segments with MEDIUM bias risk", ln=True)
                    pdf.set_text_color(0, 0, 0)
        
        else:
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8, "No timeline data available", ln=True)
            pdf.set_text_color(0, 0, 0)
    
    def _add_recommendations(self, pdf: FPDF, analysis_results: Dict):
        """Add recommendations section"""
        pdf.add_page()
        
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "RECOMMENDATIONS & ACTION PLAN", ln=True)
        
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
        recommendations = analysis_results.get("recommendations", [])
        risk_level = analysis_results.get("analysis_summary", {}).get("risk_level", "UNKNOWN")
        
        if recommendations:
            pdf.set_font("Arial", "B", 14)
            
            if risk_level == "HIGH":
                pdf.set_text_color(220, 53, 69)  # Red
                pdf.cell(0, 8, "🚨 IMMEDIATE ACTIONS REQUIRED", ln=True)
            elif risk_level == "MEDIUM":
                pdf.set_text_color(255, 193, 7)  # Yellow
                pdf.cell(0, 8, "⚠️ RECOMMENDED IMPROVEMENTS", ln=True)
            else:
                pdf.set_text_color(40, 167, 69)  # Green
                pdf.cell(0, 8, "✅ MAINTAIN & MONITOR", ln=True)
            
            pdf.ln(5)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "", 11)
            
            for i, rec in enumerate(recommendations, 1):
                # Skip the header if present
                if "🚨" in rec or "⚠️" in rec or "✅" in rec:
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 7, rec, ln=True)
                    pdf.set_font("Arial", "", 11)
                else:
                    pdf.cell(0, 6, f"{i}. {rec}", ln=True)
            
            pdf.ln(10)
            
            # Add implementation timeline
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 8, "Implementation Timeline:", ln=True)
            pdf.ln(5)
            pdf.set_font("Arial", "", 11)
            
            if risk_level == "HIGH":
                timeline = [
                    "Immediate (1-7 days): Address critical bias issues",
                    "Short-term (2-4 weeks): Implement diversity guidelines",
                    "Medium-term (1-3 months): Training and policy updates",
                    "Long-term (3-6 months): Systemic changes and monitoring"
                ]
            elif risk_level == "MEDIUM":
                timeline = [
                    "Short-term (2-4 weeks): Review and plan improvements",
                    "Medium-term (1-3 months): Implement changes",
                    "Long-term (3-6 months): Monitor and optimize"
                ]
            else:
                timeline = [
                    "Ongoing: Regular monitoring and maintenance",
                    "Quarterly: Diversity assessment reviews",
                    "Annually: Comprehensive bias audit"
                ]
            
            for item in timeline:
                pdf.cell(0, 6, f"• {item}", ln=True)
        
        else:
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8, "No specific recommendations available", ln=True)
            pdf.set_text_color(0, 0, 0)
    
    def _add_footer(self, pdf: FPDF):
        """Add report footer"""
        pdf.set_y(-30)
        
        # Footer separator
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Footer text
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(128, 128, 128)
        
        footer_text = [
            "Fair Frame - Multimodal Bias Detection System",
            "Generated by AI-powered analysis",
            "This report is for informational purposes only",
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        for text in footer_text:
            pdf.cell(0, 4, text, ln=True, align="C")
    
    def _generate_text_report(self, analysis_results: Dict, job_id: str, media_type: str) -> str:
        """Generate simple text report as fallback"""
        try:
            report_path = self.report_dir / f"fairframe_report_{job_id}.txt"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("FAIR FRAME - BIAS ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Basic info
                f.write(f"Report ID: {job_id}\n")
                f.write(f"Media Type: {media_type}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary
                summary = analysis_results.get("analysis_summary", {})
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall Bias Score: {summary.get('overall_bias_score', 0):.3f}/1.0\n")
                f.write(f"Risk Level: {summary.get('risk_level', 'UNKNOWN')}\n\n")
                
                # Key findings
                findings = analysis_results.get("key_findings", [])
                if findings:
                    f.write("KEY FINDINGS\n")
                    f.write("-" * 40 + "\n")
                    for finding in findings[:10]:
                        f.write(f"• {finding}\n")
                    f.write("\n")
                
                # Recommendations
                recommendations = analysis_results.get("recommendations", [])
                if recommendations:
                    f.write("RECOMMENDATIONS\n")
                    f.write("-" * 40 + "\n")
                    for rec in recommendations:
                        f.write(f"• {rec}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("End of Report\n")
                f.write("=" * 60 + "\n")
            
            print(f"✅ Text report generated: {report_path}")
            return str(report_path)
            
        except Exceptvenv\Scripts\activateion as e:
            print(f"❌ Error generating text report: {e}")
            return ""