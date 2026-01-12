# backend/models/bias_analyzer.py
import cv2
import numpy as np
import asyncio
import librosa
import whisper
import torch
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from collections import defaultdict
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json
import subprocess
import traceback
import math

class MultimodalBiasAnalyzer:
    def __init__(self, model_size: str = "base"):
        """
        Initialize the bias analyzer with specified models.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        print("🔧 Initializing FairFrame Bias Analyzer...")
        print("=" * 50)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Using device: {self.device}")
        
        try:
            # Load Whisper model for speech recognition
            print(f"🗣️ Loading Whisper model ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            print("  ✅ Whisper loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Whisper: {e}")
            self.whisper_model = None
        
        # Initialize face detector
        try:
            print("👥 Loading face detection models...")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("  ✅ OpenCV face cascade loaded")
            else:
                print("⚠️ Face cascade not found, downloading...")
                # You might need to download the cascade file
                self.face_cascade = None
        except Exception as e:
            print(f"❌ Face cascade error: {e}")
            self.face_cascade = None
        
        # Create temp directory
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Load bias configuration
        self.bias_config = self._load_bias_config()
        
        print("=" * 50)
        print("✅ Bias Analyzer initialized successfully!")
    
    def _load_bias_config(self) -> Dict:
        """Load bias detection configuration"""
        return {
            "gender_keywords": {
                "male": ["he", "him", "his", "man", "men", "boy", "boys", "gentleman", "gentlemen", "male"],
                "female": ["she", "her", "hers", "woman", "women", "girl", "girls", "lady", "ladies", "female"],
                "neutral": ["they", "them", "their", "person", "people", "individual", "one"]
            },
            "racial_keywords": {
                "asian": ["asian", "chinese", "japanese", "korean", "indian", "vietnamese", "thai", "filipino"],
                "black": ["black", "african", "african-american", "afro", "black man", "black woman"],
                "white": ["white", "caucasian", "european", "white man", "white woman"],
                "hispanic": ["hispanic", "latino", "latina", "mexican", "spanish", "puerto rican"]
            },
            "stereotype_patterns": [
                "emotional woman", "strong man", "bossy woman", "angry black",
                "smart asian", "lazy mexican", "old and slow", "young and naive",
                "tech guy", "nurse girl", "female driver", "male nurse",
                "bitchy", "hysterical", "too emotional", "not technical",
                "aggressive black", "model minority", "welfare queen"
            ],
            "profession_stereotypes": {
                "male_dominated": ["engineer", "programmer", "ceo", "doctor", "surgeon", "pilot", "soldier", "firefighter"],
                "female_dominated": ["nurse", "teacher", "secretary", "receptionist", "librarian", "hairdresser"]
            }
        }
    
    async def analyze_video_detailed(self, video_path: str, job_id: str, 
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Complete video analysis with frame-by-frame processing
        
        Args:
            video_path: Path to video file
            job_id: Unique job identifier
            progress_callback: Callback for progress updates
            
        Returns:
            Comprehensive analysis results
        """
        print(f"🎬 Starting detailed analysis for job {job_id}")
        print(f"📁 Video: {video_path}")
        
        # Update progress
        if progress_callback:
            await asyncio.sleep(0.1)  # Small delay for UI
            progress_callback(job_id, 5, "Initializing analysis...")
        
        try:
            # Get video metadata
            if progress_callback:
                progress_callback(job_id, 10, "Extracting video metadata...")
            
            video_info = await self._get_video_metadata(video_path)
            print(f"📊 Video Info: {video_info}")
            
            # Extract audio
            if progress_callback:
                progress_callback(job_id, 15, "Extracting audio...")
            
            audio_path = await self._extract_audio(video_path, job_id)
            if not audio_path:
                raise ValueError("Failed to extract audio from video")
            
            print(f"🎵 Audio extracted to: {audio_path}")
            
            # Run face analysis
            if progress_callback:
                progress_callback(job_id, 20, "Starting face detection...")
            
            face_results = await self._analyze_video_faces(
                video_path, job_id, progress_callback
            )
            
            # Run audio analysis
            if progress_callback:
                progress_callback(job_id, 60, "Analyzing audio content...")
            
            audio_results = await self._analyze_audio_detailed(
                audio_path, video_info["duration"], job_id, progress_callback
            )
            
            # Combine results
            if progress_callback:
                progress_callback(job_id, 85, "Combining analysis results...")
            
            combined_results = self._combine_video_analysis(
                face_results, audio_results, video_info
            )
            
            # Generate visualizations
            if progress_callback:
                progress_callback(job_id, 90, "Generating visualizations...")
            
            visualizations = await self._generate_visualizations(combined_results, job_id)
            
            # Clean up temp files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Prepare final results
            result = {
                "job_id": job_id,
                "status": "completed",
                "media_type": "video",
                "timestamp": datetime.now().isoformat(),
                "video_info": video_info,
                "analysis_summary": {
                    "overall_bias_score": combined_results["overall_bias_score"],
                    "risk_level": combined_results["risk_level"],
                    "total_faces": face_results.get("total_faces_detected", 0),
                    "speech_duration": audio_results.get("speech_duration", 0),
                    "key_findings": combined_results["key_findings"][:5]  # Top 5 findings
                },
                "detailed_results": {
                    "face_analysis": face_results,
                    "audio_analysis": audio_results,
                    "combined_analysis": combined_results
                },
                "timeline_analysis": combined_results.get("timeline", []),
                "visualizations": visualizations,
                "recommendations": combined_results.get("recommendations", []),
                "download_links": {
                    "report": f"/api/report/{job_id}/download",
                    "graphs": {
                        "bias_timeline": f"/api/graph/{job_id}/timeline",
                        "demographics": f"/api/graph/{job_id}/demographics",
                        "speech_analysis": f"/api/graph/{job_id}/speech"
                    }
                }
            }
            
            if progress_callback:
                progress_callback(job_id, 100, "Analysis complete!")
            
            print(f"✅ Analysis completed for job {job_id}")
            return result
            
        except Exception as e:
            print(f"❌ Analysis failed for job {job_id}: {str(e)}")
            print(traceback.format_exc())
            
            if progress_callback:
                progress_callback(job_id, -1, f"Analysis failed: {str(e)}")
            
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            if fps > 0:
                duration = total_frames / fps
            else:
                # Try alternative method
                cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            cap.release()
            
            # Get file size
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            
            return {
                "fps": fps,
                "total_frames": total_frames,
                "duration": duration,
                "resolution": f"{width}x{height}",
                "file_size_mb": round(file_size, 2),
                "file_path": video_path
            }
            
        except Exception as e:
            print(f"Error getting video metadata: {e}")
            return {
                "fps": 0,
                "total_frames": 0,
                "duration": 0,
                "resolution": "Unknown",
                "file_size_mb": 0,
                "file_path": video_path,
                "error": str(e)
            }
    
    async def _extract_audio(self, video_path: str, job_id: str) -> Optional[str]:
        """Extract audio from video file"""
        try:
            audio_path = self.temp_dir / f"audio_{job_id}.wav"
            
            # Use moviepy to extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                print("⚠️ No audio track found in video")
                return None
            
            audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            
            # Close clips to free resources
            audio.close()
            video.close()
            
            return str(audio_path)
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    async def _analyze_video_faces(self, video_path: str, job_id: str, 
                                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Analyze faces in video"""
        print(f"👥 Starting face analysis for {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file", "total_faces_detected": 0}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return {"error": "Invalid video properties", "total_faces_detected": 0}
        
        # Analyze every nth frame (for performance)
        frame_interval = max(1, int(fps // 2))  # 2 frames per second
        frame_count = 0
        all_faces = []
        timeline_data = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    
                    # Convert to RGB for DeepFace
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    try:
                        # Use DeepFace for face detection and analysis
                        face_analyses = DeepFace.analyze(
                            img_path=rgb_frame,
                            actions=['age', 'gender', 'race', 'emotion'],
                            detector_backend='retinaface',  # or 'opencv', 'ssd', 'mtcnn'
                            enforce_detection=False,
                            silent=True
                        )
                        
                        # Process each detected face
                        frame_faces = []
                        if isinstance(face_analyses, list):
                            for i, analysis in enumerate(face_analyses):
                                face_info = {
                                    "timestamp": timestamp,
                                    "frame": frame_count,
                                    "face_id": f"face_{len(all_faces) + i}",
                                    "demographics": {
                                        "age": analysis.get("age", 0),
                                        "gender": analysis.get("dominant_gender", "unknown").lower(),
                                        "race": analysis.get("dominant_race", "unknown").lower(),
                                        "emotion": analysis.get("dominant_emotion", "neutral").lower(),
                                        "confidence": analysis.get("face_confidence", 0)
                                    }
                                }
                                frame_faces.append(face_info)
                                all_faces.append(face_info)
                        
                        timeline_data.append({
                            "timestamp": timestamp,
                            "frame": frame_count,
                            "face_count": len(frame_faces),
                            "faces": frame_faces
                        })
                        
                    except Exception as e:
                        # No faces detected or error in analysis
                        timeline_data.append({
                            "timestamp": timestamp,
                            "frame": frame_count,
                            "face_count": 0,
                            "faces": []
                        })
                
                frame_count += 1
                
                # Update progress periodically
                if progress_callback and frame_count % (fps * 5) == 0:
                    progress = 20 + int((frame_count / total_frames) * 35)
                    progress_callback(job_id, progress, 
                                    f"Face analysis: {frame_count}/{total_frames} frames")
                
                # Break early for testing (remove in production)
                if os.getenv("FF_TEST_MODE") and frame_count > fps * 30:  # 30 seconds max
                    print("⚠️ Test mode: Breaking early after 30 seconds")
                    break
        
        finally:
            cap.release()
        
        # Calculate statistics
        stats = self._calculate_face_statistics(all_faces)
        
        return {
            "total_faces_detected": len(all_faces),
            "face_timeline": timeline_data,
            "face_statistics": stats,
            "demographic_summary": self._summarize_demographics(all_faces),
            "analysis_settings": {
                "frame_interval": frame_interval,
                "total_frames_processed": frame_count,
                "fps": fps
            }
        }
    
    def _calculate_face_statistics(self, faces: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive face statistics"""
        if not faces:
            return {
                "message": "No faces detected",
                "gender": {},
                "race": {},
                "age_groups": {},
                "emotion": {}
            }
        
        stats = {
            "gender": defaultdict(int),
            "race": defaultdict(int),
            "age_groups": defaultdict(int),
            "emotion": defaultdict(int),
            "total_faces": len(faces)
        }
        
        for face in faces:
            demo = face["demographics"]
            
            # Gender
            gender = demo.get("gender", "unknown")
            stats["gender"][gender] += 1
            
            # Race
            race = demo.get("race", "unknown")
            stats["race"][race] += 1
            
            # Age group
            age = demo.get("age", 0)
            if age < 18:
                stats["age_groups"]["under_18"] += 1
            elif age < 30:
                stats["age_groups"]["18_29"] += 1
            elif age < 50:
                stats["age_groups"]["30_49"] += 1
            elif age < 65:
                stats["age_groups"]["50_64"] += 1
            else:
                stats["age_groups"]["65_plus"] += 1
            
            # Emotion
            emotion = demo.get("emotion", "neutral")
            stats["emotion"][emotion] += 1
        
        # Convert to percentages
        total = len(faces)
        for category in ["gender", "race", "emotion"]:
            for key in list(stats[category].keys()):
                stats[category][key] = {
                    "count": stats[category][key],
                    "percentage": round((stats[category][key] / total) * 100, 2)
                }
        
        # Age groups percentages
        for key in list(stats["age_groups"].keys()):
            stats["age_groups"][key] = {
                "count": stats["age_groups"][key],
                "percentage": round((stats["age_groups"][key] / total) * 100, 2)
            }
        
        return stats
    
    def _summarize_demographics(self, faces: List[Dict]) -> Dict[str, Any]:
        """Create demographic summary"""
        if not faces:
            return {"message": "No faces detected", "diversity_score": 0.0}
        
        # Count unique demographics
        unique_genders = set()
        unique_races = set()
        ages = []
        
        for face in faces:
            demo = face["demographics"]
            unique_genders.add(demo.get("gender", "unknown"))
            unique_races.add(demo.get("race", "unknown"))
            ages.append(demo.get("age", 0))
        
        # Calculate diversity score (0-100)
        gender_diversity = (len(unique_genders) / 3) * 50  # Max 3 genders
        race_diversity = min(len(unique_races) / 5, 1) * 50  # Max 5 races
        age_diversity = min(len(set([a//10 for a in ages])) / 6, 1) * 50  # Age decades
        
        diversity_score = gender_diversity + race_diversity + age_diversity
        
        return {
            "unique_genders": len(unique_genders),
            "unique_races": len(unique_races),
            "age_range": f"{int(min(ages))}-{int(max(ages))}" if ages else "N/A",
            "average_age": round(np.mean(ages), 1) if ages else 0,
            "diversity_score": round(diversity_score, 1)
        }
    
    async def _analyze_audio_detailed(self, audio_path: str, duration: float,
                                    job_id: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Analyze audio content"""
        print(f"🎵 Analyzing audio: {audio_path}")
        
        if not self.whisper_model:
            return {
                "error": "Whisper model not available",
                "transcription": "",
                "speech_analysis": {}
            }
        
        try:
            # Transcribe audio
            if progress_callback:
                progress_callback(job_id, 65, "Transcribing audio...")
            
            result = self.whisper_model.transcribe(audio_path)
            transcription = result.get("text", "")
            segments = result.get("segments", [])
            
            if progress_callback:
                progress_callback(job_id, 75, "Analyzing speech patterns...")
            
            # Analyze speech content
            speech_analysis = self._analyze_speech_content(transcription, segments)
            
            # Analyze speaking time
            speaking_stats = self._calculate_speaking_statistics(segments, duration)
            
            # Detect bias in speech
            bias_analysis = self._detect_speech_bias(transcription)
            
            # Estimate speaker characteristics
            speaker_analysis = await self._analyze_speaker_characteristics(audio_path)
            
            return {
                "transcription": transcription,
                "transcription_segments": segments,
                "speech_analysis": speech_analysis,
                "speaking_statistics": speaking_stats,
                "bias_detection": bias_analysis,
                "speaker_analysis": speaker_analysis,
                "word_count": len(transcription.split()),
                "speech_duration": speech_analysis.get("total_speech_time", 0)
            }
            
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return {
                "error": str(e),
                "transcription": "",
                "speech_analysis": {}
            }
    
    def _analyze_speech_content(self, transcription: str, segments: List[Dict]) -> Dict[str, Any]:
        """Analyze speech content for patterns"""
        if not transcription:
            return {
                "pronoun_counts": {"male": 0, "female": 0, "neutral": 0},
                "speaking_segments": [],
                "total_speech_time": 0,
                "speech_density": 0
            }
        
        text_lower = transcription.lower()
        
        # Count pronouns
        pronoun_counts = {
            "male": sum(text_lower.count(word) for word in [" he ", " him ", " his ", " man ", " men "]),
            "female": sum(text_lower.count(word) for word in [" she ", " her ", " hers ", " woman ", " women "]),
            "neutral": sum(text_lower.count(word) for word in [" they ", " them ", " their ", " person ", " people "])
        }
        
        # Extract speaking segments
        speaking_segments = []
        total_speech_time = 0
        
        for seg in segments:
            if 'start' in seg and 'end' in seg:
                seg_duration = seg['end'] - seg['start']
                speaking_segments.append({
                    "start": seg['start'],
                    "end": seg['end'],
                    "duration": seg_duration,
                    "text": seg.get('text', '')[:150]  # First 150 chars
                })
                total_speech_time += seg_duration
        
        # Calculate speech density
        speech_density = total_speech_time / len(speaking_segments) if speaking_segments else 0
        
        return {
            "pronoun_counts": pronoun_counts,
            "speaking_segments": speaking_segments,
            "total_speech_time": total_speech_time,
            "speech_density": round(speech_density, 2),
            "segment_count": len(speaking_segments)
        }
    
    def _calculate_speaking_statistics(self, segments: List[Dict], total_duration: float) -> Dict[str, Any]:
        """Calculate speaking time statistics"""
        if not segments or total_duration <= 0:
            return {
                "speaking_percentage": 0,
                "silence_percentage": 100,
                "average_segment_duration": 0
            }
        
        speaking_time = 0
        segment_durations = []
        
        for seg in segments:
            if 'start' in seg and 'end' in seg:
                duration = seg['end'] - seg['start']
                speaking_time += duration
                segment_durations.append(duration)
        
        speaking_percentage = (speaking_time / total_duration) * 100
        
        stats = {
            "speaking_time_seconds": round(speaking_time, 2),
            "silence_time_seconds": round(total_duration - speaking_time, 2),
            "speaking_percentage": round(speaking_percentage, 2),
            "silence_percentage": round(100 - speaking_percentage, 2),
            "segment_count": len(segment_durations)
        }
        
        if segment_durations:
            stats.update({
                "average_segment_duration": round(np.mean(segment_durations), 2),
                "max_segment_duration": round(max(segment_durations), 2),
                "min_segment_duration": round(min(segment_durations), 2)
            })
        else:
            stats.update({
                "average_segment_duration": 0,
                "max_segment_duration": 0,
                "min_segment_duration": 0
            })
        
        return stats
    
    def _detect_speech_bias(self, transcription: str) -> Dict[str, Any]:
        """Detect bias in speech content"""
        if not transcription:
            return {
                "biased_phrases_found": [],
                "bias_by_category": {},
                "pronoun_counts": {"male": 0, "female": 0},
                "pronoun_imbalance_score": 0,
                "total_biased_phrases": 0,
                "has_bias": False
            }
        
        text_lower = transcription.lower()
        biased_phrases = []
        bias_categories = defaultdict(int)
        
        # Check for stereotype patterns
        for pattern in self.bias_config["stereotype_patterns"]:
            if pattern in text_lower:
                biased_phrases.append(pattern)
                # Categorize
                if any(word in pattern for word in ["woman", "female", "girl", "lady"]):
                    bias_categories["gender"] += 1
                elif any(word in pattern for word in ["man", "male", "boy", "gentleman"]):
                    bias_categories["gender"] += 1
                elif any(word in pattern for word in ["black", "asian", "mexican", "white", "racial"]):
                    bias_categories["racial"] += 1
                elif any(word in pattern for word in ["old", "young", "age"]):
                    bias_categories["age"] += 1
        
        # Check pronoun balance
        pronoun_counts = {
            "male": sum(text_lower.count(word) for word in [" he ", " him ", " his "]),
            "female": sum(text_lower.count(word) for word in [" she ", " her ", " hers "])
        }
        
        total_pronouns = pronoun_counts["male"] + pronoun_counts["female"]
        pronoun_imbalance = 0
        
        if total_pronouns > 0:
            male_ratio = pronoun_counts["male"] / total_pronouns
            female_ratio = pronoun_counts["female"] / total_pronouns
            pronoun_imbalance = abs(male_ratio - female_ratio)
        
        return {
            "biased_phrases_found": biased_phrases,
            "bias_by_category": dict(bias_categories),
            "pronoun_counts": pronoun_counts,
            "pronoun_imbalance_score": round(pronoun_imbalance, 3),
            "total_biased_phrases": len(biased_phrases),
            "has_bias": len(biased_phrases) > 0 or pronoun_imbalance > 0.6
        }
    
    async def _analyze_speaker_characteristics(self, audio_path: str) -> Dict[str, Any]:
        """Analyze speaker characteristics from audio"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Basic audio features
            features = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "rms_energy": np.sqrt(np.mean(y**2))
            }
            
            # Pitch analysis for gender estimation
            pitches = []
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, 
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )
                pitches = f0[voiced_flag]
            except:
                pass
            
            if len(pitches) > 0:
                avg_pitch = np.mean(pitches)
                # Simple gender estimation
                if avg_pitch < 120:  # Typically male
                    gender_estimate = {"male": 0.8, "female": 0.2}
                elif avg_pitch > 200:  # Typically female
                    gender_estimate = {"male": 0.2, "female": 0.8}
                else:  # Ambiguous
                    gender_estimate = {"male": 0.5, "female": 0.5}
                
                features["pitch_analysis"] = {
                    "average_pitch_hz": round(avg_pitch, 2),
                    "gender_estimate": gender_estimate,
                    "pitch_samples": len(pitches)
                }
            else:
                features["pitch_analysis"] = {
                    "gender_estimate": {"male": 0.5, "female": 0.5},
                    "message": "Insufficient pitch data"
                }
            
            # Speaking rate estimation
            try:
                onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
                speaking_rate = len(onsets) / features["duration"] if features["duration"] > 0 else 0
                features["speaking_rate"] = round(speaking_rate, 2)
            except:
                features["speaking_rate"] = 0
            
            return features
            
        except Exception as e:
            print(f"Speaker analysis error: {e}")
            return {
                "error": str(e),
                "gender_estimate": {"male": 0.5, "female": 0.5}
            }
    
    def _combine_video_analysis(self, face_results: Dict, audio_results: Dict,
                               video_info: Dict) -> Dict[str, Any]:
        """Combine face and audio analysis"""
        
        # Calculate component scores
        face_bias_score = self._calculate_face_bias_score(face_results)
        speech_bias_score = self._calculate_speech_bias_score(audio_results)
        presence_bias = self._calculate_presence_speech_bias(face_results, audio_results)
        
        # Overall weighted score
        overall_score = (
            face_bias_score * 0.4 +
            speech_bias_score * 0.4 +
            presence_bias * 0.2
        )
        
        # Risk level
        if overall_score >= 0.7:
            risk_level = "HIGH"
        elif overall_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Create timeline analysis
        timeline = self._create_timeline_analysis(
            face_results.get("face_timeline", []),
            audio_results.get("speech_analysis", {}).get("speaking_segments", []),
            video_info.get("duration", 0)
        )
        
        # Generate findings and recommendations
        key_findings = self._generate_key_findings(
            face_results, audio_results, overall_score, face_bias_score, speech_bias_score
        )
        
        recommendations = self._generate_recommendations(risk_level, key_findings)
        
        return {
            "overall_bias_score": round(overall_score, 3),
            "risk_level": risk_level,
            "component_scores": {
                "face_bias": round(face_bias_score, 3),
                "speech_bias": round(speech_bias_score, 3),
                "presence_speech_bias": round(presence_bias, 3)
            },
            "timeline": timeline,
            "key_findings": key_findings,
            "recommendations": recommendations,
            "summary": {
                "face_summary": face_results.get("demographic_summary", {}),
                "speech_summary": {
                    "has_bias": audio_results.get("bias_detection", {}).get("has_bias", False),
                    "biased_phrases": audio_results.get("bias_detection", {}).get("total_biased_phrases", 0),
                    "word_count": audio_results.get("word_count", 0)
                }
            }
        }
    
    def _calculate_face_bias_score(self, face_results: Dict) -> float:
        """Calculate bias score from face analysis"""
        stats = face_results.get("face_statistics", {})
        
        if not stats or "total_faces" not in stats or stats["total_faces"] == 0:
            return 0.3  # Neutral score for no faces
        
        score = 0.0
        
        # Gender imbalance
        gender_data = stats.get("gender", {})
        if gender_data:
            # Find the most represented gender
            max_gender_pct = 0
            for gender, data in gender_data.items():
                if isinstance(data, dict) and "percentage" in data:
                    max_gender_pct = max(max_gender_pct, data["percentage"])
            
            # Penalize high imbalance (>70% one gender)
            if max_gender_pct > 70:
                imbalance = (max_gender_pct - 70) / 30  # 0-1 based on 70-100%
                score += imbalance * 0.5
        
        # Racial diversity
        race_data = stats.get("race", {})
        if race_data:
            unique_races = len([r for r in race_data.keys() if r != "unknown"])
            if unique_races < 2:
                score += 0.3
        
        # Age diversity
        age_data = stats.get("age_groups", {})
        if age_data:
            represented_groups = len([a for a in age_data.keys() if age_data.get(a, {}).get("count", 0) > 0])
            if represented_groups < 2:
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_speech_bias_score(self, audio_results: Dict) -> float:
        """Calculate bias score from speech analysis"""
        bias_detection = audio_results.get("bias_detection", {})
        
        score = 0.0
        
        # Biased phrases
        biased_phrases = bias_detection.get("total_biased_phrases", 0)
        score += min(biased_phrases * 0.15, 0.5)  # Max 0.5 for phrases
        
        # Pronoun imbalance
        pronoun_imbalance = bias_detection.get("pronoun_imbalance_score", 0)
        if pronoun_imbalance > 0.6:
            score += 0.3
        
        # Speaking time concentration
        speaking_stats = audio_results.get("speaking_statistics", {})
        if speaking_stats.get("speaking_percentage", 0) > 85:
            score += 0.2  # Very dense speech
        
        return min(score, 1.0)
    
    # Add this to the end of bias_analyzer.py (after the broken method)

    def _calculate_presence_speech_bias(self, face_results: Dict, audio_results: Dict) -> float:
        """Calculate bias between who is present vs who is speaking"""
        # Get gender distributions
        face_gender_data = face_results.get("face_statistics", {}).get("gender", {})
        audio_gender_estimate = audio_results.get("speaker_analysis", {}).get("pitch_analysis", {}).get("gender_estimate", {})
        
        if not face_gender_data or not audio_gender_estimate:
            return 0.3  # Neutral if missing data
        
        # Extract percentages from face data
        face_male_pct = 0
        face_female_pct = 0
        
        for gender, data in face_gender_data.items():
            if isinstance(data, dict):
                pct = data.get("percentage", 0)
                if "man" in gender or "male" in gender:
                    face_male_pct += pct
                elif "woman" in gender or "female" in gender:
                    face_female_pct += pct
        
        # Normalize to 100%
        total_face = face_male_pct + face_female_pct
        if total_face > 0:
            face_male_pct = (face_male_pct / total_face) * 100
            face_female_pct = (face_female_pct / total_face) * 100
        
        # Get audio gender estimates
        audio_male_pct = audio_gender_estimate.get("male", 0.5) * 100
        audio_female_pct = audio_gender_estimate.get("female", 0.5) * 100
        
        # Calculate discrepancy
        male_discrepancy = abs(face_male_pct - audio_male_pct) / 100
        female_discrepancy = abs(face_female_pct - audio_female_pct) / 100
        
        # Average discrepancy
        avg_discrepancy = (male_discrepancy + female_discrepancy) / 2
        
        return min(avg_discrepancy, 1.0)
    
    def _create_timeline_analysis(self, face_timeline: List[Dict], 
                                 speech_segments: List[Dict], 
                                 total_duration: float) -> List[Dict]:
        """Create timeline analysis with synchronized data"""
        if total_duration <= 0:
            return []
        
        # Create time buckets (every 10 seconds)
        bucket_size = 10
        num_buckets = int(math.ceil(total_duration / bucket_size))
        timeline = []
        
        for i in range(num_buckets):
            start_time = i * bucket_size
            end_time = min((i + 1) * bucket_size, total_duration)
            
            # Count faces in this bucket
            faces_in_bucket = []
            for face_data in face_timeline:
                timestamp = face_data.get("timestamp", 0)
                if start_time <= timestamp < end_time:
                    faces_in_bucket.extend(face_data.get("faces", []))
            
            # Calculate speech in this bucket
            speech_in_bucket = 0
            for seg in speech_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                seg_duration = seg.get("duration", 0)
                
                # Check if segment overlaps with bucket
                if not (seg_end <= start_time or seg_start >= end_time):
                    overlap_start = max(seg_start, start_time)
                    overlap_end = min(seg_end, end_time)
                    speech_in_bucket += max(0, overlap_end - overlap_start)
            
            # Calculate bucket bias score
            bucket_score = 0.0
            reasons = []
            
            # Face vs speech discrepancy
            if len(faces_in_bucket) > 0 and speech_in_bucket == 0:
                bucket_score += 0.4
                reasons.append("Faces on screen but no speech")
            elif len(faces_in_bucket) == 0 and speech_in_bucket > 0:
                bucket_score += 0.3
                reasons.append("Speech but no faces on screen")
            
            # Check demographic balance in faces
            if len(faces_in_bucket) > 0:
                genders = [face.get("demographics", {}).get("gender", "unknown") 
                          for face in faces_in_bucket]
                male_count = sum(1 for g in genders if "man" in g or "male" in g)
                female_count = sum(1 for g in genders if "woman" in g or "female" in g)
                
                if male_count > 0 and female_count == 0:
                    bucket_score += 0.3
                    reasons.append("Only male faces on screen")
                elif female_count > 0 and male_count == 0:
                    bucket_score += 0.3
                    reasons.append("Only female faces on screen")
            
            timeline.append({
                "time_range": f"{start_time:.1f}-{end_time:.1f}s",
                "start_time": start_time,
                "end_time": end_time,
                "face_count": len(faces_in_bucket),
                "speech_duration": round(speech_in_bucket, 2),
                "bias_score": round(min(bucket_score, 1.0), 3),
                "risk_level": "HIGH" if bucket_score > 0.6 else "MEDIUM" if bucket_score > 0.3 else "LOW",
                "key_events": reasons[:3]  # Top 3 reasons
            })
        
        return timeline
    
    def _generate_key_findings(self, face_results: Dict, audio_results: Dict,
                              overall_score: float, face_score: float, speech_score: float) -> List[str]:
        """Generate key findings from analysis"""
        findings = []
        
        # Overall findings
        if overall_score > 0.7:
            findings.append("HIGH bias risk detected - significant improvements needed")
        elif overall_score > 0.4:
            findings.append("MODERATE bias risk - some areas need attention")
        else:
            findings.append("LOW bias risk - content appears balanced")
        
        # Face analysis findings
        face_stats = face_results.get("face_statistics", {})
        if face_stats.get("total_faces", 0) > 0:
            gender_data = face_stats.get("gender", {})
            if gender_data:
                # Check gender balance
                for gender, data in gender_data.items():
                    if isinstance(data, dict):
                        pct = data.get("percentage", 0)
                        if pct > 70:
                            findings.append(f"Gender imbalance: {gender} represents {pct:.1f}% of screen presence")
            
            # Check racial diversity
            race_data = face_stats.get("race", {})
            if race_data:
                unique_races = len([r for r in race_data.keys() if r != "unknown"])
                if unique_races < 2:
                    findings.append(f"Limited racial diversity: only {unique_races} race(s) represented")
        
        # Speech analysis findings
        bias_detection = audio_results.get("bias_detection", {})
        if bias_detection.get("has_bias", False):
            biased_count = bias_detection.get("total_biased_phrases", 0)
            if biased_count > 0:
                findings.append(f"Found {biased_count} potentially biased phrases in speech")
            
            pronoun_imbalance = bias_detection.get("pronoun_imbalance_score", 0)
            if pronoun_imbalance > 0.6:
                findings.append(f"Pronoun imbalance detected (score: {pronoun_imbalance:.2f})")
        
        # Presence vs speech findings
        presence_bias = self._calculate_presence_speech_bias(face_results, audio_results)
        if presence_bias > 0.5:
            findings.append("Significant discrepancy between who is on screen and who is speaking")
        
        # Add component scores
        findings.append(f"Face bias score: {face_score:.3f}")
        findings.append(f"Speech bias score: {speech_score:.3f}")
        findings.append(f"Presence-Speech bias: {presence_bias:.3f}")
        
        return findings[:10]  # Limit to top 10 findings
    
    def _generate_recommendations(self, risk_level: str, key_findings: List[str]) -> List[str]:
        """Generate recommendations based on risk level and findings"""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED",
                "1. Review and balance gender representation in visuals",
                "2. Remove or rephrase biased language from script",
                "3. Ensure diverse casting for all roles",
                "4. Balance speaking time across demographics",
                "5. Conduct bias sensitivity training for content team",
                "6. Implement diversity guidelines for future content",
                "7. Seek diverse perspectives in content review"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "⚠️ IMPROVEMENTS RECOMMENDED",
                "1. Improve demographic diversity in visuals",
                "2. Balance pronoun usage in narration",
                "3. Monitor speaking time distribution",
                "4. Review content for stereotype reinforcement",
                "5. Consider more inclusive language alternatives",
                "6. Test content with diverse audience groups"
            ])
        else:
            recommendations.extend([
                "✅ MAINTAIN GOOD PRACTICES",
                "1. Continue current balanced practices",
                "2. Regularly monitor for emerging bias patterns",
                "3. Consider incremental improvements in diversity",
                "4. Stay updated on bias detection research",
                "5. Share best practices with other teams"
            ])
        
        return recommendations
    
    async def _generate_visualizations(self, analysis_results: Dict, job_id: str) -> Dict[str, str]:
        """Generate visualization files and return their paths"""
        try:
            # Create visualizations directory
            vis_dir = Path("static") / "visualizations" / job_id
            vis_dir.mkdir(exist_ok=True, parents=True)
            
            vis_paths = {}
            
            # 1. Bias Timeline Chart
            timeline_data = analysis_results.get("timeline", [])
            if timeline_data:
                times = [f"{item['start_time']:.1f}" for item in timeline_data]
                bias_scores = [item["bias_score"] for item in timeline_data]
                face_counts = [item["face_count"] for item in timeline_data]
                
                fig = go.Figure()
                
                # Add bias score line
                fig.add_trace(go.Scatter(
                    x=times,
                    y=bias_scores,
                    mode='lines+markers',
                    name='Bias Score',
                    line=dict(color='red', width=2),
                    yaxis='y1'
                ))
                
                # Add face count bars
                fig.add_trace(go.Bar(
                    x=times,
                    y=face_counts,
                    name='Face Count',
                    yaxis='y2',
                    marker_color='blue',
                    opacity=0.6
                ))
                
                fig.update_layout(
                    title="Bias Timeline Analysis",
                    xaxis_title="Time (seconds)",
                    yaxis=dict(
                        title="Bias Score (0-1)",
                        range=[0, 1],
                        side='left'
                    ),
                    yaxis2=dict(
                        title="Face Count",
                        range=[0, max(face_counts) * 1.2 if face_counts else 10],
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(x=1.1, y=1)
                )
                
                timeline_path = vis_dir / "bias_timeline.html"
                fig.write_html(str(timeline_path))
                vis_paths["bias_timeline"] = f"/static/visualizations/{job_id}/bias_timeline.html"
            
            # 2. Demographics Pie Chart
            face_results = analysis_results.get("detailed_results", {}).get("face_analysis", {})
            face_stats = face_results.get("face_statistics", {})
            
            if face_stats.get("total_faces", 0) > 0:
                gender_data = face_stats.get("gender", {})
                if gender_data:
                    labels = []
                    values = []
                    colors = []
                    
                    color_map = {
                        "man": "#1f77b4", "male": "#1f77b4",
                        "woman": "#ff7f0e", "female": "#ff7f0e",
                        "unknown": "#d62728"
                    }
                    
                    for gender, data in gender_data.items():
                        if isinstance(data, dict):
                            labels.append(gender.capitalize())
                            values.append(data.get("percentage", 0))
                            colors.append(color_map.get(gender, "#7f7f7f"))
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3,
                        marker=dict(colors=colors)
                    )])
                    
                    fig.update_layout(title="Gender Distribution")
                    
                    gender_path = vis_dir / "gender_distribution.html"
                    fig.write_html(str(gender_path))
                    vis_paths["gender_distribution"] = f"/static/visualizations/{job_id}/gender_distribution.html"
            
            # 3. Speech Analysis Chart
            audio_results = analysis_results.get("detailed_results", {}).get("audio_analysis", {})
            bias_detection = audio_results.get("bias_detection", {})
            
            if bias_detection:
                categories = list(bias_detection.get("bias_by_category", {}).keys())
                counts = list(bias_detection.get("bias_by_category", {}).values())
                
                if categories:
                    fig = go.Figure(data=[go.Bar(
                        x=categories,
                        y=counts,
                        marker_color=['#e74c3c' if c > 0 else '#2ecc71' for c in counts]
                    )])
                    
                    fig.update_layout(
                        title="Bias Detection by Category",
                        xaxis_title="Bias Category",
                        yaxis_title="Count"
                    )
                    
                    bias_path = vis_dir / "bias_categories.html"
                    fig.write_html(str(bias_path))
                    vis_paths["bias_categories"] = f"/static/visualizations/{job_id}/bias_categories.html"
            
            return vis_paths
            
        except Exception as e:
            print(f"Visualization generation error: {e}")
            return {}
    
    # ========== IMAGE ANALYSIS ==========
    async def analyze_image_detailed(self, image_path: str, job_id: str, 
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Analyze image for bias"""
        print(f"🖼️ Analyzing image: {image_path}")
        
        if progress_callback:
            progress_callback(job_id, 20, "Loading image...")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            if progress_callback:
                progress_callback(job_id, 40, "Analyzing faces...")
            
            # Analyze faces in image
            face_results = await self._analyze_image_faces(image_path)
            
            if progress_callback:
                progress_callback(job_id, 70, "Analyzing composition...")
            
            # Analyze image composition
            composition_analysis = self._analyze_image_composition(image)
            
            if progress_callback:
                progress_callback(job_id, 90, "Generating results...")
            
            # Calculate bias score
            bias_score = self._calculate_image_bias_score(face_results, composition_analysis)
            
            risk_level = "HIGH" if bias_score > 0.7 else "MEDIUM" if bias_score > 0.4 else "LOW"
            
            return {
                "job_id": job_id,
                "media_type": "image",
                "timestamp": datetime.now().isoformat(),
                "image_info": {
                    "dimensions": f"{image.width}x{image.height}",
                    "format": image.format,
                    "size_mb": os.path.getsize(image_path) / (1024 * 1024)
                },
                "analysis_summary": {
                    "overall_bias_score": bias_score,
                    "risk_level": risk_level,
                    "total_faces": face_results.get("total_faces", 0),
                    "diversity_score": face_results.get("diversity_score", 0)
                },
                "detailed_results": {
                    "face_analysis": face_results,
                    "composition_analysis": composition_analysis
                },
                "key_findings": self._generate_image_findings(face_results, bias_score),
                "recommendations": self._generate_image_recommendations(risk_level)
            }
            
        except Exception as e:
            print(f"Image analysis error: {e}")
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_image_faces(self, image_path: str) -> Dict[str, Any]:
        """Analyze faces in image"""
        try:
            # Use DeepFace for face analysis
            face_analyses = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend='retinaface',
                enforce_detection=False,
                silent=True
            )
            
            if not isinstance(face_analyses, list):
                face_analyses = [face_analyses]
            
            faces = []
            for i, analysis in enumerate(face_analyses):
                face_info = {
                    "face_id": f"face_{i}",
                    "demographics": {
                        "age": analysis.get("age", 0),
                        "gender": analysis.get("dominant_gender", "unknown").lower(),
                        "race": analysis.get("dominant_race", "unknown").lower(),
                        "emotion": analysis.get("dominant_emotion", "neutral").lower(),
                        "confidence": analysis.get("face_confidence", 0)
                    }
                }
                faces.append(face_info)
            
            # Calculate statistics
            stats = self._calculate_face_statistics(faces)
            
            return {
                "total_faces": len(faces),
                "faces": faces,
                "statistics": stats,
                "diversity_score": self._calculate_diversity_score(faces)
            }
            
        except Exception as e:
            print(f"Image face analysis error: {e}")
            return {
                "total_faces": 0,
                "faces": [],
                "statistics": {},
                "diversity_score": 0
            }
    
    def _analyze_image_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image composition for bias indicators"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image data
        width, height = image.size
        total_pixels = width * height
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple color analysis (placeholder for more advanced analysis)
        avg_color = np.mean(img_array, axis=(0, 1))
        
        return {
            "dimensions": f"{width}x{height}",
            "aspect_ratio": width / height,
            "average_color": avg_color.tolist(),
            "pixel_count": total_pixels
        }
    
    def _calculate_image_bias_score(self, face_results: Dict, composition: Dict) -> float:
        """Calculate bias score for image"""
        score = 0.0
        
        # Face-based bias
        stats = face_results.get("statistics", {})
        total_faces = face_results.get("total_faces", 0)
        
        if total_faces > 0:
            gender_data = stats.get("gender", {})
            if gender_data:
                # Check gender balance
                for gender, data in gender_data.items():
                    if isinstance(data, dict):
                        pct = data.get("percentage", 0)
                        if pct > 80:  # Very high concentration of one gender
                            score += 0.6
                        elif pct > 60:
                            score += 0.3
            
            # Check racial diversity
            race_data = stats.get("race", {})
            if race_data:
                unique_races = len([r for r in race_data.keys() if r != "unknown"])
                if unique_races == 1:
                    score += 0.4
                elif unique_races < 2:
                    score += 0.2
        
        # No faces penalty (might indicate exclusion)
        elif total_faces == 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_image_findings(self, face_results: Dict, bias_score: float) -> List[str]:
        """Generate findings for image analysis"""
        findings = []
        
        total_faces = face_results.get("total_faces", 0)
        stats = face_results.get("statistics", {})
        
        if total_faces == 0:
            findings.append("No faces detected in image")
        else:
            findings.append(f"Detected {total_faces} face(s)")
            
            # Gender findings
            gender_data = stats.get("gender", {})
            if gender_data:
                for gender, data in gender_data.items():
                    if isinstance(data, dict):
                        pct = data.get("percentage", 0)
                        if pct > 70:
                            findings.append(f"Gender concentration: {gender} ({pct:.1f}%)")
            
            # Diversity finding
            diversity_score = face_results.get("diversity_score", 0)
            if diversity_score < 30:
                findings.append(f"Low diversity score: {diversity_score:.1f}/100")
        
        # Bias score finding
        if bias_score > 0.7:
            findings.append("High bias risk in image composition")
        elif bias_score > 0.4:
            findings.append("Moderate bias risk detected")
        else:
            findings.append("Low bias risk - image appears balanced")
        
        return findings
    
    def _generate_image_recommendations(self, risk_level: str) -> List[str]:
        """Generate recommendations for image"""
        if risk_level == "HIGH":
            return [
                "Consider more diverse representation in images",
                "Balance gender representation",
                "Include multiple racial/ethnic groups",
                "Review composition for exclusionary elements"
            ]
        elif risk_level == "MEDIUM":
            return [
                "Improve demographic diversity",
                "Consider more inclusive framing",
                "Balance visual representation"
            ]
        else:
            return [
                "Continue current practices",
                "Maintain diversity standards",
                "Regularly review for emerging patterns"
            ]
    
    # ========== AUDIO ANALYSIS (Standalone) ==========
    async def analyze_audio_detailed_standalone(self, audio_path: str, job_id: str,
                                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Standalone audio analysis"""
        return await self._analyze_audio_detailed(
            audio_path, 
            0,  # Duration unknown for standalone
            job_id, 
            progress_callback
        )
    
    def _calculate_diversity_score(self, faces: List[Dict]) -> float:
        """Calculate diversity score for faces (0-100)"""
        if not faces:
            return 0.0
        
        unique_genders = set()
        unique_races = set()
        age_groups = set()
        
        for face in faces:
            demo = face.get("demographics", {})
            unique_genders.add(demo.get("gender", "unknown"))
            unique_races.add(demo.get("race", "unknown"))
            
            age = demo.get("age", 0)
            if age < 20:
                age_groups.add("teen")
            elif age < 40:
                age_groups.add("young_adult")
            elif age < 60:
                age_groups.add("middle_age")
            else:
                age_groups.add("senior")
        
        # Score components (max 100)
        gender_score = (len(unique_genders) / 3) * 30  # Max 3 genders
        race_score = min(len(unique_races) / 5, 1) * 40  # Max 5 races
        age_score = (len(age_groups) / 4) * 30  # Max 4 age groups
        
        return gender_score + race_score + age_score

# Export the class
__all__ = ["MultimodalBiasAnalyzer"]

# Test function
if __name__ == "__main__":
    print("🧪 Testing Bias Analyzer...")
    analyzer = MultimodalBiasAnalyzer(model_size="base")
    print("✅ Bias Analyzer test complete!")
               