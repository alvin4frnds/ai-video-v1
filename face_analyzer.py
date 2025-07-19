import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import os
from PIL import Image

class FaceAnalyzer:
    """Face detection and analysis for image quality assessment"""
    
    def __init__(self):
        self.face_cascade = None
        self.mp_face_detection = None
        self.mp_drawing = None
        
        # Initialize face detection methods
        self._init_opencv_detector()
        self._init_mediapipe_detector()
        
        logging.info("FaceAnalyzer initialized")
    
    def _init_opencv_detector(self):
        """Initialize OpenCV Haar Cascade face detector"""
        try:
            # Try to load the face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logging.warning("OpenCV face cascade not loaded properly")
                self.face_cascade = None
            else:
                logging.info("OpenCV face detector initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize OpenCV face detector: {str(e)}")
            self.face_cascade = None
    
    def _init_mediapipe_detector(self):
        """Initialize MediaPipe face detector"""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 0 for close-range, 1 for full-range
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            logging.info("MediaPipe face detector initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize MediaPipe: {str(e)}")
            self.mp_face_detection = None
    
    def detect_faces(self, image_path: str) -> Dict:
        """Detect faces in image using multiple methods"""
        try:
            filename = os.path.basename(image_path)
            logging.debug(f"🔍 Loading image for face detection: {filename}")
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"❌ Could not load image: {filename}")
                return {'error': 'Could not load image', 'face_count': 0}
            
            img_height, img_width = img.shape[:2]
            img_size = f"{img_width}x{img_height}"
            logging.debug(f"📐 Image loaded: {img_size}, size: {os.path.getsize(image_path)/1024:.1f} KB")
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            results = {
                'face_count': 0,
                'faces': [],
                'quality_score': 0.0,
                'has_realistic_face': False,
                'detection_methods': []
            }
            
            # MediaPipe detection (primary)
            logging.debug(f"🤖 Running MediaPipe face detection...")
            mp_faces = self._detect_faces_mediapipe(rgb_img)
            if mp_faces:
                results['faces'].extend(mp_faces)
                results['detection_methods'].append('MediaPipe')
                logging.debug(f"   ✅ MediaPipe found {len(mp_faces)} face(s)")
            else:
                logging.debug(f"   ❌ MediaPipe found no faces")
            
            # OpenCV detection (fallback/validation)
            logging.debug(f"🔧 Running OpenCV face detection...")
            cv_faces = self._detect_faces_opencv(gray_img, img.shape)
            if cv_faces:
                results['detection_methods'].append('OpenCV')
                # Merge with MediaPipe results (avoid duplicates)
                filtered_cv_faces = self._filter_duplicate_faces(results['faces'], cv_faces)
                results['faces'].extend(filtered_cv_faces)
                logging.debug(f"   ✅ OpenCV found {len(cv_faces)} face(s), {len(filtered_cv_faces)} new")
            else:
                logging.debug(f"   ❌ OpenCV found no faces")
            
            results['face_count'] = len(results['faces'])
            
            if results['face_count'] > 0:
                logging.debug(f"📊 Calculating face quality for {results['face_count']} face(s)...")
                results['quality_score'] = self._calculate_face_quality(results['faces'], img.shape)
                results['has_realistic_face'] = results['quality_score'] > 0.6
                
                quality_emoji = "🟢" if results['quality_score'] > 0.7 else "🟡" if results['quality_score'] > 0.4 else "🔴"
                logging.debug(f"   {quality_emoji} Quality score: {results['quality_score']:.2f}")
            else:
                logging.debug(f"⚪ No faces detected in image")
            
            return results
            
        except Exception as e:
            logging.error(f"💥 Error in face detection for {os.path.basename(image_path)}: {str(e)}")
            return {'error': str(e), 'face_count': 0, 'quality_score': 0.0}
    
    def _detect_faces_mediapipe(self, rgb_img: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        if not self.mp_face_detection:
            return []
        
        try:
            results = self.mp_face_detection.process(rgb_img)
            faces = []
            
            if results.detections:
                h, w = rgb_img.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    face_data = {
                        'x': x, 'y': y, 'width': width, 'height': height,
                        'confidence': detection.score[0] if detection.score else 0.8,
                        'method': 'MediaPipe',
                        'area': width * height
                    }
                    faces.append(face_data)
            
            return faces
        except Exception as e:
            logging.warning(f"MediaPipe detection failed: {str(e)}")
            return []
    
    def _detect_faces_opencv(self, gray_img: np.ndarray, img_shape: Tuple) -> List[Dict]:
        """Detect faces using OpenCV"""
        if not self.face_cascade:
            return []
        
        try:
            faces_rect = self.face_cascade.detectMultiScale(
                gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            faces = []
            for (x, y, w, h) in faces_rect:
                face_data = {
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'confidence': 0.8,  # OpenCV doesn't provide confidence
                    'method': 'OpenCV',
                    'area': w * h
                }
                faces.append(face_data)
            
            return faces
        except Exception as e:
            logging.warning(f"OpenCV detection failed: {str(e)}")
            return []
    
    def _filter_duplicate_faces(self, existing_faces: List[Dict], new_faces: List[Dict]) -> List[Dict]:
        """Filter out duplicate face detections"""
        filtered = []
        
        for new_face in new_faces:
            is_duplicate = False
            for existing_face in existing_faces:
                # Calculate overlap
                overlap = self._calculate_bbox_overlap(new_face, existing_face)
                if overlap > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(new_face)
        
        return filtered
    
    def _calculate_bbox_overlap(self, face1: Dict, face2: Dict) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_min, y1_min = face1['x'], face1['y']
        x1_max, y1_max = x1_min + face1['width'], y1_min + face1['height']
        
        x2_min, y2_min = face2['x'], face2['y']
        x2_max, y2_max = x2_min + face2['width'], y2_min + face2['height']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union = face1['area'] + face2['area'] - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_face_quality(self, faces: List[Dict], img_shape: Tuple) -> float:
        """Calculate overall face quality score"""
        if not faces:
            return 0.0
        
        img_area = img_shape[0] * img_shape[1]
        scores = []
        
        for face in faces:
            # Size score (prefer faces that are 5-25% of image)
            face_ratio = face['area'] / img_area
            if 0.05 <= face_ratio <= 0.25:
                size_score = 1.0
            elif 0.02 <= face_ratio <= 0.5:
                size_score = 0.7
            else:
                size_score = 0.3
            
            # Confidence score
            confidence_score = face.get('confidence', 0.8)
            
            # Position score (prefer faces not at edges)
            img_h, img_w = img_shape[:2]
            center_x = face['x'] + face['width'] / 2
            center_y = face['y'] + face['height'] / 2
            
            # Distance from center (normalized)
            center_dist = np.sqrt(((center_x - img_w/2) / img_w)**2 + ((center_y - img_h/2) / img_h)**2)
            position_score = max(0.3, 1.0 - center_dist)
            
            # Combined score
            face_score = (size_score * 0.4 + confidence_score * 0.4 + position_score * 0.2)
            scores.append(face_score)
        
        return max(scores) if scores else 0.0
    
    def calculate_face_compatibility(self, face_data_list: List[Dict]) -> float:
        """Calculate compatibility score across multiple face detections"""
        if len(face_data_list) < 2:
            return 1.0
        
        valid_faces = [fd for fd in face_data_list if fd.get('face_count', 0) > 0]
        if len(valid_faces) < 2:
            return 0.5
        
        # Extract face features for comparison
        face_features = []
        for face_data in valid_faces:
            if face_data.get('faces'):
                # Use largest face from each image
                largest_face = max(face_data['faces'], key=lambda f: f['area'])
                features = self._extract_face_features(largest_face)
                face_features.append(features)
        
        if len(face_features) < 2:
            return 0.5
        
        # Calculate similarity between consecutive faces
        similarities = []
        for i in range(len(face_features) - 1):
            similarity = self._calculate_feature_similarity(face_features[i], face_features[i + 1])
            similarities.append(similarity)
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0.5
    
    def _extract_face_features(self, face: Dict) -> Dict:
        """Extract basic features from face detection"""
        return {
            'aspect_ratio': face['width'] / face['height'] if face['height'] > 0 else 1.0,
            'relative_size': face['area'],
            'position_x': face['x'] + face['width'] / 2,
            'position_y': face['y'] + face['height'] / 2,
            'confidence': face.get('confidence', 0.8)
        }
    
    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between face features"""
        # Aspect ratio similarity
        ar_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
        ar_score = max(0, 1.0 - ar_diff / 0.5)  # Normalize by acceptable difference
        
        # Size similarity (relative)
        size_ratio = min(features1['relative_size'], features2['relative_size']) / max(features1['relative_size'], features2['relative_size'])
        size_score = size_ratio
        
        # Position similarity (less important for different scenes)
        pos_score = 0.8  # Default good score since faces can move between scenes
        
        # Confidence similarity
        conf_score = min(features1['confidence'], features2['confidence'])
        
        # Weighted average
        similarity = (ar_score * 0.3 + size_score * 0.3 + pos_score * 0.2 + conf_score * 0.2)
        return similarity