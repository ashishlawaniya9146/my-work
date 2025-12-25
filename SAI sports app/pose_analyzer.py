import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    def __init__(self):
        """Initialize MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_pose(self, frame):
        """
        Detect pose landmarks in a frame
        Returns: landmarks if detected, None otherwise
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                return results.pose_landmarks.landmark
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in pose detection: {str(e)}")
            return None
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        """
        try:
            # Convert to numpy arrays
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
            
        except Exception as e:
            logger.error(f"Error calculating angle: {str(e)}")
            return 0
    
    def get_body_landmarks(self, landmarks):
        """
        Extract key body landmarks for analysis
        """
        if not landmarks or len(landmarks) < 33:
            return None
        
        return {
            'nose': landmarks[0],
            'left_shoulder': landmarks[11],
            'right_shoulder': landmarks[12],
            'left_elbow': landmarks[13],
            'right_elbow': landmarks[14],
            'left_wrist': landmarks[15],
            'right_wrist': landmarks[16],
            'left_hip': landmarks[23],
            'right_hip': landmarks[24],
            'left_knee': landmarks[25],
            'right_knee': landmarks[26],
            'left_ankle': landmarks[27],
            'right_ankle': landmarks[28]
        }
    
    def analyze_vertical_jump(self, landmarks_sequence):
        """
        Analyze vertical jump from sequence of landmarks
        """
        if not landmarks_sequence:
            return None
        
        jump_heights = []
        for landmarks in landmarks_sequence:
            if landmarks:
                # Use hip position as reference for jump height
                hip_y = (landmarks[23].y + landmarks[24].y) / 2
                jump_heights.append(1.0 - hip_y)  # Invert Y coordinate
        
        if not jump_heights:
            return None
        
        max_height = max(jump_heights)
        min_height = min(jump_heights)
        jump_range = max_height - min_height
        
        return {
            'max_height': max_height,
            'min_height': min_height,
            'jump_range': jump_range,
            'estimated_height_cm': jump_range * 100
        }
    
    def analyze_sit_ups(self, landmarks_sequence):
        """
        Analyze sit-ups from sequence of landmarks
        """
        if not landmarks_sequence:
            return None
        
        torso_angles = []
        for landmarks in landmarks_sequence:
            if landmarks and len(landmarks) > 24:
                # Calculate torso angle using shoulder and hip
                shoulder = landmarks[11]  # Left shoulder
                hip = landmarks[23]       # Left hip
                knee = landmarks[25]      # Left knee
                
                angle = self.calculate_angle(shoulder, hip, knee)
                torso_angles.append(angle)
        
        if not torso_angles:
            return None
        
        # Count repetitions by detecting angle changes
        rep_count = 0
        threshold = np.mean(torso_angles)
        
        for i in range(1, len(torso_angles) - 1):
            if torso_angles[i-1] > threshold and torso_angles[i] < threshold:
                rep_count += 1
        
        return {
            'rep_count': rep_count // 2,  # Divide by 2 for full cycles
            'avg_angle': np.mean(torso_angles),
            'angle_consistency': 100 - (np.std(torso_angles) * 2)
        }
    
    def analyze_running_form(self, landmarks_sequence):
        """
        Analyze running form from sequence of landmarks
        """
        if not landmarks_sequence:
            return None
        
        stride_lengths = []
        knee_lifts = []
        
        for landmarks in landmarks_sequence:
            if landmarks and len(landmarks) > 28:
                # Calculate stride length (distance between feet)
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                stride_length = abs(left_ankle.x - right_ankle.x)
                stride_lengths.append(stride_length)
                
                # Calculate knee lift
                left_knee = landmarks[25]
                right_knee = landmarks[26]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                
                left_knee_lift = abs(left_knee.y - left_hip.y)
                right_knee_lift = abs(right_knee.y - right_hip.y)
                knee_lifts.append((left_knee_lift + right_knee_lift) / 2)
        
        if not stride_lengths or not knee_lifts:
            return None
        
        return {
            'avg_stride_length': np.mean(stride_lengths),
            'stride_consistency': 100 - (np.std(stride_lengths) * 100),
            'avg_knee_lift': np.mean(knee_lifts),
            'form_stability': 100 - (np.std(knee_lifts) * 100)
        }
    
    def detect_cheating_patterns(self, landmarks_sequence, test_type):
        """
        Detect potential cheating patterns in the exercise
        """
        if not landmarks_sequence:
            return {'authentic': True, 'confidence': 50, 'notes': 'Insufficient data'}
        
        # Basic authenticity checks
        valid_frames = sum(1 for landmarks in landmarks_sequence if landmarks)
        total_frames = len(landmarks_sequence)
        
        if total_frames == 0:
            return {'authentic': False, 'confidence': 0, 'notes': 'No valid frames detected'}
        
        detection_rate = valid_frames / total_frames
        
        if detection_rate < 0.3:
            return {
                'authentic': False, 
                'confidence': int(detection_rate * 100),
                'notes': 'Poor pose detection - possible video quality issues'
            }
        
        # Test-specific cheating detection
        if test_type == 'vertical-jump':
            return self._detect_jump_cheating(landmarks_sequence)
        elif test_type == 'sit-ups':
            return self._detect_situp_cheating(landmarks_sequence)
        else:
            return {
                'authentic': True,
                'confidence': min(95, int(detection_rate * 100)),
                'notes': 'Basic authenticity checks passed'
            }
    
    def _detect_jump_cheating(self, landmarks_sequence):
        """Detect cheating in vertical jump"""
        movement_detected = False
        significant_changes = 0
        
        prev_hip_y = None
        for landmarks in landmarks_sequence:
            if landmarks and len(landmarks) > 24:
                current_hip_y = (landmarks[23].y + landmarks[24].y) / 2
                
                if prev_hip_y is not None:
                    change = abs(current_hip_y - prev_hip_y)
                    if change > 0.05:  # Significant movement threshold
                        significant_changes += 1
                        movement_detected = True
                
                prev_hip_y = current_hip_y
        
        if not movement_detected:
            return {
                'authentic': False,
                'confidence': 20,
                'notes': 'No jumping movement detected'
            }
        
        confidence = min(95, 60 + (significant_changes * 2))
        return {
            'authentic': True,
            'confidence': confidence,
            'notes': f'Jump movement verified - {significant_changes} significant movements detected'
        }
    
    def _detect_situp_cheating(self, landmarks_sequence):
        """Detect cheating in sit-ups"""
        torso_movements = 0
        prev_angle = None
        
        for landmarks in landmarks_sequence:
            if landmarks and len(landmarks) > 25:
                shoulder = landmarks[11]
                hip = landmarks[23]
                knee = landmarks[25]
                
                current_angle = self.calculate_angle(shoulder, hip, knee)
                
                if prev_angle is not None:
                    angle_change = abs(current_angle - prev_angle)
                    if angle_change > 10:  # Significant angle change
                        torso_movements += 1
                
                prev_angle = current_angle
        
        if torso_movements < 5:
            return {
                'authentic': False,
                'confidence': 30,
                'notes': 'Insufficient torso movement for sit-ups'
            }
        
        confidence = min(95, 50 + (torso_movements))
        return {
            'authentic': True,
            'confidence': confidence,
            'notes': f'Sit-up movement verified - {torso_movements} torso movements detected'
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'pose'):
            self.pose.close()