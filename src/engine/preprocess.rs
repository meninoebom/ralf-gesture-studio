//! Frame preprocessing pipeline for gesture recognition.
//!
//! Transforms raw skeleton frames into normalized, feature-enriched representations
//! before DTW comparison. Each stage is independently toggleable:
//!
//! 1. **Hip centering** — subtract hip-center position from all joints (position invariance)
//! 2. **Scale normalization** — divide by shoulder width (body-size invariance)
//! 3. **Velocity features** — append first derivative per dimension (dynamics capture)
//!
//! The pipeline sits between OSC input and the frame buffer, ensuring both
//! training and recognition paths see identical preprocessing.
//!
//! ## Architecture
//!
//! - `PreprocessingConfig` — serializable toggle struct, stored per-vocabulary
//! - `Preprocessor` — stateful runtime that applies the configured pipeline

use serde::{Deserialize, Serialize};

/// Minimum shoulder width to prevent division by zero during scale normalization.
/// If measured shoulder width falls below this, we clamp and log a warning.
const MIN_SHOULDER_WIDTH: f32 = 0.01;

/// Configuration for the preprocessing pipeline. Stored per-vocabulary in the .ralf file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreprocessingConfig {
    /// Subtract hip-center position from all joints (position invariance)
    #[serde(default)]
    pub hip_normalize: bool,
    /// Divide all coordinates by shoulder width (body-size invariance)
    #[serde(default)]
    pub scale_normalize: bool,
    /// Append first-derivative (velocity) features, doubling frame dimensionality
    #[serde(default)]
    pub velocity_features: bool,
    /// Append joint angle features (12 angles at major joints, radians)
    #[serde(default)]
    pub angle_features: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            hip_normalize: true,
            scale_normalize: true,
            velocity_features: false,
            angle_features: false,
        }
    }
}

/// Joint index layout for a specific tracking system.
/// Maps semantic joint roles to indices in the flat frame array.
#[derive(Debug, Clone)]
pub struct TrackingLayout {
    /// Number of raw input dimensions (e.g., 66 for 33 joints x XY)
    pub raw_dimensions: usize,
    /// Number of coordinates per joint (2 for XY, 3 for XYZ)
    pub coords_per_joint: usize,
    /// Index of left hip joint (joint number, not array index)
    pub left_hip: usize,
    /// Index of right hip joint
    pub right_hip: usize,
    /// Index of left shoulder joint
    pub left_shoulder: usize,
    /// Index of right shoulder joint
    pub right_shoulder: usize,
    // Angle feature joints
    pub left_elbow: usize,
    pub right_elbow: usize,
    pub left_wrist: usize,
    pub right_wrist: usize,
    pub left_knee: usize,
    pub right_knee: usize,
    pub left_ankle: usize,
    pub right_ankle: usize,
    pub left_index: usize,
    pub right_index: usize,
}

impl TrackingLayout {
    /// MediaPipe Pose 33-keypoint layout with XY coordinates (66 floats).
    pub fn mediapipe_pose_33_xy() -> Self {
        Self {
            raw_dimensions: 66,
            coords_per_joint: 2,
            left_hip: 23,
            right_hip: 24,
            left_shoulder: 11,
            right_shoulder: 12,
            left_elbow: 13,
            right_elbow: 14,
            left_wrist: 15,
            right_wrist: 16,
            left_knee: 25,
            right_knee: 26,
            left_ankle: 27,
            right_ankle: 28,
            left_index: 19,
            right_index: 20,
        }
    }

    /// Get the starting array index for a joint (joint_index * coords_per_joint).
    fn joint_offset(&self, joint: usize) -> usize {
        joint * self.coords_per_joint
    }

    /// Number of joints in this layout.
    fn joint_count(&self) -> usize {
        self.raw_dimensions / self.coords_per_joint
    }
}

/// Resolve a `TrackingLayout` from the vocabulary's `tracking_system` string.
/// Returns `None` for unknown tracking systems (preprocessing will be skipped).
pub fn layout_for_tracking_system(tracking_system: &str) -> Option<TrackingLayout> {
    match tracking_system {
        "mediapipe-pose-33-xy" => Some(TrackingLayout::mediapipe_pose_33_xy()),
        _ => None,
    }
}

/// Stateful runtime preprocessor. Holds previous frame for velocity computation.
#[derive(Debug)]
pub struct Preprocessor {
    config: PreprocessingConfig,
    layout: Option<TrackingLayout>,
    prev_frame: Option<Vec<f32>>,
}

impl Preprocessor {
    /// Create a new preprocessor with the given config and tracking system.
    pub fn new(config: PreprocessingConfig, tracking_system: &str) -> Self {
        let layout = layout_for_tracking_system(tracking_system);
        if layout.is_none() && (config.hip_normalize || config.scale_normalize) {
            eprintln!(
                "Warning: Unknown tracking system '{}'. Hip/scale normalization disabled.",
                tracking_system
            );
        }
        Self {
            config,
            layout,
            prev_frame: None,
        }
    }

    /// Create a passthrough preprocessor (all toggles off).
    #[allow(dead_code)]
    pub fn passthrough() -> Self {
        Self {
            config: PreprocessingConfig::default(),
            layout: None,
            prev_frame: None,
        }
    }

    /// Returns the current config.
    #[allow(dead_code)]
    pub fn config(&self) -> &PreprocessingConfig {
        &self.config
    }

    /// Update the config (e.g., when user toggles a feature).
    #[allow(dead_code)]
    pub fn set_config(&mut self, config: PreprocessingConfig) {
        self.config = config;
        self.prev_frame = None; // Reset velocity state on config change
    }

    /// Returns true if the pipeline will change frame dimensionality.
    #[allow(dead_code)]
    pub fn changes_dimensions(&self) -> bool {
        self.config.velocity_features || self.config.angle_features
    }

    /// Number of angle features appended (12 for MediaPipe).
    const NUM_ANGLES: usize = 12;

    /// Returns the output dimension count for a given raw input dimension count.
    #[allow(dead_code)]
    pub fn output_dimensions(&self, raw_dims: usize) -> usize {
        let with_angles = if self.config.angle_features {
            raw_dims + Self::NUM_ANGLES
        } else {
            raw_dims
        };
        if self.config.velocity_features {
            with_angles * 2
        } else {
            with_angles
        }
    }

    /// Reset stateful components (call when starting a new recording or recognition session).
    pub fn reset(&mut self) {
        self.prev_frame = None;
    }

    /// Process a single raw frame. Returns the preprocessed frame.
    ///
    /// For the first frame after reset, velocity features are zero-padded.
    pub fn process_frame(&mut self, raw: &[f32]) -> Vec<f32> {
        let mut frame = raw.to_vec();

        // Stage 1: Hip centering
        if self.config.hip_normalize {
            self.apply_hip_centering(&mut frame);
        }

        // Stage 2: Scale normalization
        if self.config.scale_normalize {
            self.apply_scale_normalization(&mut frame);
        }

        // Stage 3: Angle features (append before velocity so velocity covers angles too)
        if self.config.angle_features {
            self.apply_angle_features(&mut frame);
        }

        // Stage 4: Velocity features
        if self.config.velocity_features {
            frame = self.apply_velocity_features(&frame);
        } else {
            // Still track prev_frame for when velocity is toggled on later
            self.prev_frame = Some(raw.to_vec());
        }

        frame
    }

    /// Process a stored sequence (e.g., training examples at load time).
    /// Handles first-frame velocity boundary condition internally.
    /// This is stateless — does not affect or use `self.prev_frame`.
    pub fn process_sequence(&self, raw: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if raw.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(raw.len());
        let mut prev: Option<&Vec<f32>> = None;

        for raw_frame in raw {
            let mut frame = raw_frame.clone();

            if self.config.hip_normalize {
                self.apply_hip_centering(&mut frame);
            }

            if self.config.scale_normalize {
                self.apply_scale_normalization(&mut frame);
            }

            if self.config.angle_features {
                self.apply_angle_features(&mut frame);
            }

            if self.config.velocity_features {
                let velocity: Vec<f32> = match prev {
                    Some(p) => frame.iter().zip(p.iter()).map(|(c, p)| c - p).collect(),
                    None => vec![0.0; frame.len()],
                };
                let mut combined = frame.clone();
                combined.extend_from_slice(&velocity);
                result.push(combined);
            } else {
                result.push(frame);
            }

            prev = Some(raw_frame);
        }

        result
    }

    /// Subtract hip-center position from all joint coordinates.
    fn apply_hip_centering(&self, frame: &mut [f32]) {
        let layout = match &self.layout {
            Some(l) => l,
            None => return, // No layout = skip
        };

        if frame.len() < layout.raw_dimensions {
            return; // Frame too short for this layout
        }

        let coords = layout.coords_per_joint;
        let lh = layout.joint_offset(layout.left_hip);
        let rh = layout.joint_offset(layout.right_hip);

        // Hip center = average of left and right hip
        let mut hip_center = vec![0.0f32; coords];
        for c in 0..coords {
            hip_center[c] = (frame[lh + c] + frame[rh + c]) / 2.0;
        }

        // Subtract hip center from all joints
        let n_joints = layout.joint_count();
        for j in 0..n_joints {
            let offset = j * coords;
            for c in 0..coords {
                if offset + c < frame.len() {
                    frame[offset + c] -= hip_center[c];
                }
            }
        }
    }

    /// Divide all coordinates by shoulder width for body-size invariance.
    fn apply_scale_normalization(&self, frame: &mut [f32]) {
        let layout = match &self.layout {
            Some(l) => l,
            None => return,
        };

        if frame.len() < layout.raw_dimensions {
            return;
        }

        let coords = layout.coords_per_joint;
        let ls = layout.joint_offset(layout.left_shoulder);
        let rs = layout.joint_offset(layout.right_shoulder);

        // Shoulder width = Euclidean distance between left and right shoulder
        let mut dist_sq = 0.0f32;
        for c in 0..coords {
            let diff = frame[ls + c] - frame[rs + c];
            dist_sq += diff * diff;
        }
        let shoulder_width = dist_sq.sqrt();

        let scale = if shoulder_width < MIN_SHOULDER_WIDTH {
            eprintln!(
                "Warning: Shoulder width {:.4} below minimum {:.4}, clamping.",
                shoulder_width, MIN_SHOULDER_WIDTH
            );
            1.0 / MIN_SHOULDER_WIDTH
        } else {
            1.0 / shoulder_width
        };

        for val in frame.iter_mut() {
            *val *= scale;
        }
    }

    /// Append velocity (first derivative) features to the frame.
    /// Returns a new frame with doubled dimensionality: [positions..., velocities...].
    fn apply_velocity_features(&mut self, frame: &[f32]) -> Vec<f32> {
        let velocity: Vec<f32> = match &self.prev_frame {
            Some(prev) => frame.iter().zip(prev.iter()).map(|(c, p)| c - p).collect(),
            None => vec![0.0; frame.len()],
        };

        // Store current (pre-velocity) frame for next iteration.
        // We store the normalized position frame, not the raw input,
        // so velocity reflects changes in normalized space.
        self.prev_frame = Some(frame.to_vec());

        let mut combined = frame.to_vec();
        combined.extend_from_slice(&velocity);
        combined
    }

    /// Compute 12 joint angles and append to frame.
    /// Angles: shoulder, elbow, hip, knee, wrist (left + right) + 2 torso inclinations.
    fn apply_angle_features(&self, frame: &mut Vec<f32>) {
        let layout = match &self.layout {
            Some(l) => l,
            None => return,
        };
        if frame.len() < layout.raw_dimensions {
            return;
        }

        let angles = Self::compute_angles(frame, layout);
        frame.extend_from_slice(&angles);
    }

    /// Compute 12 joint angles from XY coordinates.
    /// Each angle is the angle at joint B in the chain A→B→C, in radians [0, π].
    fn compute_angles(frame: &[f32], layout: &TrackingLayout) -> Vec<f32> {
        let joint = |idx: usize| -> (f32, f32) {
            let o = layout.joint_offset(idx);
            (frame[o], frame[o + 1])
        };

        // 12 angle triplets: (A, B, C) where angle is measured at B
        let triplets: [(usize, usize, usize); 12] = [
            // Left shoulder: hip → shoulder → elbow
            (layout.left_hip, layout.left_shoulder, layout.left_elbow),
            // Right shoulder: hip → shoulder → elbow
            (layout.right_hip, layout.right_shoulder, layout.right_elbow),
            // Left elbow: shoulder → elbow → wrist
            (layout.left_shoulder, layout.left_elbow, layout.left_wrist),
            // Right elbow: shoulder → elbow → wrist
            (layout.right_shoulder, layout.right_elbow, layout.right_wrist),
            // Left hip: shoulder → hip → knee
            (layout.left_shoulder, layout.left_hip, layout.left_knee),
            // Right hip: shoulder → hip → knee
            (layout.right_shoulder, layout.right_hip, layout.right_knee),
            // Left knee: hip → knee → ankle
            (layout.left_hip, layout.left_knee, layout.left_ankle),
            // Right knee: hip → knee → ankle
            (layout.right_hip, layout.right_knee, layout.right_ankle),
            // Left wrist: elbow → wrist → index
            (layout.left_elbow, layout.left_wrist, layout.left_index),
            // Right wrist: elbow → wrist → index
            (layout.right_elbow, layout.right_wrist, layout.right_index),
            // Torso left: left_shoulder → left_hip (vertical angle via synthetic vertical point)
            // We use right_hip as proxy for "below left hip" direction
            (layout.left_shoulder, layout.left_hip, layout.right_hip),
            // Torso right: right_shoulder → right_hip → left_hip
            (layout.right_shoulder, layout.right_hip, layout.left_hip),
        ];

        triplets
            .iter()
            .map(|&(a_idx, b_idx, c_idx)| {
                let a = joint(a_idx);
                let b = joint(b_idx);
                let c = joint(c_idx);
                angle_at_b(a, b, c)
            })
            .collect()
    }
}

/// Compute the angle at point B given triangle A-B-C using dot product.
/// Returns radians in [0, π].
fn angle_at_b(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    let ba = (a.0 - b.0, a.1 - b.1);
    let bc = (c.0 - b.0, c.1 - b.1);

    let dot = ba.0 * bc.0 + ba.1 * bc.1;
    let mag_ba = (ba.0 * ba.0 + ba.1 * ba.1).sqrt();
    let mag_bc = (bc.0 * bc.0 + bc.1 * bc.1).sqrt();

    if mag_ba < 1e-8 || mag_bc < 1e-8 {
        return std::f32::consts::PI; // Degenerate: straight
    }

    let cos_angle = (dot / (mag_ba * mag_bc)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple 6-float frame (3 joints x XY) for a mini layout.
    fn mini_layout() -> TrackingLayout {
        // 3 joints: joint 0 = "left shoulder", joint 1 = "right shoulder", joint 2 = "hip"
        TrackingLayout {
            raw_dimensions: 6,
            coords_per_joint: 2,
            left_hip: 2,
            right_hip: 2,
            left_shoulder: 0,
            right_shoulder: 1,
            // Mini layout doesn't have these joints — angle tests use full MediaPipe layout
            left_elbow: 0,
            right_elbow: 0,
            left_wrist: 0,
            right_wrist: 0,
            left_knee: 0,
            right_knee: 0,
            left_ankle: 0,
            right_ankle: 0,
            left_index: 0,
            right_index: 0,
        }
    }

    #[test]
    fn test_passthrough_no_change() {
        let mut p = Preprocessor::passthrough();
        let frame = vec![1.0, 2.0, 3.0, 4.0];
        let result = p.process_frame(&frame);
        assert_eq!(result, frame);
    }

    #[test]
    fn test_hip_centering() {
        // MediaPipe layout: 33 joints x 2 = 66 floats
        // Left hip = joint 23 (indices 46,47), Right hip = joint 24 (indices 48,49)
        let config = PreprocessingConfig {
            hip_normalize: true,
            scale_normalize: false,
            velocity_features: false,
            angle_features: false,
        };
        let mut p = Preprocessor::new(config, "mediapipe-pose-33-xy");

        let mut frame = vec![0.0f32; 66];
        // Set left hip at (0.5, 0.6)
        frame[46] = 0.5;
        frame[47] = 0.6;
        // Set right hip at (0.5, 0.6) — same position for simple test
        frame[48] = 0.5;
        frame[49] = 0.6;
        // Set joint 0 (nose) at (0.5, 0.8)
        frame[0] = 0.5;
        frame[1] = 0.8;

        let result = p.process_frame(&frame);

        // Hip center = (0.5, 0.6)
        // Nose should be (0.5 - 0.5, 0.8 - 0.6) = (0.0, 0.2)
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.2).abs() < 1e-6);
        // Hip joints should be at origin
        assert!((result[46] - 0.0).abs() < 1e-6);
        assert!((result[47] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_normalization() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: true,
            velocity_features: false,
            angle_features: false,
        };
        let mut p = Preprocessor::new(config, "mediapipe-pose-33-xy");

        let mut frame = vec![0.0f32; 66];
        // Left shoulder (joint 11) at (0.3, 0.5)
        frame[22] = 0.3;
        frame[23] = 0.5;
        // Right shoulder (joint 12) at (0.7, 0.5)
        frame[24] = 0.7;
        frame[25] = 0.5;
        // Some other joint at (0.8, 1.0)
        frame[0] = 0.8;
        frame[1] = 1.0;

        let result = p.process_frame(&frame);

        // Shoulder width = dist((0.3,0.5), (0.7,0.5)) = 0.4
        // Everything divided by 0.4 = multiplied by 2.5
        assert!((result[0] - 2.0).abs() < 1e-5); // 0.8 / 0.4
        assert!((result[1] - 2.5).abs() < 1e-5); // 1.0 / 0.4
    }

    #[test]
    fn test_scale_normalization_near_zero_shoulder() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: true,
            velocity_features: false,
            angle_features: false,
        };
        let mut p = Preprocessor::new(config, "mediapipe-pose-33-xy");

        // Both shoulders at same point — shoulder width = 0
        let mut frame = vec![0.0f32; 66];
        frame[22] = 0.5;
        frame[23] = 0.5;
        frame[24] = 0.5;
        frame[25] = 0.5;
        frame[0] = 1.0;

        let result = p.process_frame(&frame);

        // Should clamp to MIN_SHOULDER_WIDTH, not panic or produce NaN/Inf
        assert!(result[0].is_finite());
        assert!((result[0] - 1.0 / MIN_SHOULDER_WIDTH).abs() < 1e-3);
    }

    #[test]
    fn test_velocity_features_first_frame_zero() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: true,
            angle_features: false,
        };
        let mut p = Preprocessor::new(config, "mediapipe-pose-33-xy");

        let frame = vec![1.0, 2.0, 3.0, 4.0];
        let result = p.process_frame(&frame);

        // First frame: positions + zero velocities
        assert_eq!(result.len(), 8); // 4 positions + 4 velocities
        assert_eq!(&result[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&result[4..8], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_velocity_features_second_frame() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: true,
            angle_features: false,
        };
        let mut p = Preprocessor::new(config, "mediapipe-pose-33-xy");

        let frame1 = vec![1.0, 2.0, 3.0, 4.0];
        let _ = p.process_frame(&frame1);

        let frame2 = vec![1.5, 2.5, 3.0, 5.0];
        let result = p.process_frame(&frame2);

        // Second frame: positions + velocity (current - prev)
        assert_eq!(&result[0..4], &[1.5, 2.5, 3.0, 5.0]);
        assert!((result[4] - 0.5).abs() < 1e-6); // 1.5 - 1.0
        assert!((result[5] - 0.5).abs() < 1e-6); // 2.5 - 2.0
        assert!((result[6] - 0.0).abs() < 1e-6); // 3.0 - 3.0
        assert!((result[7] - 1.0).abs() < 1e-6); // 5.0 - 4.0
    }

    #[test]
    fn test_process_sequence() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: true,
            angle_features: false,
        };
        let p = Preprocessor::new(config, "mediapipe-pose-33-xy");

        let sequence = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![2.0, 3.0]];

        let result = p.process_sequence(&sequence);

        assert_eq!(result.len(), 3);
        // Frame 0: pos + zero velocity
        assert_eq!(result[0], vec![1.0, 2.0, 0.0, 0.0]);
        // Frame 1: pos + velocity
        assert_eq!(result[1], vec![1.5, 2.5, 0.5, 0.5]);
        // Frame 2: pos + velocity
        assert_eq!(result[2], vec![2.0, 3.0, 0.5, 0.5]);
    }

    #[test]
    fn test_process_sequence_empty() {
        let config = PreprocessingConfig::default();
        let p = Preprocessor::new(config, "mediapipe-pose-33-xy");
        let result = p.process_sequence(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_output_dimensions() {
        let p_no_vel = Preprocessor::new(
            PreprocessingConfig {
                hip_normalize: true,
                scale_normalize: true,
                velocity_features: false,
                angle_features: false,
            },
            "mediapipe-pose-33-xy",
        );
        assert_eq!(p_no_vel.output_dimensions(66), 66);

        let p_vel = Preprocessor::new(
            PreprocessingConfig {
                hip_normalize: false,
                scale_normalize: false,
                velocity_features: true,
                angle_features: false,
            },
            "mediapipe-pose-33-xy",
        );
        assert_eq!(p_vel.output_dimensions(66), 132);
    }

    #[test]
    fn test_reset_clears_velocity_state() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: true,
            angle_features: false,
        };
        let mut p = Preprocessor::new(config, "mediapipe-pose-33-xy");

        // Process two frames to build velocity state
        let _ = p.process_frame(&[1.0, 2.0]);
        let _ = p.process_frame(&[3.0, 4.0]);

        // Reset
        p.reset();

        // Next frame should have zero velocity (as if first frame)
        let result = p.process_frame(&[5.0, 6.0]);
        assert_eq!(&result[2..4], &[0.0, 0.0]);
    }

    #[test]
    fn test_unknown_tracking_system_skips_normalization() {
        let config = PreprocessingConfig {
            hip_normalize: true,
            scale_normalize: true,
            velocity_features: false,
            angle_features: false,
        };
        let mut p = Preprocessor::new(config, "unknown-system");

        // Should pass through unchanged (no layout for normalization)
        let frame = vec![1.0, 2.0, 3.0];
        let result = p.process_frame(&frame);
        assert_eq!(result, frame);
    }

    #[test]
    fn test_combined_hip_scale_velocity() {
        let config = PreprocessingConfig {
            hip_normalize: true,
            scale_normalize: true,
            velocity_features: true,
            angle_features: false,
        };
        let mut p = Preprocessor {
            config,
            layout: Some(mini_layout()),
            prev_frame: None,
        };

        // 3 joints x 2 coords = 6 floats
        // Joint 0 (left shoulder) at (0.3, 0.5)
        // Joint 1 (right shoulder) at (0.7, 0.5)
        // Joint 2 (hip) at (0.5, 0.4)
        let frame = vec![0.3, 0.5, 0.7, 0.5, 0.5, 0.4];
        let result = p.process_frame(&frame);

        // After hip centering (hip center = (0.5, 0.4)):
        //   Joint 0: (-0.2, 0.1), Joint 1: (0.2, 0.1), Joint 2: (0.0, 0.0)
        // Shoulder width after centering = dist((-0.2,0.1), (0.2,0.1)) = 0.4
        // After scale (divide by 0.4):
        //   Joint 0: (-0.5, 0.25), Joint 1: (0.5, 0.25), Joint 2: (0.0, 0.0)
        // Velocity: all zeros (first frame)
        // Output: 12 floats (6 position + 6 velocity)

        assert_eq!(result.len(), 12);
        assert!((result[0] - -0.5).abs() < 1e-5);
        assert!((result[1] - 0.25).abs() < 1e-5);
        // Velocity should be zeros
        assert_eq!(&result[6..12], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_layout_for_tracking_system() {
        assert!(layout_for_tracking_system("mediapipe-pose-33-xy").is_some());
        assert!(layout_for_tracking_system("kinect-v2-25").is_none());
        assert!(layout_for_tracking_system("").is_none());
    }

    #[test]
    fn test_mediapipe_layout_constants() {
        let l = TrackingLayout::mediapipe_pose_33_xy();
        assert_eq!(l.raw_dimensions, 66);
        assert_eq!(l.coords_per_joint, 2);
        assert_eq!(l.joint_count(), 33);
        assert_eq!(l.joint_offset(23), 46); // Left hip
        assert_eq!(l.joint_offset(24), 48); // Right hip
        assert_eq!(l.joint_offset(11), 22); // Left shoulder
        assert_eq!(l.joint_offset(12), 24); // Right shoulder
    }

    #[test]
    fn test_angle_at_b_right_angle() {
        // Right angle: A=(0,1), B=(0,0), C=(1,0) → π/2
        let angle = angle_at_b((0.0, 1.0), (0.0, 0.0), (1.0, 0.0));
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn test_angle_at_b_straight_line() {
        // Straight: A=(-1,0), B=(0,0), C=(1,0) → π
        let angle = angle_at_b((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0));
        assert!((angle - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn test_angle_at_b_degenerate() {
        // Degenerate: all same point → π
        let angle = angle_at_b((1.0, 1.0), (1.0, 1.0), (1.0, 1.0));
        assert!((angle - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn test_angle_features_appends_12_values() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: false,
            angle_features: true,
        };
        let mut p = Preprocessor::new(config, "mediapipe-pose-33-xy");
        let frame = vec![0.5; 66];
        let result = p.process_frame(&frame);
        assert_eq!(result.len(), 66 + 12, "angles should append 12 values");
    }

    #[test]
    fn test_output_dimensions_with_angles() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: false,
            angle_features: true,
        };
        let p = Preprocessor::new(config, "mediapipe-pose-33-xy");
        assert_eq!(p.output_dimensions(66), 78);
    }

    #[test]
    fn test_output_dimensions_with_angles_and_velocity() {
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: true,
            angle_features: true,
        };
        let p = Preprocessor::new(config, "mediapipe-pose-33-xy");
        // 66 + 12 angles = 78, then × 2 for velocity = 156
        assert_eq!(p.output_dimensions(66), 156);
    }
}
