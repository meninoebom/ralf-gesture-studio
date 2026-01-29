use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Default coefficient for statistical threshold (μ + σ×coefficient)
fn default_threshold_coefficient() -> f32 {
    2.0
}

/// Configuration for OSC input (receiving skeleton data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Number of floats per input frame (e.g., 66 for 33 MediaPipe keypoints × XY)
    pub dimensions: usize,
    /// UDP port to listen on
    pub port: u16,
    /// OSC address to listen for
    pub address: String,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            dimensions: 66,
            port: 6448,
            address: "/wek/inputs".to_string(),
        }
    }
}

/// Configuration for OSC output (sending hit messages)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Target hostname/IP
    pub host: String,
    /// Target UDP port
    pub port: u16,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 12000,
        }
    }
}

/// One recorded instance of a gesture.
/// Contains the raw motion capture data (frames of float arrays).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// When this example was recorded
    pub recorded_at: DateTime<Utc>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Number of frames captured
    pub frame_count: usize,
    /// Motion data: each frame is a vector of floats (e.g., joint positions)
    pub frames: Vec<Vec<f32>>,
}

impl Example {
    /// Create a new example from captured frames
    pub fn new(frames: Vec<Vec<f32>>, duration_ms: u64) -> Self {
        let frame_count = frames.len();
        Self {
            recorded_at: Utc::now(),
            duration_ms,
            frame_count,
            frames,
        }
    }
}

/// A single trained movement pattern (e.g., "jack", "wave", "spin").
/// Has a name, recognition threshold, and output OSC address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gesture {
    /// Unique identifier within vocabulary (1, 2, 3...)
    pub id: u32,
    /// User-editable gesture name
    pub name: String,
    /// OSC address for hit output (e.g., "/gesture/1")
    pub osc_address: String,
    /// DTW distance threshold for recognition (lower = stricter)
    pub threshold: f32,
    /// When this gesture was created
    pub created_at: DateTime<Utc>,
    /// Recorded examples of this gesture
    pub examples: Vec<Example>,

    // --- Statistical Threshold Fields (GRT-style μ+σ approach) ---
    /// Mean distance between training examples (computed after training)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance_mean: Option<f32>,
    /// Standard deviation of distances between training examples
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance_std: Option<f32>,
    /// If true, use manual threshold; if false, use statistical threshold (μ + σ×coefficient)
    #[serde(default)]
    pub threshold_manual_override: bool,
    /// Coefficient for statistical threshold: threshold = μ + σ×coefficient (default 2.0)
    #[serde(default = "default_threshold_coefficient")]
    pub threshold_coefficient: f32,

    // --- GRT-style Best Template Selection ---
    /// Index of the best template (example with lowest average distance to others)
    /// Used during recognition to compare only against this representative example.
    /// Falls back to comparing all examples if None or only 1-2 examples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_template_index: Option<usize>,
}

impl Gesture {
    /// Create a new gesture with the given name and ID
    pub fn new(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            osc_address: format!("/gesture/{}", id),
            threshold: 100.0, // Default threshold for normalized 0-1 coordinate inputs
            created_at: Utc::now(),
            examples: Vec::new(),
            // Statistical threshold fields (computed after training)
            distance_mean: None,
            distance_std: None,
            threshold_manual_override: false,
            threshold_coefficient: default_threshold_coefficient(),
            best_template_index: None,
        }
    }

    /// Compute and update the effective threshold based on statistical data.
    /// If manual override is set or no statistics are available, uses the manual threshold.
    /// Otherwise, computes threshold = μ + σ × coefficient.
    #[allow(dead_code)]
    pub fn effective_threshold(&self) -> f32 {
        if self.threshold_manual_override {
            return self.threshold;
        }
        match (self.distance_mean, self.distance_std) {
            (Some(mean), Some(std)) => mean + std * self.threshold_coefficient,
            _ => self.threshold, // Fall back to manual threshold if no statistics
        }
    }

    /// Update the statistical threshold values and recalculate threshold.
    /// This should be called after training when new examples are added.
    pub fn update_statistics(&mut self, mean: f32, std: f32, best_template_index: Option<usize>) {
        self.distance_mean = Some(mean);
        self.distance_std = Some(std);
        self.best_template_index = best_template_index;
        // Update the threshold field if not in manual override mode
        if !self.threshold_manual_override {
            self.threshold = mean + std * self.threshold_coefficient;
        }
    }

    /// Clear statistical data (e.g., when examples are removed)
    #[allow(dead_code)]
    pub fn clear_statistics(&mut self) {
        self.distance_mean = None;
        self.distance_std = None;
        self.best_template_index = None;
    }

    /// Check if this gesture has valid statistical threshold data
    #[allow(dead_code)]
    pub fn has_statistics(&self) -> bool {
        self.distance_mean.is_some() && self.distance_std.is_some()
    }

    /// Add a recorded example to this gesture
    pub fn add_example(&mut self, example: Example) {
        self.examples.push(example);
    }

    /// Returns true if this gesture has at least one example
    #[allow(dead_code)]
    pub fn has_examples(&self) -> bool {
        !self.examples.is_empty()
    }

    /// Returns the number of recorded examples
    #[allow(dead_code)]
    pub fn example_count(&self) -> usize {
        self.examples.len()
    }
}

// --- Default functions for serde ---

fn default_tracking_system() -> String {
    "mediapipe-pose-33-xy".to_string()
}

fn default_coordinate_system() -> String {
    "normalized-0-1-xy".to_string()
}

fn default_tags() -> Vec<String> {
    Vec::new()
}

fn default_extensions() -> HashMap<String, serde_json::Value> {
    HashMap::new()
}

/// A collection of gestures that work together (e.g., "House Foundations").
/// This is the root container - one vocabulary = one .ralf file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// File format version (SchemaVer: MODEL.REVISION.ADDITION)
    pub version: String,
    /// Unique identifier for this vocabulary (UUID v4)
    #[serde(default = "Uuid::new_v4")]
    pub uuid: Uuid,
    /// User-editable vocabulary name
    pub name: String,
    /// When vocabulary was created
    pub created_at: DateTime<Utc>,
    /// When vocabulary was last modified
    pub modified_at: DateTime<Utc>,
    /// OSC input configuration
    pub input: InputConfig,
    /// OSC output configuration
    pub output: OutputConfig,

    // --- Research-ready metadata (v1.1) ---
    /// Tracking system identifier (e.g., "mediapipe-pose-33-xy", "kinect-v2-25")
    #[serde(default = "default_tracking_system")]
    pub tracking_system: String,
    /// Coordinate system description (e.g., "normalized-0-1-xy", "meters-xyz")
    #[serde(default = "default_coordinate_system")]
    pub coordinate_system: String,
    /// Frame rate of source data (if known)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_fps: Option<f32>,
    /// License for the data (e.g., "CC-BY-4.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Creator/attribution
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creator: Option<String>,
    /// Tags for discoverability (e.g., ["house", "dance", "foundations"])
    #[serde(default = "default_tags", skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Extensibility hook for future metadata without schema changes
    #[serde(default = "default_extensions", skip_serializing_if = "HashMap::is_empty")]
    pub extensions: HashMap<String, serde_json::Value>,

    /// Baseline frames (deprecated - kept for file compatibility)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<Vec<Vec<f32>>>,
    /// List of gestures in this vocabulary
    pub gestures: Vec<Gesture>,
    /// Counter for generating unique gesture IDs
    #[serde(skip)]
    next_gesture_id: u32,
}

impl Vocabulary {
    /// Current file format version
    pub const CURRENT_VERSION: &'static str = "1.1";

    /// Create a new empty vocabulary with the given name
    pub fn new(name: &str) -> Self {
        let now = Utc::now();
        Self {
            version: Self::CURRENT_VERSION.to_string(),
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            created_at: now,
            modified_at: now,
            input: InputConfig::default(),
            output: OutputConfig::default(),
            // Research metadata with sensible defaults
            tracking_system: default_tracking_system(),
            coordinate_system: default_coordinate_system(),
            source_fps: Some(60.0), // Assume 60fps unless told otherwise
            license: None,
            creator: None,
            tags: Vec::new(),
            extensions: HashMap::new(),
            // Legacy/internal fields
            baseline: None,
            gestures: Vec::new(),
            next_gesture_id: 1,
        }
    }

    /// Add a new gesture with the given name
    /// Returns the ID of the newly created gesture
    pub fn add_gesture(&mut self, name: &str) -> u32 {
        let id = self.next_gesture_id;
        self.next_gesture_id += 1;
        let gesture = Gesture::new(id, name);
        self.gestures.push(gesture);
        self.touch();
        id
    }

    /// Find a gesture by ID
    pub fn get_gesture(&self, id: u32) -> Option<&Gesture> {
        self.gestures.iter().find(|g| g.id == id)
    }

    /// Find a gesture by ID (mutable)
    pub fn get_gesture_mut(&mut self, id: u32) -> Option<&mut Gesture> {
        self.gestures.iter_mut().find(|g| g.id == id)
    }

    /// Remove a gesture by ID
    /// Returns true if a gesture was removed
    #[allow(dead_code)]
    pub fn remove_gesture(&mut self, id: u32) -> bool {
        let len_before = self.gestures.len();
        self.gestures.retain(|g| g.id != id);
        let removed = self.gestures.len() < len_before;
        if removed {
            self.touch();
        }
        removed
    }

    /// Update the modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = Utc::now();
    }

    /// Recalculate next_gesture_id after loading from file
    #[allow(dead_code)]
    pub fn recalculate_next_id(&mut self) {
        self.next_gesture_id = self
            .gestures
            .iter()
            .map(|g| g.id)
            .max()
            .unwrap_or(0)
            + 1;
    }
}
