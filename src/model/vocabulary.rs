use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
    pub fn update_statistics(&mut self, mean: f32, std: f32) {
        self.distance_mean = Some(mean);
        self.distance_std = Some(std);
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

/// A collection of gestures that work together (e.g., "House Foundations").
/// This is the root container - one vocabulary = one .ralf file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// File format version
    pub version: String,
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
    /// Create a new empty vocabulary with the given name
    pub fn new(name: &str) -> Self {
        let now = Utc::now();
        Self {
            version: "1.0".to_string(),
            name: name.to_string(),
            created_at: now,
            modified_at: now,
            input: InputConfig::default(),
            output: OutputConfig::default(),
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
