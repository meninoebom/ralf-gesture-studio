//! Diagnostic logging for gesture recognition analysis.
//!
//! Writes detailed recognition data to a log file for debugging subtle issues.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

/// Diagnostic event types
#[derive(Debug, Clone)]
pub enum DiagnosticEvent {
    /// Recognition cycle with all gesture distances
    Recognition {
        frame_num: usize,
        buffer_len: usize,
        window_size: usize,
        gestures: Vec<GestureDiag>,
    },
    /// A hit was fired
    Hit {
        frame_num: usize,
        gesture_name: String,
        distance: f32,
        threshold: f32,
        margin_pct: f32, // How far below threshold (negative = below)
    },
    /// A near-miss (close to threshold but didn't fire)
    NearMiss {
        frame_num: usize,
        gesture_name: String,
        distance: f32,
        threshold: f32,
        margin_pct: f32,
        reason: &'static str, // Why it didn't fire
    },
    /// State machine transition
    StateChange {
        gesture_name: String,
        from_state: String,
        to_state: String,
        distance: f32,
        threshold: f32,
        margin_pct: f32,
        frames_in_state: usize,
        reason: &'static str, // Why the transition happened
    },
    /// Training completed
    TrainingComplete {
        gesture_name: String,
        example_count: usize,
        mean: Option<f32>,
        std: Option<f32>,
        threshold: f32,
    },
}

/// Per-gesture diagnostic data
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GestureDiag {
    pub name: String,
    pub distance: Option<f32>,
    pub threshold: f32,
    pub armed: bool,
    pub in_cooldown: bool,
    pub example_count: usize,
}

/// Diagnostic logger
pub struct DiagnosticLogger {
    writer: Option<Mutex<BufWriter<File>>>,
    start_time: Instant,
    frame_count: usize,
    enabled: bool,
    /// Only log every Nth recognition cycle to reduce noise
    log_interval: usize,
    /// Near-miss threshold (log when within X% of threshold)
    near_miss_pct: f32,
}

impl DiagnosticLogger {
    /// Create a new diagnostic logger
    pub fn new() -> Self {
        Self {
            writer: None,
            start_time: Instant::now(),
            frame_count: 0,
            enabled: false,
            log_interval: 4, // Log every 4th recognition cycle (~4Hz at 15Hz recognition)
            near_miss_pct: 20.0, // Log when within 20% of threshold
        }
    }

    /// Enable logging to a file
    pub fn enable(&mut self, path: PathBuf) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(writer, "# RALF Gesture Studio - Diagnostic Log")?;
        writeln!(writer, "# Started: {:?}", std::time::SystemTime::now())?;
        writeln!(writer, "# Format: timestamp_ms,event_type,data...")?;
        writeln!(writer, "#")?;
        writeln!(writer, "# Event types:")?;
        writeln!(writer, "#   REC: Recognition cycle - frame,buffer,window,gesture:dist:thresh:armed:cooldown,...")?;
        writeln!(writer, "#   HIT: Hit fired - frame,gesture,distance,threshold,margin%")?;
        writeln!(writer, "#   NEAR: Near miss - frame,gesture,distance,threshold,margin%,reason")?;
        writeln!(writer, "#   STATE: State transition - gesture,from,to,distance,threshold,margin%,frames_in_state,reason")?;
        writeln!(writer, "#   TRAIN: Training complete - gesture,examples,mean,std,threshold")?;
        writeln!(writer, "#")?;
        writer.flush()?;

        self.writer = Some(Mutex::new(writer));
        self.start_time = Instant::now();
        self.enabled = true;

        Ok(())
    }

    /// Disable logging
    pub fn disable(&mut self) {
        if let Some(ref writer) = self.writer {
            if let Ok(mut w) = writer.lock() {
                let _ = writeln!(w, "# Logging disabled at {}ms", self.start_time.elapsed().as_millis());
                let _ = w.flush();
            }
        }
        self.writer = None;
        self.enabled = false;
    }

    /// Check if logging is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log an event
    pub fn log(&mut self, event: DiagnosticEvent) {
        if !self.enabled {
            return;
        }

        let timestamp = self.start_time.elapsed().as_millis();

        // Check if this is an important event that needs flushing
        let needs_flush = matches!(&event,
            DiagnosticEvent::Hit { .. } |
            DiagnosticEvent::NearMiss { .. } |
            DiagnosticEvent::StateChange { .. } |
            DiagnosticEvent::TrainingComplete { .. }
        );

        let line = match event {
            DiagnosticEvent::Recognition { frame_num, buffer_len, window_size, gestures } => {
                self.frame_count += 1;
                // Only log every Nth cycle
                if !self.frame_count.is_multiple_of(self.log_interval) {
                    return;
                }

                let gesture_data: Vec<String> = gestures.iter().map(|g| {
                    let dist = g.distance.map(|d| format!("{:.0}", d)).unwrap_or_else(|| "-".to_string());
                    let armed = if g.armed { "A" } else { "-" };
                    let cooldown = if g.in_cooldown { "C" } else { "-" };
                    format!("{}:{}:{:.0}:{}:{}", g.name, dist, g.threshold, armed, cooldown)
                }).collect();

                format!("{},REC,{},{},{},{}", timestamp, frame_num, buffer_len, window_size, gesture_data.join(","))
            }
            DiagnosticEvent::Hit { frame_num, gesture_name, distance, threshold, margin_pct } => {
                format!("{},HIT,{},{},{:.0},{:.0},{:.1}%", timestamp, frame_num, gesture_name, distance, threshold, margin_pct)
            }
            DiagnosticEvent::NearMiss { frame_num, gesture_name, distance, threshold, margin_pct, reason } => {
                format!("{},NEAR,{},{},{:.0},{:.0},{:.1}%,{}", timestamp, frame_num, gesture_name, distance, threshold, margin_pct, reason)
            }
            DiagnosticEvent::StateChange { gesture_name, from_state, to_state, distance, threshold, margin_pct, frames_in_state, reason } => {
                format!("{},STATE,{},{},{},{:.0},{:.0},{:.1}%,{},{}",
                    timestamp, gesture_name, from_state, to_state, distance, threshold, margin_pct, frames_in_state, reason)
            }
            DiagnosticEvent::TrainingComplete { gesture_name, example_count, mean, std, threshold } => {
                let mean_str = mean.map(|m| format!("{:.0}", m)).unwrap_or_else(|| "-".to_string());
                let std_str = std.map(|s| format!("{:.0}", s)).unwrap_or_else(|| "-".to_string());
                format!("{},TRAIN,{},{},{},{},{:.0}", timestamp, gesture_name, example_count, mean_str, std_str, threshold)
            }
        };

        if let Some(ref writer) = self.writer {
            if let Ok(mut w) = writer.lock() {
                let _ = writeln!(w, "{}", line);
                // Flush for important events
                if needs_flush {
                    let _ = w.flush();
                }
            }
        }
    }

    /// Get the near-miss percentage threshold
    pub fn near_miss_pct(&self) -> f32 {
        self.near_miss_pct
    }

    /// Flush the log
    pub fn flush(&self) {
        if let Some(ref writer) = self.writer {
            if let Ok(mut w) = writer.lock() {
                let _ = w.flush();
            }
        }
    }
}

impl Default for DiagnosticLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DiagnosticLogger {
    fn drop(&mut self) {
        self.flush();
    }
}
