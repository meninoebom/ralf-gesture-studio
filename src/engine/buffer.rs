//! Frame buffer for storing incoming OSC data.
//!
//! Provides a circular buffer for real-time frame storage, supporting both
//! recording (capturing a fixed duration) and sliding window matching.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::dtw::Frame;

/// A timestamped frame of data
#[derive(Debug, Clone)]
pub struct TimestampedFrame {
    pub data: Frame,
    #[allow(dead_code)]
    pub timestamp: Instant,
}

/// Circular buffer for storing incoming frames.
///
/// Maintains a fixed-size buffer of recent frames, automatically discarding
/// old frames when the buffer is full.
#[allow(dead_code)]
#[derive(Debug)]
pub struct FrameBuffer {
    /// The frames in the buffer
    frames: VecDeque<TimestampedFrame>,
    /// Maximum number of frames to store
    max_frames: usize,
}

#[allow(dead_code)]
impl FrameBuffer {
    /// Create a new frame buffer with the given capacity.
    ///
    /// # Arguments
    /// * `max_frames` - Maximum number of frames to store (older frames are dropped)
    pub fn new(max_frames: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(max_frames),
            max_frames,
        }
    }

    /// Add a new frame to the buffer.
    ///
    /// If the buffer is full, the oldest frame is removed.
    pub fn push(&mut self, frame: Frame) {
        let timestamped = TimestampedFrame {
            data: frame,
            timestamp: Instant::now(),
        };

        if self.frames.len() >= self.max_frames {
            self.frames.pop_front();
        }
        self.frames.push_back(timestamped);
    }

    /// Get the number of frames currently in the buffer.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Check if the buffer is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Clear all frames from the buffer.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Get the most recent N frames as a sequence (for DTW matching).
    ///
    /// Returns fewer frames if the buffer doesn't have enough.
    #[allow(dead_code)]
    pub fn recent_frames(&self, count: usize) -> Vec<Frame> {
        let start = self.frames.len().saturating_sub(count);
        self.frames
            .iter()
            .skip(start)
            .map(|f| f.data.clone())
            .collect()
    }

    /// Get all frames currently in the buffer.
    #[allow(dead_code)]
    pub fn all_frames(&self) -> Vec<Frame> {
        self.frames.iter().map(|f| f.data.clone()).collect()
    }

    /// Get downsampled frames (every Nth frame) for efficient matching.
    ///
    /// For DTW matching, we don't need 60fps - 15fps is sufficient.
    /// This reduces computation by 4x when using step=4.
    ///
    /// # Arguments
    /// * `count` - Maximum number of frames to include
    /// * `step` - Take every Nth frame (1 = all frames, 4 = every 4th frame)
    pub fn downsampled(&self, count: usize, step: usize) -> Vec<Frame> {
        let step = step.max(1); // Minimum step of 1
        let start = self.frames.len().saturating_sub(count * step);

        self.frames
            .iter()
            .skip(start)
            .step_by(step)
            .take(count)
            .map(|f| f.data.clone())
            .collect()
    }

    /// Get frames from the last N milliseconds.
    #[allow(dead_code)]
    pub fn frames_since(&self, duration: Duration) -> Vec<Frame> {
        let cutoff = Instant::now() - duration;
        self.frames
            .iter()
            .filter(|f| f.timestamp >= cutoff)
            .map(|f| f.data.clone())
            .collect()
    }

    /// Get the timestamp of the most recent frame.
    #[allow(dead_code)]
    pub fn last_frame_time(&self) -> Option<Instant> {
        self.frames.back().map(|f| f.timestamp)
    }

    /// Get the duration covered by the buffer (oldest to newest frame).
    #[allow(dead_code)]
    pub fn duration(&self) -> Option<Duration> {
        if self.frames.len() < 2 {
            return None;
        }
        let oldest = self.frames.front()?.timestamp;
        let newest = self.frames.back()?.timestamp;
        Some(newest.duration_since(oldest))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    // =========================================================================
    // FrameBuffer Tests
    // =========================================================================

    #[test]
    fn test_buffer_creation() {
        let buffer = FrameBuffer::new(100);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_push() {
        let mut buffer = FrameBuffer::new(100);
        buffer.push(vec![1.0, 2.0]);
        buffer.push(vec![3.0, 4.0]);

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_buffer_circular() {
        let mut buffer = FrameBuffer::new(3);
        buffer.push(vec![1.0]);
        buffer.push(vec![2.0]);
        buffer.push(vec![3.0]);
        buffer.push(vec![4.0]); // Should drop first frame

        assert_eq!(buffer.len(), 3);

        let frames = buffer.all_frames();
        assert_eq!(frames[0], vec![2.0]);
        assert_eq!(frames[1], vec![3.0]);
        assert_eq!(frames[2], vec![4.0]);
    }

    #[test]
    fn test_buffer_recent_frames() {
        let mut buffer = FrameBuffer::new(10);
        for i in 0..5 {
            buffer.push(vec![i as f32]);
        }

        let recent = buffer.recent_frames(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0], vec![2.0]);
        assert_eq!(recent[1], vec![3.0]);
        assert_eq!(recent[2], vec![4.0]);
    }

    #[test]
    fn test_buffer_recent_frames_not_enough() {
        let mut buffer = FrameBuffer::new(10);
        buffer.push(vec![1.0]);
        buffer.push(vec![2.0]);

        let recent = buffer.recent_frames(5);
        assert_eq!(recent.len(), 2); // Only has 2 frames
    }

    #[test]
    fn test_buffer_clear() {
        let mut buffer = FrameBuffer::new(10);
        buffer.push(vec![1.0]);
        buffer.push(vec![2.0]);
        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_buffer_downsampled() {
        let mut buffer = FrameBuffer::new(100);

        // Add 60 frames (simulating 1 second at 60fps)
        for i in 0..60 {
            buffer.push(vec![i as f32]);
        }

        // Downsample to every 4th frame, get 15 frames (simulating 15fps)
        let downsampled = buffer.downsampled(15, 4);
        assert_eq!(downsampled.len(), 15);

        // Check that we got every 4th frame from the end
        // Frames 0..60, step by 4 from near the end
        // The frames should represent indices that are 4 apart
        for i in 0..downsampled.len() - 1 {
            let diff = downsampled[i + 1][0] - downsampled[i][0];
            assert_eq!(diff, 4.0, "Frames should be 4 apart");
        }
    }

    #[test]
    fn test_buffer_downsampled_not_enough_frames() {
        let mut buffer = FrameBuffer::new(100);

        // Add only 20 frames
        for i in 0..20 {
            buffer.push(vec![i as f32]);
        }

        // Request 15 frames at step 4 (would need 60 frames ideally)
        let downsampled = buffer.downsampled(15, 4);
        assert_eq!(downsampled.len(), 5); // 20 / 4 = 5 frames available
    }

    #[test]
    fn test_buffer_frames_since() {
        let mut buffer = FrameBuffer::new(100);

        // Add some frames
        buffer.push(vec![1.0]);
        sleep(Duration::from_millis(50));
        buffer.push(vec![2.0]);
        sleep(Duration::from_millis(50));
        buffer.push(vec![3.0]);

        // Get frames from last 60ms (should get at least the last 1-2)
        let recent = buffer.frames_since(Duration::from_millis(60));
        assert!(!recent.is_empty());
        assert!(recent.len() <= 2);
    }

}
