//! One Euro Filter for real-time pose smoothing.
//!
//! Reduces MediaPipe jitter while preserving fast movements.
//! Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
//! for Noisy Input in Interactive Systems" (CHI 2012).

use std::f32::consts::PI;

/// Single-channel adaptive low-pass filter.
struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: f32,
    dx_prev: f32,
    initialized: bool,
}

impl OneEuroFilter {
    fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: 0.0,
            dx_prev: 0.0,
            initialized: false,
        }
    }

    fn filter(&mut self, x: f32, dt: f32) -> f32 {
        if !self.initialized {
            self.x_prev = x;
            self.initialized = true;
            return x;
        }

        // Estimate derivative
        let dx = (x - self.x_prev) / dt;
        let alpha_d = smoothing_factor(dt, self.d_cutoff);
        let dx_hat = exponential_smoothing(alpha_d, dx, self.dx_prev);

        // Adaptive cutoff based on speed
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let alpha = smoothing_factor(dt, cutoff);
        let x_hat = exponential_smoothing(alpha, x, self.x_prev);

        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        x_hat
    }
}

fn smoothing_factor(dt: f32, cutoff: f32) -> f32 {
    let r = 2.0 * PI * cutoff * dt;
    r / (r + 1.0)
}

fn exponential_smoothing(alpha: f32, x: f32, x_prev: f32) -> f32 {
    alpha * x + (1.0 - alpha) * x_prev
}

/// Multi-channel smoother for pose frames (one filter per coordinate).
pub struct PoseSmoother {
    filters: Vec<OneEuroFilter>,
}

impl PoseSmoother {
    /// Create a smoother for `num_channels` coordinates.
    /// Default params tuned for dance motion capture at ~30Hz.
    pub fn new(num_channels: usize) -> Self {
        let filters = (0..num_channels)
            .map(|_| OneEuroFilter::new(1.5, 0.007, 1.0))
            .collect();
        Self { filters }
    }

    /// Smooth a frame. `dt` is seconds since last frame.
    pub fn smooth(&mut self, frame: &[f32], dt: f32) -> Vec<f32> {
        frame
            .iter()
            .zip(self.filters.iter_mut())
            .map(|(x, f)| f.filter(*x, dt))
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_frame_passes_through() {
        let mut smoother = PoseSmoother::new(3);
        let frame = vec![1.0, 2.0, 3.0];
        let result = smoother.smooth(&frame, 1.0 / 30.0);
        assert_eq!(result, frame);
    }

    #[test]
    fn test_smooths_step_input() {
        let mut smoother = PoseSmoother::new(1);
        let dt = 1.0 / 30.0;

        // First frame at 0
        smoother.smooth(&[0.0], dt);

        // Step to 1.0 — smoothed output should lag behind
        let result = smoother.smooth(&[1.0], dt);
        assert!(
            result[0] > 0.0 && result[0] < 1.0,
            "should smooth step: got {}",
            result[0]
        );
    }

    #[test]
    fn test_preserves_fast_ramp() {
        let mut smoother = PoseSmoother::new(1);
        let dt = 1.0 / 30.0;

        // Ramp up quickly over several frames
        for i in 0..20 {
            smoother.smooth(&[i as f32 * 0.1], dt);
        }

        // After sustained movement, filter should track closely
        let result = smoother.smooth(&[2.0], dt);
        // With beta=0.007 and sustained speed, cutoff adapts up — should be close
        assert!(
            result[0] > 1.5,
            "should track fast movement: got {}",
            result[0]
        );
    }

    #[test]
    fn test_reduces_jitter_on_static_signal() {
        let mut smoother = PoseSmoother::new(1);
        let dt = 1.0 / 30.0;

        // Settle at 1.0
        for _ in 0..30 {
            smoother.smooth(&[1.0], dt);
        }

        // Add jitter
        let jittery = smoother.smooth(&[1.05], dt);
        assert!(
            (jittery[0] - 1.0).abs() < 0.03,
            "should suppress jitter on static signal: got {}",
            jittery[0]
        );
    }
}
