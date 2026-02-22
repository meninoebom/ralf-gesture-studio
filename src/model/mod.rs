mod persistence;
mod vocabulary;

pub use persistence::{default_vocabulary_dir, load_vocabulary, save_vocabulary};
pub use vocabulary::{default_threshold_coefficient, Example, Vocabulary};

// Re-export for potential external consumers
#[allow(unused_imports)]
pub use vocabulary::{Gesture, InputConfig, OutputConfig};
