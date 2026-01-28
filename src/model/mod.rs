mod vocabulary;
mod persistence;

pub use vocabulary::{Vocabulary, Example};
pub use persistence::{save_vocabulary, load_vocabulary, default_vocabulary_dir};

// Re-export for potential external consumers
#[allow(unused_imports)]
pub use vocabulary::{Gesture, InputConfig, OutputConfig};
