mod vocabulary;
mod persistence;

pub use vocabulary::{Vocabulary, Example};

// Re-export for future use (gesture editing)
#[allow(unused_imports)]
pub use vocabulary::Gesture;

// Re-export for future use (vocabulary config editing)
#[allow(unused_imports)]
pub use vocabulary::{InputConfig, OutputConfig};

// Re-export for future use (file operations)
#[allow(unused_imports)]
pub use persistence::{save_vocabulary, load_vocabulary, default_vocabulary_dir};
