mod vocabulary;
mod persistence;

pub use vocabulary::{Vocabulary, Gesture, Example, InputConfig, OutputConfig};
pub use persistence::{save_vocabulary, load_vocabulary, default_vocabulary_dir};
