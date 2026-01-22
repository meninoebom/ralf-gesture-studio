use std::fs;
use std::path::{Path, PathBuf};

use directories::UserDirs;
use thiserror::Error;

use super::Vocabulary;

/// Errors that can occur when saving or loading vocabularies
#[derive(Error, Debug)]
pub enum PersistenceError {
    #[error("Failed to read file: {0}")]
    ReadError(#[from] std::io::Error),

    #[error("Failed to parse vocabulary: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Could not determine user documents directory")]
    NoDocumentsDir,

    #[error("Failed to create directory: {0}")]
    CreateDirError(std::io::Error),
}

/// Get the default directory for vocabulary files: ~/Documents/RALF/
pub fn default_vocabulary_dir() -> Result<PathBuf, PersistenceError> {
    let user_dirs = UserDirs::new().ok_or(PersistenceError::NoDocumentsDir)?;
    let documents = user_dirs
        .document_dir()
        .ok_or(PersistenceError::NoDocumentsDir)?;
    Ok(documents.join("RALF"))
}

/// Ensure the default vocabulary directory exists
pub fn ensure_vocabulary_dir() -> Result<PathBuf, PersistenceError> {
    let dir = default_vocabulary_dir()?;
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(PersistenceError::CreateDirError)?;
    }
    Ok(dir)
}

/// Save a vocabulary to a .ralf file (JSON format)
pub fn save_vocabulary(vocabulary: &Vocabulary, path: &Path) -> Result<(), PersistenceError> {
    let json = serde_json::to_string_pretty(vocabulary)?;
    fs::write(path, json)?;
    Ok(())
}

/// Load a vocabulary from a .ralf file
pub fn load_vocabulary(path: &Path) -> Result<Vocabulary, PersistenceError> {
    let json = fs::read_to_string(path)?;
    let mut vocabulary: Vocabulary = serde_json::from_str(&json)?;
    // Recalculate the next ID counter since it's not stored in the file
    vocabulary.recalculate_next_id();
    Ok(vocabulary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load_roundtrip() {
        // Create a vocabulary with a gesture
        let mut vocab = Vocabulary::new("Test Vocabulary");
        vocab.add_gesture("wave");

        // Save to a temp file
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ralf");

        save_vocabulary(&vocab, &path).unwrap();

        // Verify file exists and contains JSON
        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains("Test Vocabulary"));
        assert!(contents.contains("wave"));

        // Load it back
        let loaded = load_vocabulary(&path).unwrap();
        assert_eq!(loaded.name, "Test Vocabulary");
        assert_eq!(loaded.gestures.len(), 1);
        assert_eq!(loaded.gestures[0].name, "wave");
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_vocabulary(Path::new("/nonexistent/path.ralf"));
        assert!(result.is_err());
    }

    #[test]
    fn test_default_vocabulary_dir() {
        let dir = default_vocabulary_dir();
        assert!(dir.is_ok());
        let path = dir.unwrap();
        assert!(path.to_string_lossy().contains("RALF"));
    }
}
