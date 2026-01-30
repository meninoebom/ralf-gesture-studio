use std::fs;
use std::path::{Path, PathBuf};

use directories::UserDirs;
use serde::Deserialize;
use thiserror::Error;
use uuid::Uuid;

use super::Vocabulary;

/// Errors that can occur when saving or loading vocabularies
#[allow(dead_code)]
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

    #[error("Unsupported file format version: {0}")]
    UnsupportedVersion(String),
}

/// Minimal struct to check file version before full parse
#[derive(Deserialize)]
struct VersionCheck {
    version: String,
}

/// Get the default directory for vocabulary files: ~/Documents/RALF/
#[allow(dead_code)]
pub fn default_vocabulary_dir() -> Result<PathBuf, PersistenceError> {
    let user_dirs = UserDirs::new().ok_or(PersistenceError::NoDocumentsDir)?;
    let documents = user_dirs
        .document_dir()
        .ok_or(PersistenceError::NoDocumentsDir)?;
    Ok(documents.join("RALF"))
}

/// Ensure the default vocabulary directory exists
#[allow(dead_code)]
pub fn ensure_vocabulary_dir() -> Result<PathBuf, PersistenceError> {
    let dir = default_vocabulary_dir()?;
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(PersistenceError::CreateDirError)?;
    }
    Ok(dir)
}

/// Save a vocabulary to a .ralf file (JSON format)
#[allow(dead_code)]
pub fn save_vocabulary(vocabulary: &Vocabulary, path: &Path) -> Result<(), PersistenceError> {
    let json = serde_json::to_string_pretty(vocabulary)?;
    fs::write(path, json)?;
    Ok(())
}

/// Load a vocabulary from a .ralf file
/// Handles migration from older file format versions.
#[allow(dead_code)]
pub fn load_vocabulary(path: &Path) -> Result<Vocabulary, PersistenceError> {
    let json = fs::read_to_string(path)?;

    // Check version first
    let version_check: VersionCheck = serde_json::from_str(&json)?;

    let mut vocabulary = match version_check.version.as_str() {
        "1.0" => {
            // v1.0 -> v1.1 migration: serde defaults handle new fields
            // UUID will be generated, metadata fields get defaults
            let mut vocab: Vocabulary = serde_json::from_str(&json)?;
            // Generate a new UUID for migrated files (they didn't have one)
            vocab.uuid = Uuid::new_v4();
            // Update version to current
            vocab.version = Vocabulary::CURRENT_VERSION.to_string();
            vocab
        }
        "1.1" => {
            // Current version, parse directly
            serde_json::from_str(&json)?
        }
        v => {
            // Unknown version - try to parse anyway (forward compatibility)
            // If parsing fails, it will return an error
            eprintln!(
                "Warning: Unknown vocabulary version '{}', attempting to load anyway",
                v
            );
            serde_json::from_str(&json)?
        }
    };

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
