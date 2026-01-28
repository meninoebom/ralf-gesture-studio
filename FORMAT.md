# RALF Vocabulary File Format (.ralf)

This document describes the `.ralf` file format used by RALF Gesture Studio to store gesture vocabularies.

## Overview

- **Format**: JSON (human-readable, pretty-printed)
- **Extension**: `.ralf`
- **Current Version**: 1.1
- **Default Location**: `~/Documents/RALF/`

## File Structure

```json
{
  "version": "1.1",
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "name": "House Foundations",
  "created_at": "2025-01-21T10:30:00Z",
  "modified_at": "2025-01-21T14:22:00Z",
  "input": { ... },
  "output": { ... },
  "tracking_system": "mediapipe-pose-33-xy",
  "coordinate_system": "normalized-0-1-xy",
  "source_fps": 60.0,
  "license": "CC-BY-4.0",
  "creator": "Your Name",
  "tags": ["house", "dance", "foundations"],
  "gestures": [ ... ]
}
```

## Field Reference

### Root Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | File format version (SchemaVer: MODEL.REVISION.ADDITION) |
| `uuid` | string | Yes | Unique identifier (UUID v4) for cross-system references |
| `name` | string | Yes | User-editable vocabulary name |
| `created_at` | ISO 8601 | Yes | When vocabulary was created |
| `modified_at` | ISO 8601 | Yes | When vocabulary was last modified |
| `input` | object | Yes | OSC input configuration |
| `output` | object | Yes | OSC output configuration |
| `gestures` | array | Yes | Array of Gesture objects |

### Research Metadata (v1.1)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tracking_system` | string | No | `"mediapipe-pose-33-xy"` | Source tracking system identifier |
| `coordinate_system` | string | No | `"normalized-0-1-xy"` | Coordinate system description |
| `source_fps` | float | No | null | Frame rate of source data |
| `license` | string | No | null | Data license (e.g., "CC-BY-4.0") |
| `creator` | string | No | null | Creator/attribution |
| `tags` | array | No | `[]` | Tags for discoverability |
| `extensions` | object | No | `{}` | Arbitrary metadata for extensibility |

### InputConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dimensions` | int | 66 | Number of floats per frame (e.g., 33 joints × 2 for XY) |
| `port` | int | 6448 | UDP port to listen on (Wekinator compatible) |
| `address` | string | `"/wek/inputs"` | OSC address to listen for |

### OutputConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Target hostname/IP for hit messages |
| `port` | int | 12000 | Target UDP port |

### Gesture

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique identifier within vocabulary |
| `name` | string | Yes | User-editable gesture name |
| `osc_address` | string | Yes | OSC address for hit output (e.g., "/gesture/1") |
| `threshold` | float | Yes | DTW distance threshold for recognition |
| `created_at` | ISO 8601 | Yes | When gesture was created |
| `examples` | array | Yes | Array of Example objects |
| `distance_mean` | float | No | Statistical: mean distance between examples (μ) |
| `distance_std` | float | No | Statistical: standard deviation (σ) |
| `threshold_manual_override` | bool | No | If true, use manual threshold instead of μ+σ |
| `threshold_coefficient` | float | No | Multiplier for auto threshold (default 2.0) |

### Example

| Field | Type | Description |
|-------|------|-------------|
| `recorded_at` | ISO 8601 | When example was recorded |
| `duration_ms` | int | Duration in milliseconds |
| `frame_count` | int | Number of frames captured |
| `frames` | array | 2D array: each row is one frame of float values |

## Frame Data Format

Each frame is an array of floats representing skeleton joint positions:

```json
"frames": [
  [0.123, 0.456, 0.789, 0.012, ...],  // Frame 1: 66 values
  [0.124, 0.458, 0.791, 0.014, ...],  // Frame 2: 66 values
  ...
]
```

**Default (MediaPipe Pose)**: 66 values = 33 keypoints × 2 (X, Y coordinates)
- Coordinates are normalized to 0-1 range
- Keypoint order follows MediaPipe convention (nose, eyes, ears, shoulders, etc.)

## Version History

| Version | Changes |
|---------|---------|
| 1.0 | Initial release |
| 1.1 | Added `uuid`, `tracking_system`, `coordinate_system`, `source_fps`, `license`, `creator`, `tags`, `extensions` |

## Migration

Files with version "1.0" are automatically migrated when loaded:
- A new UUID is generated
- New metadata fields receive default values
- File version is updated to "1.1" on next save

## File Size Estimates

| Content | Approximate Size |
|---------|------------------|
| Per frame (66 dims, JSON) | ~550 bytes |
| Per example (170 frames) | ~93 KB |
| Per gesture (5 examples) | ~467 KB |
| Full vocabulary (5 gestures × 5 examples) | ~2.3 MB |

## Tracking System Identifiers

Common values for `tracking_system`:

| Identifier | Description |
|------------|-------------|
| `mediapipe-pose-33-xy` | MediaPipe Pose (33 keypoints, XY only) |
| `mediapipe-pose-33-xyz` | MediaPipe Pose (33 keypoints, XYZ) |
| `kinect-v2-25` | Kinect v2 (25 joints) |
| `openpose-25` | OpenPose BODY_25 model |
| `custom` | Custom tracking system |

## Coordinate System Identifiers

Common values for `coordinate_system`:

| Identifier | Description |
|------------|-------------|
| `normalized-0-1-xy` | Normalized to 0-1 range, XY coordinates |
| `normalized-0-1-xyz` | Normalized to 0-1 range, XYZ coordinates |
| `pixels-xy` | Pixel coordinates, XY |
| `meters-xyz` | Real-world meters, XYZ |

## FAIR Principles Compliance

This format supports [FAIR data principles](https://www.go-fair.org/fair-principles/):

- **Findable**: UUID enables unique identification; tags enable discovery
- **Accessible**: Plain JSON format readable by any language
- **Interoperable**: Tracking system and coordinate system fields document data compatibility
- **Reusable**: License field clarifies usage rights; creator enables attribution

## Example File

```json
{
  "version": "1.1",
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "name": "House Foundations",
  "created_at": "2026-01-21T10:30:00Z",
  "modified_at": "2026-01-21T14:22:00Z",
  "input": {
    "dimensions": 66,
    "port": 6448,
    "address": "/wek/inputs"
  },
  "output": {
    "host": "127.0.0.1",
    "port": 12000
  },
  "tracking_system": "mediapipe-pose-33-xy",
  "coordinate_system": "normalized-0-1-xy",
  "source_fps": 60.0,
  "license": "CC-BY-4.0",
  "creator": "Dance Lab",
  "tags": ["house", "dance", "foundations", "beginner"],
  "gestures": [
    {
      "id": 1,
      "name": "jack",
      "osc_address": "/gesture/1",
      "threshold": 97.5,
      "created_at": "2026-01-21T10:31:00Z",
      "distance_mean": 42.5,
      "distance_std": 15.3,
      "threshold_manual_override": false,
      "threshold_coefficient": 2.0,
      "examples": [
        {
          "recorded_at": "2026-01-21T10:32:00Z",
          "duration_ms": 2850,
          "frame_count": 171,
          "frames": [
            [0.51, 0.32, 0.49, 0.31, ...],
            [0.52, 0.33, 0.48, 0.30, ...],
            ...
          ]
        }
      ]
    }
  ]
}
```
