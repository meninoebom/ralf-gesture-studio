# Rust Code Patterns

When writing Rust code for this project, follow these patterns for clean, maintainable, and modular code.

## 1. Separate Configuration from State

**Configuration structs** hold user-adjustable settings as pure data:

```rust
/// Configuration for [feature] - pure data, no behavior
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Description of what this controls
    pub setting_name: f32,
    /// Another setting
    pub another_setting: u32,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            setting_name: 3.0,
            another_setting: 5,
        }
    }
}
```

**State enums** track workflow/UI state separately:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureState {
    Idle,
    Processing,
    Complete,
}
```

**Why:** Config can be reused across different UIs (CLI, GUI, web). State is presentation-specific.

## 2. Module Organization

```
src/
├── engine/          # Core logic, no UI dependencies
│   ├── mod.rs       # Public exports
│   ├── feature.rs   # Feature implementation
│   └── config.rs    # Config structs (or inline in feature.rs)
├── gui/             # UI-specific code
│   └── mod.rs       # GUI implementation
└── model/           # Data structures, persistence
    └── mod.rs
```

**Engine module exports** - be explicit about public API:

```rust
// engine/mod.rs
pub mod feature;

// Re-export commonly used types
pub use feature::{FeatureConfig, FeatureState, FeatureManager};
```

## 3. GUI Holds Config + State, Not Raw Primitives

```rust
// BAD - scattered primitives
pub struct App {
    feature_timeout: f32,
    feature_enabled: bool,
    feature_count: u32,
}

// GOOD - grouped config struct
pub struct App {
    feature_config: FeatureConfig,
    feature_state: FeatureState,
}
```

## 4. UI Controls Bind to Config Fields

```rust
// The config struct fields are what the UI manipulates
ui.add(egui::DragValue::new(&mut self.feature_config.timeout_secs)
    .range(1.0..=10.0)
    .suffix("s"));
```

## 5. State Transitions in Processing, Not UI

```rust
// In frame processing or update loop - NOT in UI rendering
if let FeatureState::Countdown { start_time } = self.feature_state {
    if start_time.elapsed().as_secs_f32() >= self.feature_config.countdown_secs {
        self.feature_state = FeatureState::Active { /* ... */ };
    }
}
```

## 6. Config Structs Should Be:

- **Serializable** - Add `#[derive(Serialize, Deserialize)]` if saving to files
- **Cloneable** - `#[derive(Clone)]` for easy copying
- **Debuggable** - `#[derive(Debug)]` for logging
- **Defaultable** - `impl Default` with sensible values
- **Documented** - Doc comments on each field

## 7. Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Config struct | `{Feature}Config` | `TrainingConfig`, `BaselineConfig` |
| State enum | `{Feature}State` | `SessionState`, `BaselineState` |
| Manager/Session | `{Feature}Session` or `{Feature}Manager` | `TrainingSession` |
| Duration fields | `{name}_secs` | `countdown_secs`, `duration_secs` |
| Count fields | `{name}_count` or just `{plural}` | `reps`, `frame_count` |

## 8. Example: Adding a New Configurable Feature

1. **Create config struct** in `engine/`:
   ```rust
   pub struct NewFeatureConfig {
       pub option_a: f32,
       pub option_b: bool,
   }
   impl Default for NewFeatureConfig { ... }
   ```

2. **Export from engine/mod.rs**:
   ```rust
   pub use feature::{NewFeatureConfig};
   ```

3. **Add to GUI app struct**:
   ```rust
   use crate::engine::NewFeatureConfig;

   pub struct App {
       new_feature_config: NewFeatureConfig,
   }
   ```

4. **Initialize with default**:
   ```rust
   new_feature_config: NewFeatureConfig::default(),
   ```

5. **Bind UI controls to config fields**:
   ```rust
   ui.add(egui::Slider::new(&mut self.new_feature_config.option_a, 0.0..=10.0));
   ```

---

Use these patterns when:
- Adding new user-configurable settings
- Creating new workflow states
- Refactoring scattered primitives into cohesive structs
