//! Pacha: Model, Data and Recipe Registry
//!
//! Pacha provides a unified registry for machine learning artifacts—models,
//! datasets, and training recipes—with full lineage tracking, semantic
//! versioning, and cryptographic integrity.
//!
//! # Quick Start
//!
//! ```no_run
//! use pacha::prelude::*;
//!
//! // Open or create the registry
//! let registry = Registry::open_default()?;
//!
//! // Register a model
//! let model_data = std::fs::read("model.apr")?;
//! let card = ModelCard::builder()
//!     .description("Fraud detection model")
//!     .metrics([("auc", 0.95), ("f1", 0.88)])
//!     .build();
//!
//! registry.register_model(
//!     "fraud-detector",
//!     &ModelVersion::new(1, 0, 0),
//!     &model_data,
//!     card,
//! )?;
//!
//! // Retrieve the model
//! let model = registry.get_model("fraud-detector", &ModelVersion::new(1, 0, 0))?;
//! println!("Model stage: {}", model.stage);
//! # Ok::<(), pacha::error::PachaError>(())
//! ```
//!
//! # Architecture
//!
//! Pacha consists of three main registries:
//!
//! - **Model Registry** - `.apr` format files with metadata, metrics, and lineage
//! - **Data Registry** - `.ald` format files with schema and provenance
//! - **Recipe Registry** - TOML configs with hyperparameters and environment specs
//!
//! # Storage
//!
//! Pacha uses content-addressed storage with BLAKE3 hashing for:
//! - Deduplication across versions
//! - Tamper detection
//! - Efficient delta storage
//!
//! Registry metadata is stored in `SQLite` at `~/.pacha/registry.db`.

pub mod aliases;
pub mod cache;
pub mod catalog;
pub mod cli;
pub mod crypto;
pub mod data;
pub mod error;
pub mod experiment;
pub mod fetcher;
pub mod format;
pub mod lineage;
pub mod manifest;
pub mod model;
pub mod prelude;
pub mod recipe;
pub mod registry;
pub mod remote;
pub mod resolver;
pub mod signing;
pub mod storage;
pub mod uri;

pub use error::{PachaError, Result};
pub use registry::{Registry, RegistryConfig, StorageStats};
