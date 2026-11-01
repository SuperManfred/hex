//! Speech models already on this computer: GGUF files in the Hugging Face
//! hub cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
//! `~/.cache/huggingface/hub`). A scan runs only when the model picker or
//! onboarding's model step opens, never in the background, and reads
//! headers alone (`gguf_header`, the same checks as an import). Each file's
//! result is kept per path, size, modification time, and file ID, so an
//! unchanged file is never read twice.
//!
//! Hex never writes or deletes inside the cache. A file whose blob name is a
//! catalog model's pinned SHA-256 (with its size) counts as that model
//! already downloaded, used in place; any other compatible file is offered
//! under "Found on this computer" and, once chosen, is hashed in full once
//! and registered as an in-place import (`custom_models`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::SystemTime;

use super::models::{MODELS, ModelRuntime, TranscriptionModelId};

/// One compatible GGUF found in the cache.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    /// The snapshot path, which Hex reads in place.
    pub path: PathBuf,
    /// `owner/name` from the cache folder (`models--owner--name`).
    pub repository: String,
    /// The snapshot folder: the commit the file was downloaded at.
    pub revision: String,
    pub file_name: String,
    pub bytes: u64,
    /// The blob's name, which the hub cache gives the file's SHA-256; `None`
    /// when the snapshot holds a plain copy (Windows without symlinks).
    pub blob_sha256: Option<String>,
    /// The catalog model this file is, by its pinned SHA-256 and size (or
    /// its repository, file name, and size for a plain copy, checked in full
    /// when chosen).
    pub catalog: Option<TranscriptionModelId>,
    /// `general.name`, else the file name.
    pub name: String,
    pub architecture: String,
    pub variant: String,
    pub languages: Vec<String>,
    /// The quantization most of its weights use: Q8_0, F16, or F32.
    pub compression: &'static str,
    /// The tensor data it loads, which estimates the memory it needs.
    pub tensor_bytes: u64,
}

impl Found {
    /// The commit, shortened as the hub shows it.
    pub fn short_revision(&self) -> &str {
        &self.revision[..self.revision.len().min(7)]
    }

    /// `owner/name@abcdef1`.
    pub fn source(&self) -> String {
        format!("{}@{}", self.repository, self.short_revision())
    }
}

/// The hub cache folder this computer uses, when it exists.
pub fn hub_dir() -> Option<PathBuf> {
    let from_env = |name: &str| {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    let directory = from_env("HF_HUB_CACHE")
        .or_else(|| from_env("HF_HOME").map(|home| home.join("hub")))
        .or_else(|| {
            dirs::home_dir().map(|home| home.join(".cache").join("huggingface").join("hub"))
        })?;
    directory.is_dir().then_some(directory)
}

/// What one file's header said, keyed by what proves it unchanged.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Stamp {
    bytes: u64,
    modified: Option<SystemTime>,
    file_id: Option<u64>,
}

#[derive(Default)]
struct Cache {
    files: HashMap<PathBuf, (Stamp, Option<Found>)>,
    last: Vec<Found>,
}

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(Mutex::default);

/// The models the last scan found, without scanning.
pub fn last() -> Vec<Found> {
    CACHE
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .last
        .clone()
}

#[cfg(test)]
pub(crate) fn set_last_for_test(found: Vec<Found>) {
    CACHE.lock().unwrap_or_else(|error| error.into_inner()).last = found;
}

/// Scans the hub cache now: headers only, and only files that changed since
/// they were last read. Exact catalog matches come first.
pub fn scan() -> Vec<Found> {
    let found = hub_dir().map(|hub| scan_in(&hub)).unwrap_or_default();
    CACHE
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .last
        .clone_from(&found);
    found
}

fn scan_in(hub: &Path) -> Vec<Found> {
    let mut found = Vec::new();
    let Ok(repositories) = std::fs::read_dir(hub) else {
        return found;
    };
    for repository in repositories.flatten() {
        let folder = repository.file_name().to_string_lossy().into_owned();
        let Some(name) = folder.strip_prefix("models--") else {
            continue;
        };
        let repository_name = name.replacen("--", "/", 1);
        let Ok(snapshots) = std::fs::read_dir(repository.path().join("snapshots")) else {
            continue;
        };
        for snapshot in snapshots.flatten() {
            let revision = snapshot.file_name().to_string_lossy().into_owned();
            let Ok(files) = std::fs::read_dir(snapshot.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().is_none_or(|extension| extension != "gguf") {
                    continue;
                }
                if let Some(model) = read(&path, &repository_name, &revision) {
                    found.push(model);
                }
            }
        }
    }
    found.sort_by(|left, right| {
        left.catalog
            .is_none()
            .cmp(&right.catalog.is_none())
            .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
            .then_with(|| left.path.cmp(&right.path))
    });
    found
}

fn stamp(path: &Path) -> Option<Stamp> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(Stamp {
        bytes: metadata.len(),
        modified: metadata.modified().ok(),
        file_id: file_id(&metadata),
    })
}

/// The file's identity on its volume: the inode on Unix; Windows reports
/// none through stable std, so size and modification time stand alone there.
pub(crate) fn file_id(metadata: &std::fs::Metadata) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(metadata.ino() ^ metadata.dev().rotate_left(32))
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        None
    }
}

/// One file's result, from the cache while its stamp holds.
fn read(path: &Path, repository: &str, revision: &str) -> Option<Found> {
    let stamp = stamp(path)?;
    let cache = CACHE.lock().unwrap_or_else(|error| error.into_inner());
    if let Some((cached, found)) = cache.files.get(path)
        && *cached == stamp
    {
        return found.clone();
    }
    drop(cache);
    let found = inspect(path, repository, revision, stamp.bytes);
    CACHE
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .files
        .insert(path.to_path_buf(), (stamp, found.clone()));
    found
}

fn inspect(path: &Path, repository: &str, revision: &str, bytes: u64) -> Option<Found> {
    let file_name = path.file_name()?.to_string_lossy().into_owned();
    let blob_sha256 = std::fs::read_link(path)
        .ok()
        .and_then(|target| {
            target
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .filter(|name| name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(|name| name.to_lowercase());
    let catalog = catalog_match(blob_sha256.as_deref(), repository, &file_name, bytes);
    let header = match super::gguf_header::read(path) {
        Ok(header) => header,
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "skipping a cached file Hex can't read");
            return None;
        }
    };
    let profile = match super::custom_models::profile_of(&header, &file_name) {
        Ok(profile) => profile,
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "skipping a cached model Hex can't run");
            return None;
        }
    };
    Some(Found {
        path: path.to_path_buf(),
        repository: repository.to_owned(),
        revision: revision.to_owned(),
        file_name,
        bytes,
        blob_sha256,
        catalog,
        name: profile.name,
        architecture: profile.architecture,
        variant: profile.variant,
        languages: profile.languages,
        compression: compression(&header),
        tensor_bytes: header.tensor_bytes,
    })
}

/// The catalog model a cached file is: its blob name is the pinned SHA-256
/// and its size matches, or (a plain copy) its repository, file name, and
/// size match the pin.
fn catalog_match(
    blob_sha256: Option<&str>,
    repository: &str,
    file_name: &str,
    bytes: u64,
) -> Option<TranscriptionModelId> {
    MODELS
        .iter()
        .find(|model| {
            let ModelRuntime::Gguf(artifact) = model.runtime else {
                return false;
            };
            artifact.bytes == bytes
                && match blob_sha256 {
                    Some(sha256) => sha256 == artifact.sha256,
                    None => artifact.repository == repository && artifact.filename == file_name,
                }
        })
        .map(|model| model.id)
}

fn compression(header: &super::gguf_header::GgufHeader) -> &'static str {
    let count = |kind| header.tensor_types.get(&kind).copied().unwrap_or(0);
    if count(8) > 0 {
        "Q8_0"
    } else if count(1) > 0 {
        "F16"
    } else {
        "F32"
    }
}
