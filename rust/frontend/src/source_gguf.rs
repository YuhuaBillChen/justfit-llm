//! Local GGUF source assembly and tokenizer preparation, entirely in Rust.
use crate::{
    cli,
    download::{Result, Stage},
    gguf_metadata::Metadata,
    source_mlx,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};
fn source(path: &Path) -> Result<PathBuf> {
    let path = std::path::absolute(path)?;
    Ok(path
        .parent()
        .ok_or("source parent missing")?
        .canonicalize()?
        .join(path.file_name().ok_or("source name missing")?))
}
fn split_name(name: &str) -> Option<(&str, u32, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let (prefix, total) = stem.rsplit_once("-of-")?;
    let (model, part) = prefix.rsplit_once('-')?;
    if part.len() != 5
        || total.len() != 5
        || !part
            .bytes()
            .chain(total.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some((model, part.parse().ok()?, total.parse().ok()?))
}
pub fn ordered(names: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if names.is_empty() {
        return Err("source has no GGUF target".into());
    }
    let mut parts = BTreeMap::new();
    let mut identity = None;
    for path in &names {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("invalid GGUF name")?;
        let Some((model, part, total)) = split_name(name) else {
            if names.len() == 1 {
                return Ok(names);
            }
            return Err("source names several GGUF targets; select the target file".into());
        };
        if total == 0 || part == 0 || part > total {
            return Err("invalid GGUF split part".into());
        }
        if identity
            .as_ref()
            .is_some_and(|v| v != &(model.to_string(), total))
        {
            return Err("GGUF split files describe different targets".into());
        }
        identity = Some((model.to_string(), total));
        if parts.insert(part, path.clone()).is_some() {
            return Err("duplicate GGUF split part".into());
        }
    }
    let total = identity.unwrap().1;
    if parts.len() != total as usize || (1..=total).any(|p| !parts.contains_key(&p)) {
        return Err("incomplete GGUF split".into());
    }
    Ok(parts.into_values().collect())
}
fn targets(target: &Path) -> Result<Vec<PathBuf>> {
    if target.is_dir() {
        let mut paths = vec![];
        for entry in fs::read_dir(target)? {
            let path = entry?.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if path.is_file() && name.ends_with(".gguf") && !name.to_lowercase().contains("mmproj")
            {
                paths.push(path);
            }
        }
        paths.sort();
        return ordered(paths);
    }
    if !target.is_file() || target.extension().is_none_or(|e| e != "gguf") {
        return Err("target is not a GGUF file or directory".into());
    }
    let name = target
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("invalid GGUF filename")?;
    let Some((model, _, _)) = split_name(name) else {
        return Ok(vec![target.to_path_buf()]);
    };
    let mut paths = vec![];
    for entry in fs::read_dir(target.parent().ok_or("target parent missing")?)? {
        let path = entry?.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(split_name)
                .is_some_and(|(m, _, _)| m == model)
        {
            paths.push(path);
        }
    }
    ordered(paths)
}
fn check(
    project: &Path,
    vision: bool,
    config: &Path,
    metadata: &Path,
    draft: Option<&Path>,
) -> Result<String> {
    let mut command = std::process::Command::new(project.join("source/build/justfit-engine"));
    command
        .args(["model-check", "gguf", if vision { "gguf" } else { "none" }])
        .arg(config)
        .arg(metadata);
    if let Some(draft) = draft {
        command.arg(draft);
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "GGUF model check: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let value: Value = serde_json::from_slice(&output.stdout)?;
    let family = value["family"]
        .as_str()
        .filter(|f| matches!(*f, "Qwen3.8-27B" | "Qwen3.6-35B-A3B"))
        .ok_or("no known DFlash2 draft for GGUF family")?;
    Ok(family.into())
}
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().ok_or("derived path parent missing")?)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
pub fn prepare(
    project: &Path,
    target: &Path,
    draft: Option<&Path>,
    mmproj: Option<&Path>,
    language_only: bool,
) -> Result<PathBuf> {
    if language_only && mmproj.is_some() {
        return Err("--language-only cannot load a vision projector".into());
    }
    let target = source(target)?;
    let paths = targets(&target)?;
    let headers = paths
        .iter()
        .map(|p| Metadata::open(p, true))
        .collect::<Result<Vec<_>>>()?;
    let header = Metadata::merge(headers)?;
    let mut config = header.model_config()?;
    // Temporary configuration holds only metadata. No weights or owner files are
    // copied or changed while the native planner checks family/draft rules.
    let stage = Stage::new(&std::env::temp_dir(), "justfit-gguf-inspection")?;
    let config_path = stage.path().join("config.json");
    let metadata_path = stage.path().join("gguf-metadata.json");
    write(&config_path, &source_mlx::json_bytes(config.clone())?)?;
    write(
        &metadata_path,
        &source_mlx::json_bytes(header.scalar_metadata())?,
    )?;
    let family = check(project, false, &config_path, &metadata_path, None)?;
    header.require_loadable()?;
    let projector = mmproj.map(source).transpose()?;
    if let Some(path) = &projector {
        if paths.contains(path) {
            return Err("vision projector is also a target file".into());
        }
        let vision = Metadata::open(path, true)?;
        if vision.tensors.is_empty()
            || vision
                .tensors
                .values()
                .any(|v| !matches!(v.kind, 0 | 1 | 30))
        {
            return Err("GGUF projector must use F32, F16 or BF16 tensors".into());
        }
        let config_vision = vision.vision_config()?;
        if config_vision["patch_size"] != 16
            || config_vision["spatial_merge_size"] != 2
            || vision.values.get("clip.vision.image_mean") != Some(&json!([0.5, 0.5, 0.5]))
            || vision.values.get("clip.vision.image_std") != Some(&json!([0.5, 0.5, 0.5]))
        {
            return Err("unsupported GGUF vision preprocessing configuration".into());
        }
        config["vision_config"] = config_vision;
        fs::write(&config_path, source_mlx::json_bytes(config.clone())?)?;
        check(project, true, &config_path, &metadata_path, None)?;
    }
    let draft = if let Some(path) = draft {
        source(path)?
    } else {
        crate::model_install::cached_source_draft(&family)?
    };
    if !draft.is_dir() {
        return Err("DFlash2 draft must be a directory".into());
    }
    check(
        project,
        projector.is_some(),
        &config_path,
        &metadata_path,
        Some(&draft.join("config.json")),
    )?;
    let mut files = BTreeMap::new();
    for path in &paths {
        files.insert(
            format!(
                "target/{}",
                path.file_name()
                    .and_then(|s| s.to_str())
                    .ok_or("invalid GGUF filename")?
            ),
            path.clone(),
        );
    }
    if let Some(path) = &projector {
        files.insert("vision/mmproj.gguf".into(), path.clone());
    }
    files.insert("draft/config.json".into(), draft.join("config.json"));
    for name in source_mlx::shards(&draft, false, false)? {
        files.insert(format!("draft/{name}"), draft.join(name));
    }
    if draft.join("mask_embedding.safetensors").is_file() {
        files.insert(
            "draft/mask_embedding.safetensors".into(),
            draft.join("mask_embedding.safetensors"),
        );
    }
    let mut derived = header.tokenizer_files()?;
    derived.insert("config.json".into(), source_mlx::json_bytes(config)?);
    let store = project.join("local-state/models");
    source_mlx::safe_destination(&store)?;
    fs::create_dir_all(&store)?;
    let metadata_root = store.join(".metadata");
    source_mlx::safe_destination(&metadata_root)?;
    fs::create_dir_all(&metadata_root)?;
    let mut identities = vec![];
    for path in paths.iter().chain(projector.iter()) {
        identities.push(json!({"bytes":path.metadata()?.len(),"sha256":cli::digest(path)?}));
    }
    let identity = json!({"schema":1,"adapter_sha256":format!("{:x}",Sha256::digest(include_bytes!("gguf_metadata.rs"))),"tokenizers":"0.22.2","sources":identities});
    let key = format!("{:x}", Sha256::digest(source_mlx::json_bytes(identity)?));
    let destination = metadata_root.join(&key);
    source_mlx::safe_destination(&destination)?;
    if destination.exists() {
        for (name, bytes) in &derived {
            let path = destination.join(name);
            if path.is_symlink() || fs::read(path)? != *bytes {
                return Err("existing derived GGUF metadata differs; owner files preserved".into());
            }
        }
    } else {
        let prepared = Stage::new(&metadata_root, "gguf-derived")?;
        for (name, bytes) in &derived {
            write(&prepared.path().join(name), bytes)?;
        }
        match crate::model_install::rename_exclusive(prepared.path(), &destination) {
            Ok(()) => (),
            Err(error) if destination.is_dir() => {
                for (name, bytes) in &derived {
                    if fs::read(destination.join(name))? != *bytes {
                        return Err(error);
                    }
                }
            }
            Err(error) => return Err(error),
        }
    }
    for name in derived.keys() {
        files.insert(name.clone(), destination.join(name));
    }
    let selection =
        json!({"target":target,"draft":draft,"mmproj":projector,"language_only":language_only});
    let base = json!({"version":1,"model":target,"family":family,"target_format":"gguf","vision_format":if projector.is_some(){"gguf"}else{"none"},"metadata":key,"sources":{"target":{"repo":target,"revision":null},"draft":{"repo":draft,"revision":null}}});
    source_mlx::publish(&store, selection, base, files)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_parts_are_ordered_and_missing_or_mixed_parts_refused() {
        let p = |n: &str| PathBuf::from(n);
        assert_eq!(
            ordered(vec![p("M-00002-of-00002.gguf"), p("M-00001-of-00002.gguf")]).unwrap(),
            vec![p("M-00001-of-00002.gguf"), p("M-00002-of-00002.gguf")]
        );
        for names in [
            vec![p("M-00001-of-00002.gguf")],
            vec![p("M-00001-of-00002.gguf"), p("X-00002-of-00002.gguf")],
            vec![p("M-00001-of-00001.gguf"), p("M-00001-of-00001.gguf")],
        ] {
            assert!(ordered(names).is_err());
        }
    }
}
