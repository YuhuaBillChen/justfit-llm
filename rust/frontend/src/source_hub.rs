//! Commit-pinned source-model downloads using the existing Rust TLS/resume path.
use crate::{
    cli,
    download::{self, Result, Stage},
    model_install as install, source_mlx,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    cell::RefCell,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};
fn hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit())
}
fn encode(s: &str) -> String {
    percent_encoding::utf8_percent_encode(s, percent_encoding::NON_ALPHANUMERIC).to_string()
}
#[derive(Clone)]
struct Entry {
    bytes: u64,
    digest: String,
}
type SourceStamp=(u64,u64,u64,i128,i128);
fn source_stamp(m:&fs::Metadata)->SourceStamp{(m.dev(),m.ino(),m.len(),m.mtime()as i128*1_000_000_000+m.mtime_nsec()as i128,m.ctime()as i128*1_000_000_000+m.ctime_nsec()as i128)}
struct Repository {
    id: String,
    commit: String,
    root: PathBuf,
    snapshot: PathBuf,
    files: BTreeMap<String, Entry>,
    cached: bool,
    verified:RefCell<BTreeMap<String,SourceStamp>>,
}
impl Repository {
    fn open(project: &Path, id: &str, revision: &str) -> Result<Self> {
        install::repository(id)?;
        if !install::safe_path(revision) {
            return Err("invalid source revision".into());
        }
        let root = install::repository_cache(project, id);
        if install::offline() {
            return Self::cached(
                id,
                revision,
                &[
                    root,
                    install::hub_cache()?.join(format!("models--{}", id.replace('/', "--"))),
                ],
            );
        }
        let credential = install::token();
        let url = format!(
            "https://huggingface.co/api/models/{id}/revision/{}?blobs=true",
            encode(revision)
        );
        let tls = ureq::tls::TlsConfig::builder()
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build();
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .https_only(true)
            .tls_config(tls)
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build()
            .into();
        let info =
            match download::metadata_with(&agent, &url, 64 * 1024 * 1024, credential.as_deref()) {
                Ok(bytes) => serde_json::from_slice::<Value>(&bytes)?,
                Err(error) => {
                    return Self::cached(
                        id,
                        revision,
                        &[
                            root,
                            install::hub_cache()?
                                .join(format!("models--{}", id.replace('/', "--"))),
                        ],
                    )
                    .map_err(|_| error)
                }
            };
        let commit = info["sha"]
            .as_str()
            .filter(|s| hex(s, 40))
            .ok_or("source Hub commit missing")?
            .to_ascii_lowercase();
        if hex(revision, 40) && !revision.eq_ignore_ascii_case(&commit) {
            return Err("source Hub commit differs from revision".into());
        }
        let mut files = BTreeMap::new();
        for row in info["siblings"]
            .as_array()
            .ok_or("source Hub file listing missing")?
        {
            let name = row["rfilename"]
                .as_str()
                .filter(|n| install::safe_path(n))
                .ok_or("unsafe source Hub file path")?;
            let bytes = row["size"].as_u64().ok_or("source Hub file size missing")?;
            let digest = row["lfs"]["sha256"]
                .as_str()
                .filter(|s| hex(s, 64))
                .or_else(|| row["blobId"].as_str().filter(|s| hex(s, 40)))
                .ok_or("source Hub file digest missing")?
                .to_ascii_lowercase();
            if files
                .insert(name.to_string(), Entry { bytes, digest })
                .is_some()
            {
                return Err("duplicate source Hub file path".into());
            }
        }
        if files.is_empty() || files.len() > 100_000 {
            return Err("invalid source Hub file count".into());
        }
        // Reuse an existing pinned Hub snapshot before allocating another full
        // source checkpoint in the workspace. Every selected file is verified.
        let hub=install::hub_cache()?.join(format!("models--{}",id.replace('/',"--")));
        let root=if !root.join("snapshots").join(&commit).is_dir() && hub.join("snapshots").join(&commit).is_dir(){hub}else{root};
        source_mlx::safe_destination(&root)?;
        fs::create_dir_all(root.join("blobs"))?;
        let snapshot = root.join("snapshots").join(&commit);
        source_mlx::safe_destination(&snapshot)?;
        fs::create_dir_all(&snapshot)?;
        Ok(Self {
            id: id.into(),
            commit,
            root,
            snapshot,
            files,
            cached: false,
            verified:RefCell::new(BTreeMap::new()),
        })
    }
    fn cached(id: &str, revision: &str, roots: &[PathBuf]) -> Result<Self> {
        for root in roots {
            let commit = if hex(revision, 40) {
                revision.to_ascii_lowercase()
            } else {
                match fs::read_to_string(root.join("refs").join(revision)) {
                    Ok(s) => s.trim().into(),
                    Err(_) => continue,
                }
            };
            if !hex(&commit, 40) {
                continue;
            }
            let snapshot = root.join("snapshots").join(&commit);
            if !snapshot.is_dir() {
                continue;
            }
            let mut paths = vec![];
            fn walk(root: &Path, path: &Path, paths: &mut Vec<(String, PathBuf)>) -> Result<()> {
                for entry in fs::read_dir(path)? {
                    let p = entry?.path();
                    let meta = p.symlink_metadata()?;
                    if meta.is_dir() {
                        walk(root, &p, paths)?;
                    } else if p.is_file() {
                        let name = p
                            .strip_prefix(root)?
                            .to_str()
                            .filter(|s| install::safe_path(s))
                            .ok_or("unsafe cached source filename")?;
                        paths.push((name.to_string(), p));
                        if paths.len() > 100_000 {
                            return Err("too many cached source files".into());
                        }
                    }
                }
                Ok(())
            }
            walk(&snapshot, &snapshot, &mut paths)?;
            let mut files = BTreeMap::new();
            for (name, path) in paths {
                let resolved = path.canonicalize()?;
                let filename = resolved.file_name().and_then(|s| s.to_str()).unwrap_or("");
                let digest = if resolved
                    .parent()
                    .and_then(Path::file_name)
                    .is_some_and(|p| p == "blobs")
                    && (hex(filename, 40) || hex(filename, 64))
                {
                    filename.into()
                } else {
                    cli::digest(&path)?
                };
                files.insert(
                    name,
                    Entry {
                        bytes: path.metadata()?.len(),
                        digest,
                    },
                );
            }
            return Ok(Self {
                id: id.into(),
                commit,
                root: root.clone(),
                snapshot,
                files,
                cached: true,
                verified:RefCell::new(BTreeMap::new()),
            });
        }
        Err("source model is not present in offline Hub caches".into())
    }
    fn url(&self, name: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.id,
            self.commit,
            name.split('/').map(encode).collect::<Vec<_>>().join("/")
        )
    }
    fn header(&self, name: &str) -> Result<crate::gguf_metadata::Metadata> {
        let path = self.snapshot.join(name);
        if path.is_file() {
            return crate::gguf_metadata::Metadata::open(&path, true);
        }
        if self.cached {
            return Err("offline GGUF source missing".into());
        }
        if !self.files.contains_key(name) {
            return Err("GGUF source metadata missing".into());
        }
        download::inspect_prefix(&self.url(name), install::token().as_deref(), |input| {
            crate::gguf_metadata::Metadata::read(input, true)
        })
    }
    fn verify_file(&self,name:&str,path:&Path,entry:&Entry)->Result<()> {
        let before=fs::metadata(path)?;
        if !before.is_file()||before.len()!=entry.bytes{return Err("cached source file size mismatch; owner data preserved".into());}
        let stamp=source_stamp(&before);
        if self.verified.borrow().get(name)==Some(&stamp){return Ok(());}
        let digest=if entry.digest.len()==64{cli::digest(path)?}else{git_blob_digest(path)?};
        if digest!=entry.digest{return Err("cached source checksum mismatch; owner data preserved".into());}
        if source_stamp(&fs::metadata(path)?)!=stamp{return Err("cached source changed while verifying; owner data preserved".into());}
        self.verified.borrow_mut().insert(name.into(),stamp);Ok(())
    }
    fn file(&self,name:&str)->Result<PathBuf>{self.file_with(name,download::metadata,download::fetch_resumable_authenticated)}
    fn file_with(&self,name:&str,small:impl FnOnce(&str,u64,Option<&str>)->Result<Vec<u8>>,large:impl FnOnce(&str,&Path,u64,&str,Option<&str>)->Result<u64>)->Result<PathBuf>{
        let entry = self
            .files
            .get(name)
            .ok_or_else(|| format!("source repository {} is missing {name}", self.id))?;
        let path = self.snapshot.join(name);
        if path.is_file() {
            self.verify_file(name,&path,entry)?;
            return Ok(path);
        }
        if self.cached {
            return Err("cached source file is missing; offline downloads disabled".into());
        }
        source_mlx::safe_destination(&self.root)?;
        let _lock = install::Lock::take(&self.root)?;
        let blob = self.root.join("blobs").join(&entry.digest);
        source_mlx::safe_destination(&blob)?;
        if !blob.exists() {
            let stage = Stage::new(&self.root, "source-transfer")?;
            let credential = install::token();
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                self.id,
                self.commit,
                name.split('/').map(encode).collect::<Vec<_>>().join("/")
            );
            if entry.digest.len() == 64 {
                let parts = self.root.join(format!(".source-parts-{}", self.commit));
                source_mlx::safe_destination(&parts)?;
                fs::create_dir_all(&parts)?;
                let part = parts.join(&entry.digest);
                let owner = parts.join(format!("{}.owner.json", entry.digest));
                let expected = json!({"repo":self.id,"commit":self.commit,"digest":entry.digest,"bytes":entry.bytes});
                if owner.exists() {
                    if source_mlx::read_json(&owner)? != expected {
                        return Err("foreign source transfer receipt; owner data preserved".into());
                    }
                } else {
                    if part.exists() {
                        return Err("source partial has no owner receipt; preserved".into());
                    }
                    let mut output = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&owner)?;
                    output.write_all(&source_mlx::json_bytes(expected)?)?;
                    output.sync_all()?;
                }
                large(
                    &url,
                    &part,
                    entry.bytes,
                    &entry.digest,
                    credential.as_deref(),
                )?;
                install::rename_exclusive(&part, &blob)?;
            } else {
                if entry.bytes > 64 * 1024 * 1024 {
                    return Err("Git source file exceeds metadata limit".into());
                }
                let bytes = small(&url, entry.bytes, credential.as_deref())?;
                if bytes.len() as u64 != entry.bytes {
                    return Err("Git source file size mismatch".into());
                }
                let mut hash = ring::digest::Context::new(&ring::digest::SHA1_FOR_LEGACY_USE_ONLY);
                hash.update(format!("blob {}\0", entry.bytes).as_bytes());
                hash.update(&bytes);
                let digest = hash
                    .finish()
                    .as_ref()
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();
                if digest != entry.digest {
                    return Err("Git source blob SHA1 mismatch".into());
                }
                let temporary = stage.path().join("blob");
                let mut output = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                output.write_all(&bytes)?;
                output.sync_all()?;
                install::rename_exclusive(&temporary, &blob)?;
            }
        } else if blob.metadata()?.len() != entry.bytes {
            return Err("cached source blob size mismatch; owner data preserved".into());
        }
        source_mlx::safe_destination(path.parent().ok_or("source snapshot parent missing")?)?;
        fs::create_dir_all(path.parent().unwrap())?;
        if !path.exists() {
            let stage = Stage::new(&self.root, "source-link")?;
            let temporary = stage.path().join("link");
            std::os::unix::fs::symlink(&blob, &temporary)?;
            match install::rename_exclusive(&temporary, &path) {
                Ok(()) => (),
                Err(_) if path.is_file() && path.canonicalize()? == blob => (),
                Err(error) => return Err(error),
            }
        }
        self.verify_file(name,&path,entry)?;
        Ok(path)
    }
    fn pin(&self, project: &Path) -> Result<()> {
        use sha2::{Digest, Sha256};
        let root = self.root.canonicalize()?;
        source_mlx::safe_destination(&root)?;
        let _lock = install::Lock::take(&root)?;
        let owner = format!(
            "{:x}",
            Sha256::digest(project.canonicalize()?.as_os_str().as_encoded_bytes())
        );
        install::remember_revision(
            &root,
            &format!("justfit/{owner}/{}", self.commit),
            &self.commit,
        )
    }
    fn finish(&self, revision: &str) -> Result<()> {
        if self.cached || hex(revision, 40) {
            return Ok(());
        }
        let _lock = install::Lock::take(&self.root)?;
        install::remember_revision(&self.root, revision, &self.commit)
    }
}
fn checkpoint(repo: &Repository, vision: bool) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    if repo.files.contains_key("model.safetensors.index.json") {
        let index = source_mlx::read_json(&repo.file("model.safetensors.index.json")?)?;
        let map = index["weight_map"]
            .as_object()
            .filter(|m| !m.is_empty())
            .ok_or("invalid source safetensors shard index")?;
        for (tensor, name) in map {
            let name = name.as_str().ok_or("invalid source shard name")?;
            if name.contains('/')
                || !install::safe_path(name)
                || !name.ends_with(".safetensors")
                || !repo.files.contains_key(name)
            {
                return Err("invalid or missing source shard".into());
            }
            if vision
                || !["vision_tower.", "model.visual."]
                    .iter()
                    .any(|p| tensor.starts_with(p))
            {
                names.insert(name.to_string());
            }
        }
    } else if repo.files.contains_key("model.safetensors") {
        names.insert("model.safetensors".into());
    } else {
        return Err("source has no safetensors checkpoint".into());
    }
    Ok(names)
}
fn has_vision(repo: &Repository) -> Result<bool> {
    let keys = if repo.files.contains_key("model.safetensors.index.json") {
        let value = source_mlx::read_json(&repo.file("model.safetensors.index.json")?)?;
        value["weight_map"]
            .as_object()
            .ok_or("invalid safetensors shard index")?
            .keys()
            .cloned()
            .collect::<Vec<_>>()
    } else {
        let inspect = |input: &mut dyn std::io::Read| -> Result<Vec<String>> {
            let mut bytes = [0; 8];
            input.read_exact(&mut bytes)?;
            let size = u64::from_le_bytes(bytes);
            if !(2..=100_000_000).contains(&size) {
                return Err("invalid safetensors header length".into());
            }
            let mut bytes = vec![0; size as usize];
            input.read_exact(&mut bytes)?;
            let value: Value = serde_json::from_slice(&bytes)?;
            Ok(value
                .as_object()
                .ok_or("invalid safetensors header")?
                .keys()
                .cloned()
                .collect())
        };
        let path = repo.snapshot.join("model.safetensors");
        if path.is_file() {
            inspect(&mut fs::File::open(path)?)?
        } else {
            if repo.cached {
                return Err("offline safetensors checkpoint missing".into());
            }
            download::inspect_prefix(
                &repo.url("model.safetensors"),
                install::token().as_deref(),
                inspect,
            )?
        }
    };
    Ok(keys
        .iter()
        .any(|name| name.starts_with("vision_tower.") || name.starts_with("model.visual.")))
}
fn native_family(
    project: &Path,
    format: &str,
    vision: bool,
    config: &Path,
    metadata: Option<&Path>,
    draft: Option<&Path>,
) -> Result<String> {
    let vision = if !vision {
        "none"
    } else if format == "gguf" {
        "gguf"
    } else {
        "safetensors"
    };
    let mut command = std::process::Command::new(project.join("source/build/justfit-engine"));
    command.args(["model-check", format, vision]).arg(config);
    if let Some(metadata) = metadata {
        command.arg(metadata);
    }
    if let Some(draft) = draft {
        command.arg(draft);
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "source model check: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let value: Value = serde_json::from_slice(&output.stdout)?;
    Ok(value["family"]
        .as_str()
        .filter(|f| matches!(*f, "Qwen3.8-27B" | "Qwen3.6-35B-A3B"))
        .ok_or("no known draft for source family")?
        .into())
}
pub fn resolve(
    project: &Path,
    spec: &str,
    revision: Option<&str>,
    draft: Option<&str>,
    language_only: bool,
) -> Result<PathBuf> {
    let (id, variant) = spec
        .split_once(':')
        .map_or((spec, None), |(id, v)| (id, Some(v)));
    let revision = install::source_revision(id, revision);
    let repo = Repository::open(project, id, revision)?;
    if repo.files.contains_key("manifest.json") {
        if variant.is_some() || draft.is_some() {
            return Err("packed packages already select variant and draft".into());
        }
        return install::resolve(project, id, Some(revision));
    }
    let temporary = Stage::new(&std::env::temp_dir(), "source-inspection")?;
    let mut required = BTreeSet::new();
    let mlx = repo.files.contains_key("model.safetensors")
        || repo.files.contains_key("model.safetensors.index.json");
    let config_path;
    let metadata_path;
    let target;
    let format;
    if mlx {
        if variant.is_some() {
            return Err("MLX source has no GGUF variants".into());
        }
        format = "mlx-affine";
        metadata_path = None;
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            repo.file(name)?;
            required.insert(name.to_string());
        }
        config_path = repo.file("config.json")?;
        for name in [
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "processor_config.json",
        ] {
            if repo.files.contains_key(name) {
                required.insert(name.into());
                repo.file(name)?;
            }
        }
        native_family(project, format, false, &config_path, None, None)?;
        if !language_only {
            source_mlx::processor(&repo.snapshot)?;
            if !has_vision(&repo)? {
                return Err("source target has no vision tower; use --language-only".into());
            }
            native_family(project, format, true, &config_path, None, None)?;
        }
        required.extend(checkpoint(&repo, !language_only)?);
        target = repo.snapshot.clone();
    } else {
        format = "gguf";
        let names = select_gguf(repo.files.keys().cloned().collect(), variant)?;
        let headers = names
            .iter()
            .map(|n| repo.header(n))
            .collect::<Result<Vec<_>>>()?;
        let header = crate::gguf_metadata::Metadata::merge(headers)?;
        let config = header.model_config()?;
        header.require_loadable()?;
        header.tokenizer_files()?;
        config_path = temporary.path().join("config.json");
        let scalars = temporary.path().join("metadata.json");
        fs::write(&config_path, source_mlx::json_bytes(config)?)?;
        fs::write(&scalars, source_mlx::json_bytes(header.scalar_metadata())?)?;
        metadata_path = Some(scalars);
        target = repo.snapshot.join(&names[0]);
        required.extend(names);
    }
    let family = native_family(
        project,
        format,
        false,
        &config_path,
        metadata_path.as_deref(),
        None,
    )?;
    let projector = if !language_only && !mlx {
        let mut candidates = vec![];
        for name in repo.files.keys().filter(|n| {
            !n.contains('/') && n.ends_with(".gguf") && n.to_lowercase().contains("mmproj")
        }) {
            let meta = repo.header(name)?;
            if meta.vision_config().is_err()
                || meta.tensors.is_empty()
                || meta.values.get("clip.vision.image_mean") != Some(&json!([0.5, 0.5, 0.5]))
                || meta.values.get("clip.vision.image_std") != Some(&json!([0.5, 0.5, 0.5]))
            {
                continue;
            }
            let types = meta
                .tensors
                .values()
                .map(|t| t.kind)
                .collect::<BTreeSet<_>>();
            for (priority, kind) in [(0, 30), (1, 0), (2, 1)] {
                if types == BTreeSet::from([kind]) {
                    candidates.push((priority, name.clone(), meta.vision_config()?));
                }
            }
        }
        candidates.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        if candidates.len() > 1 && candidates[0].0 == candidates[1].0 {
            return Err("several compatible GGUF vision projectors".into());
        }
        let (_, name, vision) = candidates
            .into_iter()
            .next()
            .ok_or("no supported vision projector; use --language-only")?;
        let mut config = source_mlx::read_json(&config_path)?;
        config["vision_config"] = vision;
        fs::write(&config_path, source_mlx::json_bytes(config)?)?;
        required.insert(name.clone());
        Some(repo.snapshot.join(name))
    } else {
        None
    };
    let default = match family.as_str() {
        "Qwen3.8-27B" => install::DEFAULT_DRAFT,
        "Qwen3.6-35B-A3B" => "incoai/Qwen3.6-35B-A3B-DFlash2",
        _ => return Err("no known source draft".into()),
    };
    let local = draft.map(install::local_spec).transpose()?;
    let draft_revision=if draft.is_none() && default==install::DEFAULT_DRAFT {install::DEFAULT_DRAFT_REVISION}else{"main"};
    let draft_repo = if local.as_ref().is_some_and(|p| p.is_dir()) {
        None
    } else {
        Some(Repository::open(project, draft.unwrap_or(default), draft_revision)?)
    };
    let draft_path = if let Some(repo) = &draft_repo {
        repo.file("config.json")?;
        repo.snapshot.clone()
    } else {
        local.unwrap()
    };
    native_family(
        project,
        format,
        !language_only,
        &config_path,
        metadata_path.as_deref(),
        Some(&draft_path.join("config.json")),
    )?;
    // No weight transfer precedes target/vision/draft configuration acceptance.
    repo.pin(project)?;
    if let Some(repo) = &draft_repo {
        repo.pin(project)?;
    }
    for name in &required {
        repo.file(name)?;
    }
    if let Some(repo) = &draft_repo {
        for name in checkpoint(repo, true)? {
            repo.file(&name)?;
        }
        if repo.files.contains_key("mask_embedding.safetensors") {
            repo.file("mask_embedding.safetensors")?;
        }
    }
    let prepared = source_mlx::prepare_local(
        project,
        &target,
        Some(&draft_path),
        projector.as_deref(),
        language_only,
    )?;
    let mut base = source_mlx::read_json(&prepared.join("model.json"))?;
    base["model"] = json!(spec);
    base["sources"]["target"] = json!({"repo":id,"revision":repo.commit});
    if let Some(draft) = &draft_repo {
        base["sources"]["draft"] = json!({"repo":draft.id,"revision":draft.commit});
    }
    let files = base["files"]
        .as_object()
        .ok_or("prepared source file records missing")?
        .iter()
        .map(|(n, v)| {
            Ok((
                n.clone(),
                PathBuf::from(v["path"].as_str().ok_or("source path missing")?),
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let selection =
        json!({"model":spec,"revision":revision,"draft":draft,"language_only":language_only});
    let model = source_mlx::publish(&project.join("local-state/models"), selection, base, files)?;
    repo.finish(revision)?;
    if let Some(draft) = draft_repo {
        draft.finish(draft_revision)?;
    }
    Ok(model)
}
pub fn select_gguf(files: Vec<String>, variant: Option<&str>) -> Result<Vec<String>> {
    let candidates = files
        .iter()
        .filter(|n| {
            !n.contains('/') && n.ends_with(".gguf") && !n.to_lowercase().contains("mmproj")
        })
        .cloned()
        .collect::<Vec<_>>();
    if let Some(variant) = variant {
        if variant.is_empty()
            || variant.len() > 64
            || !variant
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
            || variant.contains("..")
        {
            return Err("invalid GGUF variant".into());
        }
        let words = candidates
            .iter()
            .map(|n| {
                n.strip_suffix(".gguf")
                    .unwrap()
                    .split('-')
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut shared = 0;
        while words.first().is_some_and(|first| {
            shared < first.len() && words.iter().all(|w| w.get(shared) == first.get(shared))
        }) {
            shared += 1;
        }
        let exact = candidates
            .iter()
            .zip(&words)
            .filter(|(_, w)| w[shared..].join("-").eq_ignore_ascii_case(variant))
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>();
        if exact.len() == 1 {
            return Ok(exact);
        }
        let ending = candidates
            .iter()
            .filter(|n| {
                n.strip_suffix(".gguf")
                    .unwrap()
                    .to_lowercase()
                    .ends_with(&format!("-{}", variant.to_lowercase()))
            })
            .cloned()
            .collect::<Vec<_>>();
        if ending.len() == 1 {
            return Ok(ending);
        }
        let prefix = format!("{}/", variant.to_lowercase());
        let parts = files
            .iter()
            .filter(|n| {
                n.to_lowercase().starts_with(&prefix)
                    && n.matches('/').count() == 1
                    && n.ends_with(".gguf")
                    && !n.to_lowercase().contains("mmproj")
            })
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        if !parts.is_empty() {
            return Ok(crate::source_gguf::ordered(parts)?
                .into_iter()
                .map(|p| p.to_str().unwrap().to_string())
                .collect());
        }
    } else if candidates.len() == 1 {
        return Ok(candidates);
    }
    Err(format!(
        "select one GGUF with OWNER/REPO:VARIANT (root files: {})",
        candidates.join(", ")
    )
    .into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn variants_choose_exact_suffix_and_complete_splits() {
        let files = ["M-UD-Q4_K_M.gguf", "M-Q4_K_M.gguf", "mmproj-BF16.gguf"]
            .map(str::to_string)
            .to_vec();
        assert_eq!(
            select_gguf(files.clone(), Some("Q4_K_M")).unwrap(),
            vec!["M-Q4_K_M.gguf"]
        );
        assert!(select_gguf(files, None).is_err());
        assert_eq!(
            select_gguf(
                vec![
                    "Q4/M-Q4-00002-of-00002.gguf".into(),
                    "Q4/M-Q4-00001-of-00002.gguf".into()
                ],
                Some("q4")
            )
            .unwrap()[0],
            "Q4/M-Q4-00001-of-00002.gguf"
        );
    }
}

fn git_blob_digest(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut input = fs::File::open(path)?;
    let mut hash = ring::digest::Context::new(&ring::digest::SHA1_FOR_LEGACY_USE_ONLY);
    hash.update(format!("blob {}\0", input.metadata()?.len()).as_bytes());
    let mut bytes = [0; 65536];
    loop {
        let n = input.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(hash
        .finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
pub fn verify_assembly(path: &Path) -> Result<usize> {
    let held = crate::model_hold::hold(path)?;
    let record = source_mlx::read_json(&held.path.join("model.json"))?;
    crate::model_hold::verify(&held.path, &record)?;
    let files = record["files"]
        .as_object()
        .ok_or("source model file records missing")?;
    let mut checked = BTreeSet::new();
    for entry in files.values() {
        let path = Path::new(entry["path"].as_str().ok_or("source file path missing")?);
        if !checked.insert(path) {
            continue;
        }
        let expected = entry["digest"].as_str().ok_or("source digest missing")?;
        let actual = if expected.len() == 40 {
            git_blob_digest(path)?
        } else {
            cli::digest(path)?
        };
        if actual != expected {
            return Err(format!("source content hash mismatch: {}", path.display()).into());
        }
    }
    Ok(files.len())
}
#[cfg(test)]
mod digest_tests {
    use super::*;
    #[test]
    fn git_blob_hash_matches_huggingface_git_identity() {
        let stage = Stage::new(&std::env::temp_dir(), "source-git-hash").unwrap();
        let path = stage.path().join("small");
        fs::write(&path, b"hello\n").unwrap();
        assert_eq!(
            git_blob_digest(&path).unwrap(),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }
}
#[cfg(test)]mod transfer_tests{
 use super::*;
 use sha2::{Digest,Sha256};
 fn fixture(name:&str,payload:&[u8],git:bool)->(Stage,Repository){let stage=Stage::new(&std::env::temp_dir().canonicalize().unwrap(),"source-transfer-test").unwrap();let root=stage.path().join("repo");fs::create_dir_all(root.join("blobs")).unwrap();let commit="a".repeat(40);let snapshot=root.join("snapshots").join(&commit);fs::create_dir_all(&snapshot).unwrap();let digest=if git{let mut hash=ring::digest::Context::new(&ring::digest::SHA1_FOR_LEGACY_USE_ONLY);hash.update(format!("blob {}\0",payload.len()).as_bytes());hash.update(payload);hash.finish().as_ref().iter().map(|b|format!("{b:02x}")).collect()}else{format!("{:x}",Sha256::digest(payload))};let repo=Repository{id:"owner/source".into(),commit,root,snapshot,files:BTreeMap::from([(name.into(),Entry{bytes:payload.len()as u64,digest})]),cached:false,verified:RefCell::new(BTreeMap::new())};(stage,repo)}
 #[test]fn git_transfer_validates_commit_path_and_hash_before_publication(){let (_stage,repo)=fixture("config.json",b"{}",true);assert!(repo.file_with("config.json",|_,_,_|Ok(b"XX".to_vec()),|_,_,_,_,_|panic!("not LFS")).is_err());assert!(!repo.snapshot.join("config.json").exists());let file=repo.file_with("config.json",|url,limit,_|{assert!(url.contains(&format!("/resolve/{}/",repo.commit)));assert_eq!(limit,2);Ok(b"{}".to_vec())},|_,_,_,_,_|panic!("not LFS")).unwrap();assert!(file.is_symlink());assert_eq!(fs::read(file).unwrap(),b"{}");}
 #[test]fn lfs_resume_requires_owned_partial_and_publishes_only_after_completion(){let payload=b"complete-model-fixture";let (_stage,repo)=fixture("model.safetensors",payload,false);let result=repo.file_with("model.safetensors",|_,_,_|panic!("not git"),|_,path,_,_,_|{fs::write(path,&payload[..4])?;Err("injected interrupted transfer".into())});assert!(result.is_err());assert!(!repo.snapshot.join("model.safetensors").exists());let file=repo.file_with("model.safetensors",|_,_,_|panic!("not git"),|_,path,bytes,sha,_|{assert_eq!(fs::read(path)?,payload[..4]);assert_eq!(bytes,payload.len()as u64);assert_eq!(sha,format!("{:x}",Sha256::digest(payload)));fs::write(path,payload)?;Ok(bytes)}).unwrap();assert_eq!(fs::read(file).unwrap(),payload);}
 #[test]fn foreign_partial_is_not_modified_or_adopted(){let (_stage,repo)=fixture("model.safetensors",b"weights",false);let parts=repo.root.join(format!(".source-parts-{}",repo.commit));fs::create_dir(&parts).unwrap();let part=parts.join(&repo.files["model.safetensors"].digest);fs::write(&part,b"owner").unwrap();assert!(repo.file_with("model.safetensors",|_,_,_|panic!("network forbidden"),|_,_,_,_,_|panic!("network forbidden")).is_err());assert_eq!(fs::read(part).unwrap(),b"owner");}
 #[test]fn cached_snapshot_and_blob_corruption_are_refused_without_overwrite(){
  for snapshot in [true,false]{let (_stage,repo)=fixture("config.json",b"{}",true);let path=if snapshot{repo.snapshot.join("config.json")}else{repo.root.join("blobs").join(&repo.files["config.json"].digest)};fs::write(&path,b"[]").unwrap();let error=repo.file_with("config.json",|_,_,_|panic!("cached file must not download"),|_,_,_,_,_|panic!("cached file must not download")).unwrap_err();assert!(error.to_string().contains("checksum"));assert_eq!(fs::read(&path).unwrap(),b"[]");}
 }
 #[test]fn previously_verified_cache_detects_a_same_size_replacement(){
  let (_stage,repo)=fixture("config.json",b"{}",true);let path=repo.snapshot.join("config.json");fs::write(&path,b"{}").unwrap();repo.file("config.json").unwrap();let replacement=repo.snapshot.join("replacement");fs::write(&replacement,b"[]").unwrap();fs::rename(replacement,&path).unwrap();assert!(repo.file("config.json").unwrap_err().to_string().contains("checksum"));assert_eq!(fs::read(path).unwrap(),b"[]");
 }

}
