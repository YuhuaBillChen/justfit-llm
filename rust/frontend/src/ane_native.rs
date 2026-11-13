//! Cached-only grouped ANE tower, using the existing Swift C ABI directly.
use crate::{
    bad,
    coreai::{Api, Buffer, Spec},
    vision_math, Result,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    rc::Rc,
    time::Instant,
};

pub fn sha(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).map_err(|_| bad(format!("missing pinned file {}", path.display())))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|_| bad("pinned file read failed"))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn read_json(path: &Path) -> Result<Value> {
    let raw = fs::read(path).map_err(|_| bad("configuration file unavailable"))?;
    serde_json::from_slice(&raw).map_err(|_| bad("invalid configuration JSON"))
}
fn command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|_| bad("native inspection unavailable"))?;
    if !output.status.success() {
        return Err(bad(String::from_utf8_lossy(&output.stderr)));
    }
    String::from_utf8(output.stdout).map_err(|_| bad("inspection returned invalid UTF8"))
}
pub struct Profile {
    pub data: Value,
    pub root: PathBuf,
    pub assets: PathBuf,
    pub positions: PathBuf,
    pub catalog: Value,
    pub library: PathBuf,
    pub capacity: usize,
}
impl Profile {
    pub fn open(path: &Path) -> Result<Self> {
        Self::load(path, true)
    }
    pub fn open_for_preparation(path: &Path) -> Result<Self> {
        Self::load(path, false)
    }
    fn load(path: &Path, cached: bool) -> Result<Self> {
        let data = read_json(path)?;
        let root = path
            .parent()
            .ok_or_else(|| bad("profile has no parent"))?
            .to_path_buf();
        if data["schema"] != 2 || data["kind"] != "grouped-vision" || data["group_count"] != 9 {
            return Err(bad("native worker requires qualified nine-group profile"));
        }
        let capacity = data["capacity_raw_patches"]
            .as_u64()
            .filter(|n| matches!(n, 4096 | 8192))
            .ok_or_else(|| bad("unsupported native ANE capacity"))? as usize;
        let files = data["files"]
            .as_object()
            .ok_or_else(|| bad("profile file pins missing"))?;
        let mut library = None;
        for (name, digest) in files {
            let p = root.join(name);
            if fs::symlink_metadata(&p)
                .map_err(|_| bad("missing pinned asset"))?
                .file_type()
                .is_symlink()
                || sha(&p)? != digest.as_str().unwrap_or("")
            {
                return Err(bad("profile file digest mismatch"));
            }
            if p.file_name().is_some_and(|n| n == "libcoreai_bridge.dylib") {
                library = Some(p)
            }
        }
        let assets = root.join(
            data["assets"]
                .as_str()
                .ok_or_else(|| bad("asset path missing"))?,
        );
        let positions = root.join(
            data["position_table"]
                .as_str()
                .ok_or_else(|| bad("position path missing"))?,
        );
        let catalog = read_json(
            &root.join(
                data["cache_record"]
                    .as_str()
                    .ok_or_else(|| bad("cache record missing"))?,
            ),
        )?;
        if catalog["build"].as_str()
            != Some(command("/usr/bin/sw_vers", &["-buildVersion"])?.trim())
            || catalog["groups"].as_array().map(Vec::len) != Some(9)
        {
            return Err(bad(
                "offline cache preparation required: OS or groups changed",
            ));
        }
        let profile = Self {
            data,
            root,
            assets,
            positions,
            catalog,
            library: library.ok_or_else(|| bad("CoreAI library not pinned"))?,
            capacity,
        };
        // Native serving validates its exact worker-owned catalog separately.
        // Compilation validates graph pins, not a Python client's old caches.
        if cached && profile.data["native_cache_abi"] != 1 {
            for i in 0..9 {
                profile.check_group(i)?
            }
        }
        Ok(profile)
    }
    pub fn check_group(&self, i: usize) -> Result<()> {
        let row = &self.catalog["groups"][i];
        let asset = self.assets.join(format!("group-{i:02}/model.aimodel"));
        let bytes =
            fs::read(asset.join("main.hash")).map_err(|_| bad("asset signature missing"))?;
        let actual = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        if row["asset_hash"] != actual {
            return Err(bad("cached asset identity changed"));
        }
        let files = row["files"]
            .as_array()
            .filter(|f| !f.is_empty())
            .ok_or_else(|| bad("empty cache record"))?;
        for file in files {
            let path = file[0].as_str().ok_or_else(|| bad("invalid cached path"))?;
            let metadata = fs::metadata(path).map_err(|_| bad("cached specialization missing"))?;
            let time = metadata
                .modified()
                .map_err(|_| bad("cache mtime unavailable"))?
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| bad("invalid cache mtime"))?
                .as_nanos();
            if Some(metadata.len()) != file[1].as_u64() || Some(time as u64) != file[2].as_u64() {
                return Err(bad("cached specialization changed"));
            }
        }
        let guard = &row["guard"];
        if sha(Path::new(
            guard["path"]
                .as_str()
                .ok_or_else(|| bad("placement path missing"))?,
        ))? != guard["manifest_sha256"].as_str().unwrap_or("")
        {
            return Err(bad("cached placement metadata changed"));
        }
        Ok(())
    }
}
pub fn placement(asset: &Path) -> Result<Value> {
    let key = fs::read(asset.join("main.hash"))
        .map_err(|_| bad("asset signature missing"))?
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let files = command(
        "/usr/sbin/lsof",
        &["-nP", "-p", &std::process::id().to_string(), "-Fn"],
    )?;
    let mut roots = BTreeSet::new();
    for line in files.lines().filter_map(|s| s.strip_prefix('n')) {
        let p = Path::new(line);
        if !p.components().any(|v| v.as_os_str() == key.as_str()) || !line.contains("coreai-cache")
        {
            continue;
        }
        for ancestor in p.ancestors() {
            if ancestor.file_name().is_some_and(|n| n == "model.aimodelx") {
                roots.insert(ancestor.to_path_buf());
                break;
            }
        }
    }
    if roots.len() != 1 {
        return Err(bad("loaded ANE specialization absent or ambiguous"));
    }
    let root = roots.into_iter().next().unwrap();
    let base = root.join("main-this-delegates/MPSGraph");
    let mut manifests = Vec::new();
    for item in fs::read_dir(base).map_err(|_| bad("unsupported specialization layout"))? {
        let p = item
            .map_err(|_| bad("unreadable specialization"))?
            .path()
            .join("manifest.plist");
        if p.is_file() {
            manifests.push(p)
        }
    }
    if manifests.len() != 1 {
        return Err(bad("placement manifest absent or ambiguous"));
    }
    let manifest = &manifests[0];
    let mut result=placement_manifest(manifest)?;
    result["specialization_root"]=json!(root);
    Ok(result)
}
/// Verify an already pinned compiled specialization before binding its profile.
pub fn placement_manifest(manifest:&Path)->Result<Value> {
    let text = command(
        "/usr/bin/plutil",
        &[
            "-convert",
            "json",
            "-o",
            "-",
            manifest
                .to_str()
                .ok_or_else(|| bad("invalid manifest path"))?,
        ],
    )?;
    let parsed: Value = serde_json::from_str(&text).map_err(|_| bad("invalid placement plist"))?;
    let mut count = 0;
    for version in parsed["Package Version"]
        .as_object()
        .ok_or_else(|| bad("placement versions missing"))?
        .values()
    {
        for module in version["Optimized Modules"]
            .as_object()
            .ok_or_else(|| bad("optimized modules missing"))?
            .values()
        {
            for (name, attrs) in module["Entry Function Attributes"]
                .as_object()
                .ok_or_else(|| bad("entry attributes missing"))?
            {
                if name != "group" && !name.starts_with("group_") {
                    continue;
                }
                let attrs = attrs
                    .as_array()
                    .ok_or_else(|| bad("invalid entry attributes"))?;
                if !attrs.contains(&json!("mps.fullyPlacedOnANE"))
                    || !attrs.contains(&json!("mps.noGPUActivity"))
                    || !attrs.iter().any(|a| {
                        a["mps.ane.calleeName"]
                            .as_str()
                            .is_some_and(|s| !s.is_empty())
                    })
                {
                    return Err(bad("graph is not fully placed on ANE"));
                }
                count += 1;
            }
        }
    }
    if count == 0 {
        return Err(bad("no group placement entry"));
    }
    Ok(
        json!({"full_ane":true,"no_gpu":true,"entries":count,"manifest_sha256":sha(manifest)?,"manifest_path":manifest}),
    )
}
pub fn probe(path: &Path) -> Result<Value> {
    let p = Profile::open(path)?;
    let api = Api::open(&p.library)?;
    let asset = p.assets.join("group-00/model.aimodel");
    let began = Instant::now();
    let f = api.model(&asset)?.function("group")?;
    Ok(
        json!({"load_seconds":began.elapsed().as_secs_f64(),"placement":placement(&asset)?,"inputs":f.descriptor.inputs.iter().map(|s|json!({"name":s.name,"dtype":s.dtype,"shape":s.shape,"strides":s.strides,"nbytes":s.nbytes})).collect::<Vec<_>>(),"outputs":f.descriptor.outputs.iter().map(|s|json!({"name":s.name,"dtype":s.dtype,"shape":s.shape,"strides":s.strides,"nbytes":s.nbytes})).collect::<Vec<_>>()}),
    )
}
fn cache_files(path: &Path, out: &mut Vec<Value>) -> Result<()> {
    for item in fs::read_dir(path).map_err(|_| bad("compiled cache directory unavailable"))? {
        let item = item.map_err(|_| bad("compiled cache entry unavailable"))?;
        let path = item.path();
        let meta =
            fs::symlink_metadata(&path).map_err(|_| bad("compiled cache metadata unavailable"))?;
        if meta.file_type().is_symlink() {
            return Err(bad("compiled cache contains symlinks"));
        }
        if meta.is_dir() {
            cache_files(&path, out)?
        } else if meta.is_file() {
            out.push(json!([
                path,
                meta.len(),
                meta.modified()
                    .map_err(|_| bad("cache mtime unavailable"))?
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| bad("invalid cache mtime"))?
                    .as_nanos() as u64
            ]));
        }
    }
    Ok(())
}
/// Explicit offline operation. A native host needs its own specialization
/// namespace; a Python-namespaced cache can fail native function loading.
pub fn prepare_native_cache(top: &Path, target: &Path) -> Result<Value> {
    if target.exists() {
        return Err(bad("native cache record already exists"));
    }
    let root = top
        .parent()
        .ok_or_else(|| bad("tier profile has no parent"))?;
    let config = read_json(top)?;
    if config["schema"] != 3 {
        return Err(bad("native cache preparation requires tiered profile"));
    }
    let mut tiers = serde_json::Map::new();
    let mut meter = None;
    for tier in config["tiers"]
        .as_array()
        .ok_or_else(|| bad("tier list missing"))?
    {
        let leaf = root.join(
            tier["profile"]
                .as_str()
                .ok_or_else(|| bad("leaf path missing"))?,
        );
        if sha(&leaf)? != tier["sha256"].as_str().unwrap_or("") {
            return Err(bad("leaf profile digest mismatch"));
        }
        let profile = Profile::open_for_preparation(&leaf)?;
        if meter.is_none() {
            meter = Some(crate::worker_meter::Meter::start(
                &profile.library.parent().ok_or_else(|| bad("bridge parent missing"))?
                    .join("attributed_memory.dylib"))?);
        }
        let api = Api::open(&profile.library)?;
        let mut groups = Vec::new();
        for i in 0..9 {
            let asset = profile.assets.join(format!("group-{i:02}/model.aimodel"));
            let started = Instant::now();
            let placed = {
                let _function = api.model(&asset)?.function("group")?;
                placement(&asset)?
            };
            let cache_root = Path::new(
                placed["specialization_root"]
                    .as_str()
                    .ok_or_else(|| bad("native cache root missing"))?,
            );
            let host =
                std::env::var("JUSTFIT_NATIVE_CACHE_ID").unwrap_or_else(|_| "justfit-rs".into());
            if !cache_root
                .components()
                .any(|c| c.as_os_str() == host.as_str())
            {
                return Err(bad(
                    "native cache was prepared under an unexpected host identity",
                ));
            }
            let mut files = Vec::new();
            cache_files(cache_root, &mut files)?;
            groups.push(json!({"asset_hash":profile.catalog["groups"][i]["asset_hash"],"guard":{"path":placed["manifest_path"],"manifest_sha256":placed["manifest_sha256"]},"files":files,"prepare_seconds":started.elapsed().as_secs_f64()}));
            eprintln!(
                "NATIVE_CACHE_PREPARE {}",
                json!({"capacity":profile.capacity,"group":i,"seconds":started.elapsed().as_secs_f64(),"placement":placed})
            );
            fs::write(
                target.with_extension("partial.json"),
                json!({"capacity":profile.capacity,"groups":groups}).to_string(),
            )
            .map_err(|_| bad("cache preparation progress write failed"))?;
        }
        tiers.insert(profile.capacity.to_string(),json!({"profile_sha256":sha(&leaf)?,"library_sha256":sha(&profile.library)?,"groups":groups}));
    }
    let result = json!({"schema":1,"host":std::env::var("JUSTFIT_NATIVE_CACHE_ID").unwrap_or_else(|_|"justfit-rs".into()),"worker_sha256":sha(&std::env::current_exe().map_err(|_|bad("worker executable unavailable"))?)?,"build":command("/usr/bin/sw_vers",&["-buildVersion"])?.trim(),"tiered_profile_sha256":sha(top)?,"tiers":tiers});
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(target)
        .map_err(|_| bad("native cache record unavailable"))?;
    file.write_all(result.to_string().as_bytes())
        .map_err(|_| bad("native cache record write failed"))?;
    Ok(result)
}
pub fn native_cache(profile: &Profile, path: &Path) -> Result<Value> {
    let record = read_json(Path::new(
        &std::env::var("JUSTFIT_NATIVE_VISION_CATALOG")
            .map_err(|_| bad("offline native cache preparation is required"))?,
    ))?;
    if record["schema"] != 1
        || record["host"]
            != std::env::var("JUSTFIT_NATIVE_CACHE_ID").unwrap_or_else(|_| "justfit-rs".into())
        || record["worker_sha256"]
            != sha(&std::env::current_exe().map_err(|_| bad("worker executable unavailable"))?)?
        || record["build"].as_str() != Some(command("/usr/bin/sw_vers", &["-buildVersion"])?.trim())
    {
        return Err(bad("native cache identity/OS mismatch"));
    }
    let row = record["tiers"][profile.capacity.to_string()].clone();
    if row["profile_sha256"] != sha(path)?
        || row["library_sha256"] != sha(&profile.library)?
        || row["groups"].as_array().map(Vec::len) != Some(9)
    {
        return Err(bad("native cache source/profile mismatch"));
    }
    Ok(row)
}
pub fn check_native_group(row: &Value) -> Result<()> {
    for file in row["files"]
        .as_array()
        .filter(|f| !f.is_empty())
        .ok_or_else(|| bad("native cache file list missing"))?
    {
        let meta = fs::metadata(
            file[0]
                .as_str()
                .ok_or_else(|| bad("native cache file path invalid"))?,
        )
        .map_err(|_| bad("native cache missing; offline preparation required"))?;
        if Some(meta.len()) != file[1].as_u64()
            || Some(
                meta.modified()
                    .map_err(|_| bad("cache mtime unavailable"))?
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| bad("invalid cache mtime"))?
                    .as_nanos() as u64,
            ) != file[2].as_u64()
        {
            return Err(bad("native cache changed; offline preparation required"));
        }
    }
    let guard = &row["guard"];
    if sha(Path::new(
        guard["path"]
            .as_str()
            .ok_or_else(|| bad("native manifest path invalid"))?,
    ))? != guard["manifest_sha256"].as_str().unwrap_or("")
    {
        return Err(bad("native placement metadata changed"));
    }
    Ok(())
}

fn side_input(buffer: &Buffer, name: &str, table: &[f32], h: usize, w: usize) -> Result<()> {
    let count = h * w;
    match name {
        "position" => {
            if buffer.spec.shape.len() != 4
                || buffer.spec.shape != [1, 1152, 1, buffer.spec.shape[3]]
                || buffer.spec.shape[3] < count
            {
                return Err(bad("invalid position signature"));
            }
            buffer.fill(0.)?;
            let values = vision_math::positions(table, h, w)?;
            for token in 0..count {
                for c in 0..1152 {
                    buffer.set(&[0, c, 0, token], values[token * 1152 + c])?
                }
            }
        }
        "cosine" | "sine" => {
            if buffer.spec.shape.len() != 4
                || buffer.spec.shape[0] != 1
                || buffer.spec.shape[1] != 72
                || buffer.spec.shape[2] != 1
                || buffer.spec.shape[3] < count
            {
                return Err(bad("invalid RoPE signature"));
            }
            buffer.fill(if name == "cosine" { 1. } else { 0. })?;
            let (cos, sin) = vision_math::rope(h, w)?;
            let values = if name == "cosine" { cos } else { sin };
            for token in 0..count {
                for c in 0..72 {
                    buffer.set(&[0, c, 0, token], values[token * 72 + c])?
                }
            }
        }
        "key_mask" => {
            if buffer.spec.shape.len() != 4
                || buffer.spec.shape[0] != 1
                || buffer.spec.shape[1] < count
                || buffer.spec.shape[2..] != [1, 1]
            {
                return Err(bad("invalid key mask signature"));
            }
            buffer.fill(-10000.)?;
            for token in 0..count {
                buffer.set(&[0, token, 0, 0], 0.)?
            }
        }
        _ => return Err(bad("unknown graph side input")),
    }
    Ok(())
}
pub fn encode(
    profile: &Path,
    source: &Path,
    target: &Path,
    grids: &[(usize, usize)],
) -> Result<Value> {
    let p = Profile::open(profile)?;
    let cache = native_cache(&p, profile)?;
    let api = Api::open(&p.library)?;
    let table = vision_math::npy_positions(&p.positions)?;
    if grids.is_empty() || grids.len() > 4 {
        return Err(bad("invalid ANE image count"));
    }
    let size = grids.iter().try_fold(0usize, |n, (h, w)| {
        vision_math::grid(*h, *w)?;
        if h * w > p.capacity {
            return Err(bad("grid exceeds compiled capacity"));
        }
        Ok(n + h * w * 768)
    })?;
    if fs::metadata(source)
        .map_err(|_| bad("RGB source unavailable"))?
        .len()
        != size as u64
    {
        return Err(bad("RGB payload length mismatch"));
    }
    let raw = fs::read(source).map_err(|_| bad("RGB source unavailable"))?;
    let mut dst = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)
        .map_err(|_| bad("embedding output already exists or unavailable"))?;
    let mut offset = 0;
    let mut metrics = Vec::new();
    for &(h, w) in grids {
        let started = Instant::now();
        let rgb = &raw[offset..offset + h * w * 768];
        offset += h * w * 768;
        let mut side: HashMap<String, Rc<Buffer>> = HashMap::new();
        let mut previous: Option<Rc<Buffer>> = None;
        let mut buffers: Vec<(usize, Spec, Rc<Buffer>)> = Vec::new();
        let mut rows = Vec::new();
        for i in 0..9 {
            if p.data["native_cache_abi"] != 1 {p.check_group(i)?;}
            check_native_group(&cache["groups"][i])?;
            let asset = p.assets.join(format!("group-{i:02}/model.aimodel"));
            let tick = Instant::now();
            let function = api.model(&asset)?.function("group")?;
            let placed = placement(&asset)?;
            if placed["manifest_sha256"] != cache["groups"][i]["guard"]["manifest_sha256"]
                || placed["manifest_path"] != cache["groups"][i]["guard"]["path"]
            {
                return Err(bad(
                    "loaded native specialization differs from frozen cache",
                ));
            }
            let load = tick.elapsed().as_secs_f64();
            let mut inputs = Vec::new();
            for spec in &function.descriptor.inputs {
                let buffer = if spec.name == "x" {
                    if let Some(ref prev) = previous {
                        let mut expected = prev.spec.clone();
                        expected.name = spec.name.clone();
                        if expected != *spec {
                            return Err(bad("group handoff layout mismatch"));
                        }
                        prev.clone()
                    } else {
                        let b = Buffer::new(&api, spec.clone())?;
                        if b.spec.shape != [p.capacity, 1536] {
                            return Err(bad("patch input signature mismatch"));
                        }
                        b.fill(0.)?;
                        let patch = vision_math::patchify(rgb, h, w)?;
                        for token in 0..h * w {
                            for c in 0..1536 {
                                b.set(&[token, c], patch[token * 1536 + c])?
                            }
                        }
                        b
                    }
                } else if let Some(b) = side.get(&spec.name) {
                    if b.spec != *spec {
                        return Err(bad("side input layout mismatch"));
                    }
                    b.clone()
                } else {
                    let b = Buffer::new(&api, spec.clone())?;
                    side_input(&b, &spec.name, &table, h, w)?;
                    side.insert(spec.name.clone(), b.clone());
                    b
                };
                inputs.push((spec.name.clone(), buffer));
            }
            let spec = function
                .descriptor
                .outputs
                .iter()
                .find(|s| s.name == "output")
                .ok_or_else(|| bad("graph output missing"))?;
            let output = if let Some((_, _, buffer)) = buffers
                .iter()
                .find(|(parity, s, _)| *parity == i % 2 && s == spec)
            {
                buffer.clone()
            } else {
                let b = Buffer::new(&api, spec.clone())?;
                buffers.push((i % 2, spec.clone(), b.clone()));
                b
            };
            let seconds = function.bind_run(&inputs, output.clone())?;
            rows.push(
                json!({"group":i,"load_seconds":load,"encode_seconds":seconds,"placement":placed}),
            );
            previous = Some(output);
        }
        let output = previous.ok_or_else(|| bad("tower produced no output"))?;
        if output.spec.shape != [p.capacity / 4, 5120] {
            return Err(bad("embedding signature mismatch"));
        }
        for start in (0..h * w / 4).step_by(32) {
            let mut values = Vec::new();
            for row in start..(start + 32).min(h * w / 4) {
                for c in 0..5120 {
                    values.push(output.get(&[row, c])?)
                }
            }
            dst.write_all(&vision_math::bf16(&values)?)
                .map_err(|_| bad("embedding write failed"))?
        }
        metrics.push(
            json!({"grid":[h,w],"wall_seconds":started.elapsed().as_secs_f64(),"groups":rows}),
        );
    }
    Ok(
        json!({"images":metrics,"pid":std::process::id(),"runtime_python":false,"scope":"native cached-only grouped tower, existing C ABI and assets"}),
    )
}
