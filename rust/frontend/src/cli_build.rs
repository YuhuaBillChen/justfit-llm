//! Native runtime build orchestration. Compiler/build tools run directly;
//! the make configuration and identity helpers execute in this Rust binary.
use crate::cli::{digest,verify};
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},process::Command,sync::atomic::{AtomicU64,Ordering}};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
static NEXT:AtomicU64=AtomicU64::new(0);
pub const OUTPUTS:[&str;11]=["source/build/justfit-engine","source/build/justfit.metallib","source/build/justfit-verify16.metallib","source/build/justfit-rs","source/build/JustFitVision.app/Contents/MacOS/justfit-ane-worker","source/build/JustFitVision.app/Contents/Info.plist","source/build/JustFitVision.app/Contents/_CodeSignature/CodeResources","vision-tools/attributed_memory.dylib","vision-tools/libcoreai_bridge.dylib","source/build/libpdfium.dylib","source/build/PDFIUM-LICENSES.txt"];
const WORKER_SOURCES:[&str;8]=["rust/ane-worker/Cargo.toml","rust/ane-worker/Cargo.lock","rust/ane-worker/src/main.rs","rust/frontend/src/coreai.rs","rust/frontend/src/ane_native.rs","rust/frontend/src/vision_math.rs","rust/frontend/src/worker_meter.rs","rust/frontend/data/rope-m4-reference-f32.bin"];
const CACHE_ID:&str="org.justfit.vision.aaa0bdc9ed86239e";
pub(crate) fn is_built(root:&Path)->bool {
    (||->Result<bool>{
        let receipt:Value=serde_json::from_slice(&fs::read(root.join("build-receipt.json"))?)?;
        if receipt["source_manifest_sha256"]!=digest(&root.join("package-manifest.json"))?{return Ok(false);}
        let outputs=receipt["outputs"].as_object().ok_or("runtime output list missing")?;
        if outputs.len()!=OUTPUTS.len(){return Ok(false);}
        for name in OUTPUTS{let path=root.join(name);let meta=path.symlink_metadata()?;if !meta.is_file()||meta.file_type().is_symlink()||outputs.get(name).and_then(Value::as_str)!=Some(digest(&path)?.as_str()){return Ok(false);}}
        Ok(true)
    })().unwrap_or(false)
}
fn shell_word(text:&str)->String{format!("'{}'",text.replace('\'',"'\"'\"'"))}
fn copy_atomic(source:&Path,target:&Path)->Result<()> {
    let parent=target.parent().ok_or("runtime output has no parent")?;fs::create_dir_all(parent)?;
    let temporary=parent.join(format!(".justfit-build-output-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
    let mut output=fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
    let result=(||{let mut input=fs::File::open(source)?;std::io::copy(&mut input,&mut output)?;output.sync_all()?;drop(output);fs::set_permissions(&temporary,fs::metadata(source)?.permissions())?;fs::rename(&temporary,target)?;Ok(())})();
    if temporary.exists(){fs::remove_file(temporary)?;}result
}
fn run(root:&Path,args:Vec<String>,commands:&mut Vec<Vec<String>>,jobs:usize)->Result<()> {
    commands.push(args.clone());
    let status=Command::new(&args[0]).args(&args[1..]).current_dir(root).env("CARGO_BUILD_JOBS",jobs.to_string()).env("CMAKE_BUILD_PARALLEL_LEVEL",jobs.to_string()).status()?;
    if !status.success(){return Err(format!("build command failed ({status}): {}",args[0]).into());}Ok(())
}
pub fn run_cli(args:&[String])->Result<()> {
    let mut flags=BTreeMap::new();let mut index=0;
    while index<args.len() {
        let flag=args[index].as_str();
        if !["--project","--metal-bin","--jobs","--pdfium-library"].contains(&flag){return Err(format!("unknown build option: {flag}").into());}
        let value=args.get(index+1).ok_or_else(||format!("{flag} needs a value"))?;
        if flags.insert(flag,value.as_str()).is_some(){return Err(format!("duplicate build option: {flag}").into());}index+=2;
    }
    crate::cli_platform::require_mac()?;
    let root=match flags.get("--project"){Some(path)=>PathBuf::from(path).canonicalize()?,None=>crate::cli::default_project()?.canonicalize()?};verify(&root)?;
    let jobs:usize=flags.get("--jobs").copied().unwrap_or("4").parse()?;if jobs==0{return Err("--jobs must be positive".into());}
    let metal=if let Some(path)=flags.get("--metal-bin"){PathBuf::from(path).canonicalize().map_err(|_|format!("Metal toolchain directory unavailable: {path}"))?}else{
        crate::cli_platform::find_metal_bin().ok_or("Metal toolchain not found; install it with `xcodebuild -downloadComponent MetalToolchain` or pass --metal-bin")?
    };
    for name in ["metal","metallib"]{if !metal.join(name).is_file(){return Err(format!("missing Metal tool: {name}").into());}}
    let libraries:Value=serde_json::from_slice(&fs::read(root.join("config/native-libraries.json"))?)?;
    if libraries["schema"]!=1 || libraries["architecture"]!="arm64"{return Err("invalid native library manifest".into());}
    let mut inputs=vec![];
    let name="source/build/libpdfium.dylib";
    let (path,_pdfium_stage)=crate::native_libraries::pdfium(&root,flags.get("--pdfium-library").copied())?;
    let record=&libraries["libraries"][name];
    if record["sha256"].as_str()!=Some(digest(&path)?.as_str()){return Err("PDFium upstream SHA256 mismatch".into());}
    inputs.push((name,path.canonicalize()?));
    let license=root.join(record["license_file"].as_str().ok_or("missing PDFium license path")?);
    if record["license_sha256"].as_str()!=Some(digest(&license)?.as_str()){return Err("PDFium license SHA256 mismatch".into());}
    inputs.push(("source/build/PDFIUM-LICENSES.txt",license));
    let src=root.join("source");let tools=root.join("vision-tools");let exe=std::env::current_exe()?;
    let mut commands=vec![];
    run(&src,vec!["/usr/bin/make".into(),format!("-j{jobs}"),format!("METAL={}",shell_word(metal.join("metal").to_str().ok_or("non-UTF8 Metal tool path")?)),format!("METALLIB={}",shell_word(metal.join("metallib").to_str().ok_or("non-UTF8 Metal tool path")?)),format!("JUSTFIT_BUILD_HELPER={} build-helper",shell_word(exe.to_str().ok_or("non-UTF8 build executable")?)),"all".into()],&mut commands,jobs)?;
    let wide=src.join("build/verify16");fs::create_dir_all(&wide)?;let mut airs=vec![];
    for name in ["gdn","draft","attention_qkv","paged_attention_store","tq4_store","tq4_attention","tq4_rotated","sampling","tree"] {
        let source=src.join(format!("runtime/metal/kernels/decode/{name}.metal"));let out=wide.join(format!("decode-{name}.air"));
        let mut command=vec![metal.join("metal").display().to_string()];command.extend(["-std=metal4.0","-O3","-Wall","-Wextra","-Werror","-Iruntime","-mmacosx-version-min=26.4","-DJUSTFIT_TARGET_VERIFY_ROWS=16u","-c"].map(str::to_string));command.extend([source.display().to_string(),"-o".into(),out.display().to_string()]);run(&src,command,&mut commands,jobs)?;airs.push(out);
    }
    let mut link=vec![metal.join("metallib").display().to_string()];link.extend(airs.iter().map(|p|p.display().to_string()));link.extend(["-o".into(),src.join("build/justfit-verify16.metallib").display().to_string()]);run(&src,link,&mut commands,jobs)?;
    run(&src,vec!["/usr/bin/xcrun".into(),"clang".into(),"-O2".into(),"-dynamiclib".into(),tools.join("attributed_memory.c").display().to_string(),"-o".into(),tools.join("attributed_memory.dylib").display().to_string()],&mut commands,jobs)?;
    run(&src,vec!["/usr/bin/xcrun".into(),"swiftc".into(),"-O".into(),"-parse-as-library".into(),"-emit-library".into(),tools.join("CoreAIBridge.swift").display().to_string(),"-o".into(),tools.join("libcoreai_bridge.dylib").display().to_string()],&mut commands,jobs)?;
    for name in ["frontend","ane-worker"] {
        let crate_=root.join("rust").join(name);fs::File::open(crate_.join("src/main.rs"))?.set_modified(std::time::SystemTime::now())?;
        // Explicit target directory prevents environment leakage between crates.
        let command=vec!["cargo".into(),"build".into(),"--release".into(),"--locked".into(),"-j".into(),jobs.to_string()];commands.push(command.clone());
        let status=Command::new(&command[0]).args(&command[1..]).current_dir(&crate_).env("CARGO_TARGET_DIR",crate_.join("target")).env("CARGO_BUILD_JOBS",jobs.to_string()).env("CMAKE_BUILD_PARALLEL_LEVEL",jobs.to_string()).status()?;
        if !status.success(){return Err(format!("Rust {name} build failed: {status}").into());}
    }
    let build=src.join("build");copy_atomic(&root.join("rust/frontend/target/release/justfit-rs"),&build.join("justfit-rs"))?;
    let app=build.join("JustFitVision.app");let contents=app.join("Contents");let worker=contents.join("MacOS/justfit-ane-worker");copy_atomic(&root.join("rust/ane-worker/target/release/justfit-ane-worker"),&worker)?;
    let plist=format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n\t<key>CFBundleExecutable</key>\n\t<string>justfit-ane-worker</string>\n\t<key>CFBundleIdentifier</key>\n\t<string>{CACHE_ID}</string>\n\t<key>CFBundleName</key>\n\t<string>JustFit Native Vision</string>\n\t<key>CFBundlePackageType</key>\n\t<string>APPL</string>\n\t<key>CFBundleVersion</key>\n\t<string>2.5</string>\n</dict>\n</plist>\n");fs::write(contents.join("Info.plist"),plist)?;
    run(&src,vec!["/usr/bin/codesign".into(),"--force".into(),"--sign".into(),"-".into(),app.display().to_string()],&mut commands,jobs)?;
    for (name,path) in &inputs {copy_atomic(path,&root.join(name))?;}
    verify(&root)?;
    let mut outputs=BTreeMap::new();for name in OUTPUTS{outputs.insert(name,digest(&root.join(name))?);}
    let mut source_pins=BTreeMap::new();for name in WORKER_SOURCES{source_pins.insert(name,digest(&root.join(name))?);}
    let toolchain=BTreeMap::from([(metal.join("metal").display().to_string(),digest(&metal.join("metal"))?),(metal.join("metallib").display().to_string(),digest(&metal.join("metallib"))?)]);
    let receipt=json!({"commands":commands,"outputs":outputs,"toolchain":toolchain,"source_manifest_sha256":digest(&root.join("package-manifest.json"))?,"native_vision":{"identifier":CACHE_ID,"worker_sha256":digest(&worker)?,"source_pins":source_pins},"native_libraries":libraries,"status":"build_complete_runtime_unqualified","builder":"rust"});
    let temporary=root.join(format!(".build-receipt-{}.json",std::process::id()));
    use std::io::Write;let mut output=fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;output.write_all((serde_json::to_string_pretty(&receipt)?+"\n").as_bytes())?;output.sync_all()?;drop(output);fs::rename(temporary,root.join("build-receipt.json"))?;
    println!("{}",serde_json::json!({"built":true,"outputs":OUTPUTS.len(),"builder":"rust"}));Ok(())
}
