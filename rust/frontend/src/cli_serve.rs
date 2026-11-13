//! Build the public local-serving configuration directly from the node preset.
use crate::cli::{digest, verify};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::{Path,PathBuf}};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;

fn absolute(path: &str) -> Result<PathBuf> {
    let path=if path=="~" {PathBuf::from(std::env::var_os("HOME").ok_or("HOME is missing")?)} else if let Some(tail)=path.strip_prefix("~/") {PathBuf::from(std::env::var_os("HOME").ok_or("HOME is missing")?).join(tail)} else {PathBuf::from(path)};
    Ok(if path.exists() {path.canonicalize()?}else{std::path::absolute(path)?})
}
fn size(text: &str) -> Result<u64> {
    let end=text.find(|ch:char|!ch.is_ascii_digit()).unwrap_or(text.len());
    let number:u64=text[..end].parse()?;
    let unit=match &text[end..] {""|"B"=>1,"K"|"KiB"=>1024,"M"|"MiB"=>1024*1024,"G"|"GiB"=>1024*1024*1024,_=>return Err(format!("invalid size: {text}").into())};
    number.checked_mul(unit).ok_or_else(||format!("size overflow: {text}").into())
}
fn preset(root: &Path,node: &str,model: &Path,port:u16,state: &Path,frontend: &Path,profile: Option<&Path>) -> Result<Value> {
    let preset:Value=serde_json::from_slice(&std::fs::read(root.join(format!("config/nodes/{node}.json")))?)?;
    if preset["schema"]!=2 || preset["node"]!=node {return Err("invalid node preset schema or identity".into());}
    let command=preset["server_arguments"].as_array().ok_or("invalid node server arguments")?;
    let replacements=[("{release}",root.display().to_string()),("{model}",model.display().to_string()),("{port}",port.to_string()),("{cache}",state.join("cache").display().to_string())];
    let mut flags=BTreeMap::new();let mut index=0;
    while index<command.len() {
        let flag=command[index].as_str().ok_or("invalid node command argument")?;
        if ["--disable-ane","--persistent-cache","--no-webui"].contains(&flag) {flags.insert(flag.to_string(),String::new());index+=1;continue;}
        if !["--tokenizer","--model","--binary","--host","--port","--max-context","--max-memory","--host-reserve","--kv-format","--max-cache-disk","--cache-dir","--request-timeout"].contains(&flag) {return Err(format!("unsupported node option: {flag}").into());}
        let mut value=command.get(index+1).and_then(Value::as_str).ok_or("node option needs a value")?.to_string();
        for (key,text) in &replacements {value=value.replace(key,text);}
        if flags.insert(flag.to_string(),value).is_some() {return Err(format!("duplicate node option: {flag}").into());}
        index+=2;
    }
    let get=|key:&str|flags.get(key).map(String::as_str);
    let required=|key:&str|->Result<String>{get(key).map(str::to_string).ok_or_else(||format!("node option missing: {key}").into())};
    let reserve=size(get("--host-reserve").unwrap_or("0"))?;
    let mut native=vec![required("--binary")?,"serve-native".into(),model.display().to_string(),get("--max-context").unwrap_or("auto").into(),match get("--max-memory"){Some(v)=>size(v)?.to_string(),None=>"auto".into()}];
    if let Some(value)=get("--max-cache-disk") {native.push(size(value)?.to_string());}
    if flags.contains_key("--persistent-cache") {native.extend(["--cache-dir".into(),required("--cache-dir")?]);}
    if get("--host-reserve").is_some() {native.extend(["--host-reserve".into(),reserve.to_string()]);}
    if let Some(value)=get("--kv-format").filter(|v|*v!="int8") {native.extend(["--kv-format".into(),value.into()]);}
    if flags.contains_key("--disable-ane") {native.extend(["--ane".into(),"off".into()]);}
    let mut vision=json!({"command":[frontend,"media-driver"],"model_root":model,"profile":profile,"host_reserve":reserve.max(1)});
    if let Some(profile)=profile {
        if reserve==0 {return Err("ANE vision requires an explicit positive host reserve".into());}
        let worker=root.join("source/build/JustFitVision.app/Contents/MacOS/justfit-ane-worker");
        if !worker.is_file() {return Err("native ANE worker missing; install the matching native vision runtime".into());}
        let build=std::process::Command::new("/usr/bin/sw_vers").arg("-buildVersion").output()?;
        if !build.status.success() {return Err("cannot identify macOS build".into());}
        let hash=format!("{:x}",Sha256::digest(format!("{}{}{}",digest(profile)?,digest(&worker)?,String::from_utf8(build.stdout)?).as_bytes()));
        let home=PathBuf::from(std::env::var_os("HOME").ok_or("HOME is missing")?).join("Library/Application Support/JustFit");
        vision["native_worker"]=json!(worker);
        vision["native_catalog"]=json!(home.join("native-vision").join(hash).join("native-catalog.json"));
        vision["native_cache_home"]=json!(home.join("native-home"));
        vision["native_prepare"]=json!(true);
    }
    let mut config=json!({"tokenizer":required("--tokenizer")?,"model":required("--model")?,"native":native,"host":get("--host").unwrap_or("127.0.0.1"),"port":port,"allowed_hosts":[],"allowed_origins":[],"max_request_bytes":128*1024*1024,"queue_size":32,"webui":!flags.contains_key("--no-webui"),"served_model_names":preset.get("served_model_names").cloned().unwrap_or(json!([])),"announce_served_name":false,"default_reasoning_effort":null,"max_image_pixels":crate::media::MAX_PIXELS,"request_timeout":get("--request-timeout").map(str::parse::<f64>).transpose()?,"vision":vision});
    let pdfium=root.join("source/build/libpdfium.dylib");
    if pdfium.is_file() {config["documents"]=json!({"library":pdfium,"zlib":"/usr/lib/libz.dylib"});}
    let mut environment=preset["environment"].as_object().ok_or("invalid node environment")?.clone();
    if environment.values().any(|value|!value.is_string()) {return Err("node environment values must be strings".into());}
    environment.insert("DASH_APC_DIR".into(),json!(state.join("cache")));
    environment.insert("ANE_METER_DIR".into(),json!(state.join("ane-meter")));
    environment.insert("JUSTFIT_VERIFY16_LIBRARY".into(),json!(root.join("source/build/justfit-verify16.metallib")));
    if let Some(profile)=profile {environment.insert("JUSTFIT_ANE_VISION_PROFILE".into(),json!(profile));}
    Ok(json!({"config":config,"environment":environment,"cwd":root.join("source")}))
}
pub fn run(args: &[String]) -> Result<()> {
    let original_args=args;
    let (filtered,options)=crate::serve_options::Options::parse(args)?;
    if options.key_on_command_line&&!args.iter().any(|v|v=="--dry-run"){
        use std::os::unix::process::CommandExt;
        let mut command=std::process::Command::new(std::env::current_exe()?);command.arg("serve").args(options.scrub_key_args(args));
        if let Some(key)=options.key(){command.env("JUSTFIT_API_KEY",key);}
        return Err(command.exec().into());
    }
    let args=filtered.as_slice();
    let mut flags=BTreeMap::new();let mut ane_bundles=vec![];let mut ane_pins=vec![];let mut dry=false;let mut no_ane=false;let mut language_only=false;let mut offline=false;let mut index=0;
    while index<args.len() {
        let flag=args[index].as_str();
        if flag=="--dry-run" {dry=true;index+=1;continue;}
        if flag=="--language-only" {language_only=true;index+=1;continue;}
        if flag=="--offline" {offline=true;index+=1;continue;}
        if flag=="--no-ane-vision" {no_ane=true;index+=1;continue;}
        if !["--project","--node","--model","--revision","--draft-model","--mmproj","--jobs","--metal-bin","--port","--state-dir","--vision-profile","--runtime-bundle","--runtime-manifest","--runtime-sha256","--ane-bundle","--ane-sha256","--ane-manifest","--ane-manifest-sha256"].contains(&flag) {return Err(format!("unknown serve option: {flag}").into());}
        let value=args.get(index+1).ok_or_else(||format!("{flag} needs a value"))?;
        if flag=="--ane-bundle"{ane_bundles.push(value.clone());}else if flag=="--ane-sha256"{ane_pins.push(value.clone());}else if flags.insert(flag,value.as_str()).is_some() {return Err(format!("duplicate serve option: {flag}").into());}
        index+=2;
    }
    let root=match flags.get("--project") {Some(path)=>absolute(path)?,None=>crate::cli::default_project()?};
    let node=match flags.get("--node"){Some(node)=>*node,None=>crate::cli::detect_preset()?};
    if !["m4","m5","m6"].contains(&node) {return Err("node must be m4, m5 or m6".into());}
    let model_spec=flags.get("--model").copied().unwrap_or(crate::model_install::DEFAULT_MODEL);
    let revision=flags.get("--revision").copied();
    if revision.is_some_and(|value|!crate::model_install::safe_path(value)){return Err("invalid model revision".into());}
    let jobs:usize=flags.get("--jobs").copied().unwrap_or("4").parse().map_err(|_|"--jobs must be positive")?;
    if jobs==0{return Err("--jobs must be positive".into());}
    let environment_port=std::env::var("JUSTFIT_PORT").unwrap_or_else(|_|"8000".into());
    let port:u16=flags.get("--port").copied().unwrap_or(&environment_port).parse()?;
    let state=match flags.get("--state-dir") {Some(path)=>absolute(path)?,None=>root.join("local-state").join(node)};
    let ane_manifest=flags.get("--ane-manifest").copied();let ane_manifest_pin=flags.get("--ane-manifest-sha256").copied();
    crate::ane_bundle::validate_options(&ane_bundles,&ane_pins,ane_manifest,ane_manifest_pin)?;
    let ane_requested=!ane_bundles.is_empty()||ane_manifest.is_some();
    if language_only&&(flags.contains_key("--vision-profile")||ane_requested){return Err("--language-only cannot load vision assets or a vision profile".into());}
    if language_only{no_ane=true;}
    if no_ane && (flags.contains_key("--vision-profile")||ane_requested) {return Err("--no-ane-vision cannot be combined with ANE vision assets or a profile".into());}
    if flags.contains_key("--vision-profile")&&ane_requested{return Err("choose an existing vision profile or precompiled ANE assets".into());}
    let bundle=flags.get("--runtime-bundle");let manifest=flags.get("--runtime-manifest");let pin=flags.get("--runtime-sha256");
    let provisioning=bundle.is_some()||manifest.is_some()||pin.is_some();
    if provisioning && (bundle.is_some()==manifest.is_some() || pin.is_none()) {return Err("choose one --runtime-bundle or --runtime-manifest with --runtime-sha256".into());}
    if let Some(pin)=pin {crate::download::digest(pin)?;}
    if !root.exists() {
        if dry {return Err(format!("workspace {} does not exist; run without --dry-run to create it",root.display()).into());}
        crate::cli::initialize(&crate::cli::payload_root()?,&root)?;
    }
    let root=root.canonicalize()?;verify(&root)?;
    let automatic=root.join("local-state/ane").join(node).join("qualified/tiered.json");
    let mut profile=if no_ane {None}else if let Some(path)=flags.get("--vision-profile") {Some(absolute(path)?)}else{automatic.is_file().then_some(automatic)};
    let frontend=std::env::current_exe()?;
    if dry {let model=absolute(model_spec)?;let mut plan=preset(&root,node,&model,port,&state,&frontend,profile.as_deref())?;if language_only{plan["config"]["vision"]=Value::Null;}options.apply(&mut plan,true)?;println!("{}",serde_json::to_string_pretty(&plan)?);return Ok(());}
    crate::cli_platform::require_mac()?;
    if let Some(path)=bundle {crate::runtime_bundle::install(&root,&absolute(path)?,pin.unwrap())?;}
    if let Some(url)=manifest {crate::runtime_bundle::fetch_and_install(&root,url,pin.unwrap())?;}

    if provisioning && !crate::cli_build::is_built(&root){return Err("prebuilt runtime receipt verification failed".into());}
    if !provisioning && !crate::cli_build::is_built(&root){
        if let Ok(payload)=crate::cli::payload_root(){crate::runtime_bundle::install_bundled(&root,&payload)?;}
    }
    if !crate::cli_build::is_built(&root){
        let mut options=vec!["--project".into(),root.to_str().ok_or("non-UTF8 build workspace")?.into(),"--jobs".into(),jobs.to_string()];
        if let Some(path)=flags.get("--metal-bin"){options.extend(["--metal-bin".into(),(*path).into()]);}
        crate::cli_build::run_cli(&options)?;
    }
    let receipt:Value=serde_json::from_slice(&std::fs::read(root.join("build-receipt.json"))?)?;
    if receipt["source_manifest_sha256"]!=digest(&root.join("package-manifest.json"))? {return Err("runtime receipt does not match source manifest".into());}
    let outputs=receipt["outputs"].as_object().ok_or("runtime receipt has no outputs")?;
    for name in crate::cli_build::OUTPUTS {
        let path=root.join(name);
        if path.symlink_metadata()?.file_type().is_symlink() || outputs.get(name).and_then(Value::as_str)!=Some(digest(&path)?.as_str()) {return Err(format!("runtime output digest mismatch: {name}").into());}
    }
    let installed=root.join("source/build/justfit-rs");
    if digest(&installed)?!=digest(&frontend)? {
        // Replace the bootstrap process after validating the complete runtime.
        // Provisioning flags are consumed once; the installed frontend validates
        // its own receipt again before opening any HTTP listener.
        use std::os::unix::process::CommandExt;
        let mut command=std::process::Command::new(&installed);command.arg("serve");
        let mut index=0;
        while index<original_args.len() {
            if ["--runtime-bundle","--runtime-manifest","--runtime-sha256"].contains(&original_args[index].as_str()) {index+=2;}else{command.arg(&original_args[index]);index+=1;}
        }
        return Err(command.exec().into());
    }
    if offline{std::env::set_var("HF_HUB_OFFLINE","1");}
    let held=crate::model_install::resolve_for_serve_options(&root,model_spec,revision,flags.get("--draft-model").copied(),flags.get("--mmproj").copied(),language_only)?;
    let model=held.path.clone();
    let _model_hold=held;
    let media_enabled=!language_only&&if model.join("model.json").is_file(){crate::source_mlx::read_json(&model.join("model.json"))?["vision_format"]!="none"}else{true};
    if !media_enabled{if ane_requested||flags.contains_key("--vision-profile"){return Err("text-only model cannot load ANE vision assets or profile".into());}profile=None;}

    for name in ["target","draft","tokenizer"] {if !model.join(name).is_dir() {return Err(format!("model directory must contain {name}/").into());}}
    let native_vision=media_enabled&&(model.join("vision/model.bin").is_file()||model.join("model.json").is_file());
    if !native_vision {
        if ane_requested||flags.contains_key("--vision-profile"){return Err("this source model does not support packed ANE vision assets".into());}
        profile=None;
    }
    let delivery=if native_vision&&!no_ane&&!ane_requested&&profile.is_none() {
        crate::ane_distribution::discover(&root,node,&model)?
    }else{None};
    if delivery.is_some(){profile=None;}
    if native_vision&&(ane_requested||(!no_ane&&profile.is_none()&&delivery.is_none())){
        if let Some(prepared)=crate::ane_bundle::install_options(&root,node,&model,&ane_bundles,&ane_pins,ane_manifest,ane_manifest_pin)?{profile=Some(prepared);}
    }
    let mut plan=preset(&root,node,&model,port,&state,&frontend,profile.as_deref())?;
    if let Some(delivery)=delivery{
        let worker=root.join("source/build/JustFitVision.app/Contents/MacOS/justfit-ane-worker");
        if !worker.is_file(){return Err("native ANE worker missing from runtime".into());}
        let profile=crate::ane_distribution::profile(&delivery)?;
        let vision=&mut plan["config"]["vision"];
        vision["profile"]=json!(profile);
        vision["native_catalog"]=json!(crate::ane_distribution::catalog(&delivery,&worker)?);
        vision["native_worker"]=json!(worker);
        vision["native_cache_home"]=json!(PathBuf::from(std::env::var_os("HOME").ok_or("HOME is missing")?).join("Library/Application Support/JustFit/native-home"));
        vision["native_prepare"]=json!(true);
        vision["native_distribution"]=serde_json::to_value(delivery)?;
    }
    if !media_enabled{plan["config"]["vision"]=Value::Null;}
    if !model_spec.starts_with(['/', '.', '~'])&&!std::path::Path::new(model_spec).is_dir(){plan["config"]["model"]=json!(model_spec);}
    options.apply(&mut plan,false)?;
    for name in ["cache","ane-meter"] {std::fs::create_dir_all(state.join(name))?;}
    for (name,_) in std::env::vars_os() {
        let text=name.to_string_lossy();
        if (["JUSTFIT_","DASH_","MLX_","APC_","ANE_"].iter().any(|prefix|text.starts_with(prefix)) && text!="JUSTFIT_CRASH_TRACE") || text=="PYTHONPATH" || text=="PYTHONHOME" {std::env::remove_var(name);}
    }
    for (name,value) in plan["environment"].as_object().unwrap() {std::env::set_var(name,value.as_str().unwrap());}
    std::env::set_current_dir(root.join("source"))?;
    let config=serde_json::from_value(plan["config"].clone())?;
    tokio::runtime::Builder::new_multi_thread().enable_all().worker_threads(2).build()?.block_on(crate::http::serve(config))
}
