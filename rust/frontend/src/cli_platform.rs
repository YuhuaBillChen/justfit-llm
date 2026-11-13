//! Platform and Metal discovery for public Rust CLI commands.
use serde_json::json;
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},process::Command};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn supported(system:&str,architecture:&str,version:&str)->std::result::Result<(),String>{
    if system!="macos"||architecture!="aarch64"{return Err("native build and inference require an Apple silicon Mac".into());}
    let mut parts=version.trim().split('.');let major=parts.next().and_then(|p|p.parse::<u64>().ok());let minor=parts.next().unwrap_or("0").parse::<u64>().ok();
    match (major,minor){(Some(major),Some(minor)) if (major,minor)>=(27,0)=>Ok(()),_=>Err("the complete JustFit runtime requires macOS 27.0 or later (CoreAI helpers)".into())}
}
pub(crate) fn require_mac()->Result<()> {
    let version=if cfg!(target_os="macos"){let output=Command::new("/usr/bin/sw_vers").arg("-productVersion").output()?;if !output.status.success(){return Err("cannot identify macOS release".into());}String::from_utf8(output.stdout)?}else{String::new()};
    supported(std::env::consts::OS,std::env::consts::ARCH,&version).map_err(Into::into)
}
fn complete(path:&Path)->bool{["metal","metallib"].iter().all(|name|path.join(name).is_file())}
fn directories(root:&Path,prefix:&str,suffix:&str)->Vec<PathBuf>{
    let mut rows=fs::read_dir(root).into_iter().flatten().filter_map(|entry|entry.ok()).filter_map(|entry|{let name=entry.file_name();let name=name.to_str()?;let path=entry.path();(name.starts_with(prefix)&&name.ends_with(suffix)&&path.is_dir()).then_some(path)}).collect::<Vec<_>>();rows.sort_by(|a,b|b.cmp(a));rows
}
pub(crate) fn find_metal_bin()->Option<PathBuf>{
    let found=["metal","metallib"].map(|name|{let output=Command::new("/usr/bin/xcrun").args(["-f",name]).output().ok()?;if !output.status.success(){return None;}let path=PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());path.parent().map(Path::to_path_buf)});
    if let [Some(first),Some(second)]=found{if first==second&&complete(&first){return Some(first);}}
    for root in directories(Path::new("/private/var/run/com.apple.security.cryptexd/mnt"),"com.apple.MobileAsset.MetalToolchain-",""){let path=root.join("Metal.xctoolchain/usr/bin");if complete(&path){return Some(path);}}
    let mut roots=vec![PathBuf::from("/Library/Developer/Toolchains")];if let Some(home)=std::env::var_os("HOME"){roots.push(PathBuf::from(home).join("Library/Developer/Toolchains"));}
    for root in roots{for path in directories(&root,"",".xctoolchain"){let path=path.join("usr/bin");if complete(&path){return Some(path);}}}
    for app in directories(Path::new("/Applications"),"Xcode",".app"){for root in directories(&app.join("Contents/Developer/Toolchains"),"",".xctoolchain"){let path=root.join("usr/bin");if complete(&path){return Some(path);}}}
    None
}
pub fn doctor(args:&[String])->Result<()> {
    let metal=match args{[]=>None,[flag,path] if flag=="--metal-bin"=>Some(PathBuf::from(path)),_=>return Err("usage: justfit doctor [--metal-bin PATH]".into())};
    let platform=require_mac();let reason=platform.as_ref().err().map(ToString::to_string);let tools:BTreeMap<_,_>=metal.as_ref().map(|path|["metal","metallib"].into_iter().map(|name|(name,path.join(name).is_file())).collect()).unwrap_or_default();
    println!("{}",serde_json::to_string_pretty(&json!({"version":crate::cli::VERSION,"native_platform_supported":platform.is_ok(),"reason":reason,"dependencies_ok":true,"dependency_details":"Native Rust CLI; no Python package dependencies","metal_tools":tools}))?);
    if platform.is_err()||tools.values().any(|present|!*present){std::process::exit(1);}Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn same_reference_platform_floor(){
        assert!(supported("macos","aarch64","27.0").is_ok());assert!(supported("macos","aarch64","28.1.2").is_ok());
        for (system,arch,version) in [("linux","aarch64","27"),("macos","x86_64","27"),("macos","aarch64","26.4"),("macos","aarch64","unknown")]{assert!(supported(system,arch,version).is_err());}
    }
}
