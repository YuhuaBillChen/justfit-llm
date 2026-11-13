//! Bounded, pinned HTTPS transfers used by the standalone Rust installer.
use sha2::{Digest,Sha256};
use std::{fs,io::{Read,Write},path::{Path,PathBuf},sync::atomic::{AtomicU64,Ordering},time::Duration};
pub type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
static NEXT:AtomicU64=AtomicU64::new(0);
pub fn digest(value:&str)->Result<&str> {
    if value.len()!=64 || !value.bytes().all(|c|c.is_ascii_digit()||(b'a'..=b'f').contains(&c)){return Err("SHA256 must be 64 lowercase hexadecimal characters".into());}Ok(value)
}
pub fn agent()->ureq::Agent {
    let tls=ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build();
    ureq::Agent::config_builder().https_only(true).tls_config(tls).timeout_connect(Some(Duration::from_secs(30))).timeout_per_read(Some(Duration::from_secs(60))).build().into()
}
pub fn fetch(url:&str,path:&Path,limit:u64,expected:&str)->Result<u64> {
    fetch_with(&agent(),url,path,limit,expected)
}
pub(crate) fn fetch_with(agent:&ureq::Agent,url:&str,path:&Path,limit:u64,expected:&str)->Result<u64> {
    fetch_authenticated_with(agent,url,path,limit,expected,None)
}
pub(crate) fn fetch_resumable_authenticated(url:&str,path:&Path,limit:u64,expected:&str,token:Option<&str>)->Result<u64> {
    fetch_resumable_with(&agent(),url,path,limit,expected,token)
}
fn response(agent:&ureq::Agent,url:&str,token:Option<&str>)->Result<ureq::http::Response<ureq::Body>> {
    response_at(agent,url,token,None)
}
fn response_at(agent:&ureq::Agent,url:&str,token:Option<&str>,offset:Option<u64>)->Result<ureq::http::Response<ureq::Body>> {
    if !url.starts_with("https://"){return Err("downloads require HTTPS".into());}
    let mut request=agent.get(url).header("Accept-Encoding","identity");
    if let Some(offset)=offset {request=request.header("Range",format!("bytes={offset}-"));}
    if let Some(token)=token.filter(|_|url.starts_with("https://huggingface.co/")) {request=request.header("Authorization",format!("Bearer {token}"));}
    // ureq's redirect-auth policy is Never; bearer tokens are not sent to CDNs.
    Ok(request.call().map_err(|error|match error {ureq::Error::StatusCode(code)=>format!("download HTTP status {code}"),_=>"HTTPS download failed".to_string()})?)
}
/// Inspect a bounded immutable metadata prefix without materializing weights.
pub(crate) fn inspect_prefix<T>(url:&str,token:Option<&str>,inspect:impl FnOnce(&mut dyn Read)->Result<T>)->Result<T>{
    let mut response=response_at(&agent(),url,token,Some(0))?;inspect(&mut response.body_mut().as_reader())
}
pub(crate) fn metadata(url:&str,limit:u64,token:Option<&str>)->Result<Vec<u8>> {
    metadata_with(&agent(),url,limit,token)
}
pub(crate) fn metadata_with(agent:&ureq::Agent,url:&str,limit:u64,token:Option<&str>)->Result<Vec<u8>> {
    if limit>64*1024*1024{return Err("metadata limit exceeds 64 MiB".into());}
    let mut response=response(agent,url,token)?;let mut bytes=vec![];
    response.body_mut().as_reader().take(limit+1).read_to_end(&mut bytes).map_err(|_|"HTTPS metadata body read failed")?;
    if bytes.len() as u64>limit{return Err("metadata exceeds size limit".into());}Ok(bytes)
}
fn fetch_authenticated_with(agent:&ureq::Agent,url:&str,path:&Path,limit:u64,expected:&str,token:Option<&str>)->Result<u64> {
    digest(expected)?;
    if !url.starts_with("https://"){return Err("downloads require HTTPS".into());}
    // Exclusive creation protects owner files, even if the network call fails.
    use std::os::unix::fs::OpenOptionsExt;
    let mut output=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;
    let result=(|| {
        let mut response=response(agent,url,token)?;
        if response.headers().get("content-length").and_then(|v|v.to_str().ok()).and_then(|v|v.parse::<u64>().ok()).is_some_and(|bytes|bytes>limit){return Err("download exceeds size limit".into());}
        let mut reader=response.body_mut().as_reader();let mut hash=Sha256::new();let mut total=0u64;let mut bytes=[0u8;65536];
        loop {
            let n=reader.read(&mut bytes).map_err(|_|"HTTPS body read failed")?;if n==0{break;}
            total=total.checked_add(n as u64).ok_or("download size overflow")?;
            if total>limit{return Err("download exceeds size limit".into());}
            hash.update(&bytes[..n]);output.write_all(&bytes[..n])?;
        }
        if format!("{:x}",hash.finalize())!=expected{return Err("download SHA256 mismatch".into());}
        output.sync_all()?;Ok(total)
    })();
    drop(output);
    if result.is_err(){fs::remove_file(path)?;}
    result
}
/// Only for a model installer's receipt-owned .part file. Failed reads retain
/// its received prefix; callers publish it exclusively only after this returns.
pub(crate) fn fetch_resumable_with(agent:&ureq::Agent,url:&str,path:&Path,limit:u64,expected:&str,token:Option<&str>)->Result<u64> {
    use std::{io::{Seek,SeekFrom},os::unix::fs::OpenOptionsExt};
    digest(expected)?;if !url.starts_with("https://")||limit==0{return Err("resumable downloads require HTTPS and a positive size".into());}
    let mut output=fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(path)?;
    if !output.metadata()?.is_file()||output.metadata()?.len()>limit{return Err("invalid owned download partial".into());}
    let mut offset=output.metadata()?.len();let mut hash=Sha256::new();let mut buffer=[0u8;65536];
    loop{let n=output.read(&mut buffer)?;if n==0{break;}hash.update(&buffer[..n]);}
    if offset==limit {
        if format!("{:x}",hash.clone().finalize())==expected{return Ok(limit);}
        // This is verified installer-owned scratch, never an installed artifact.
        output.set_len(0)?;output.seek(SeekFrom::Start(0))?;offset=0;hash=Sha256::new();
    }
    let mut response=response_at(agent,url,token,(offset>0).then_some(offset))?;
    match response.status().as_u16() {
        200=>{if offset>0{output.set_len(0)?;output.seek(SeekFrom::Start(0))?;offset=0;hash=Sha256::new();}},
        206=>{
            let valid=response.headers().get("content-range").and_then(|v|v.to_str().ok()).and_then(|v|v.strip_prefix("bytes ")).and_then(|v|v.split_once('/')).and_then(|(range,total)|Some((range.split_once('-')?,total.parse::<u64>().ok()?))).is_some_and(|((start,end),total)|start.parse::<u64>().ok()==Some(offset)&&end.parse::<u64>().ok()==Some(limit-1)&&total==limit);
            if !valid{return Err("download Content-Range does not match the pinned artifact".into());}
        },
        _=>return Err("unexpected resumable download HTTP status".into()),
    }
    if response.headers().get("content-length").and_then(|v|v.to_str().ok()).and_then(|v|v.parse::<u64>().ok()).is_some_and(|n|n>limit-offset){return Err("download exceeds size limit".into());}
    let mut reader=response.body_mut().as_reader();let mut total=offset;
    loop {
        let n=reader.read(&mut buffer).map_err(|_|"HTTPS body interrupted; owned partial retained")?;if n==0{break;}
        total=total.checked_add(n as u64).ok_or("download size overflow")?;if total>limit{return Err("download exceeds size limit".into());}
        output.write_all(&buffer[..n])?;hash.update(&buffer[..n]);
    }
    if total!=limit{return Err("HTTPS body incomplete; owned partial retained".into());}
    if format!("{:x}",hash.finalize())!=expected{return Err("download SHA256 mismatch".into());}
    output.sync_all()?;Ok(total)
}
/// Owned same-volume staging. Names never come from downloaded archives.
pub struct Stage {path:PathBuf}
impl Stage {
    pub fn new(root:&Path,prefix:&str)->Result<Self> {
        if !prefix.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'-'){return Err("invalid staging prefix".into());}
        use std::os::unix::fs::DirBuilderExt;
        for _ in 0..32 {
            let path=root.join(format!(".{prefix}-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            match fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(())=>return Ok(Self{path}),Err(error) if error.kind()==std::io::ErrorKind::AlreadyExists=>continue,Err(error)=>return Err(error.into()),
            }
        }
        Err("cannot allocate exclusive installer staging".into())
    }
    pub fn path(&self)->&Path{&self.path}
}
impl Drop for Stage {fn drop(&mut self){let _=fs::remove_dir_all(&self.path);}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn insecure_urls_and_bad_pins_do_not_create_or_replace_files() {
        let stage=Stage::new(&std::env::temp_dir(),"justfit-download-test").unwrap();let target=stage.path().join("owner-file");
        for url in ["http://127.0.0.1:1","file:///etc/passwd","HTTPS://invalid"] {
            assert!(fetch(url,&target,1024,&"a".repeat(64)).unwrap_err().to_string().contains("HTTPS"));assert!(!target.exists());
        }
        assert!(fetch("https://127.0.0.1:1",&target,1024,"bad-pin").is_err());assert!(!target.exists());
        fs::write(&target,b"owner data").unwrap();
        assert!(fetch("https://127.0.0.1:1",&target,1024,&"a".repeat(64)).is_err());assert_eq!(fs::read(target).unwrap(),b"owner data");
    }
}

#[cfg(test)]
mod https_contract {
    use super::*;
    #[test]
    #[ignore = "requires the M4 controlled TLS fixture"]
    fn pinned_https_transfer_contract() {
        let url=std::env::var("JUSTFIT_TEST_HTTPS_URL").expect("TLS fixture URL");
        let cert=fs::read(std::env::var("JUSTFIT_TEST_HTTPS_CERT").expect("TLS fixture certificate")).unwrap();
        let cert=ureq::tls::Certificate::from_pem(&cert).unwrap().to_owned();
        let tls=ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::Specific(vec![cert].into())).build();
        let agent:ureq::Agent=ureq::Agent::config_builder().https_only(true).tls_config(tls).timeout_connect(Some(Duration::from_secs(5))).timeout_per_read(Some(Duration::from_secs(5))).build().into();
        let stage=Stage::new(&std::env::temp_dir(),"justfit-https-test").unwrap();let target=stage.path().join("download");
        let data=b"verified HTTPS runtime bytes";let pin=format!("{:x}",Sha256::digest(data));
        for endpoint in ["/ok","/chunked"] {
            assert_eq!(fetch_with(&agent,&format!("{url}{endpoint}"),&target,1024,&pin).unwrap(),data.len() as u64);
            assert_eq!(fs::read(&target).unwrap(),data);fs::remove_file(&target).unwrap();
        }
        for (endpoint,limit,expected,message) in [("/ok",1024,"a".repeat(64),"SHA256"),("/ok",1,pin.clone(),"size limit"),("/chunked",1,pin.clone(),"size limit"),("/missing",1024,pin.clone(),"404"),("/downgrade",1024,pin.clone(),"HTTPS download failed")] {
            let error=fetch_with(&agent,&format!("{url}{endpoint}"),&target,limit,&expected).unwrap_err();
            assert!(error.to_string().contains(message),"{endpoint}: {error}");assert!(!target.exists());
        }
        // The unmodified production trust store must reject this private fixture.
        assert!(fetch(&format!("{url}/ok"),&target,1024,&pin).is_err());assert!(!target.exists());
        fs::write(&target,b"owner data").unwrap();assert!(fetch_with(&agent,&format!("{url}/ok"),&target,1024,&pin).is_err());assert_eq!(fs::read(target).unwrap(),b"owner data");
    }
    #[test]
    #[ignore = "requires the M4 controlled TLS range fixture"]
    fn owned_partial_range_resume_and_server_failures() {
        let url=std::env::var("JUSTFIT_TEST_HTTPS_URL").unwrap();let cert=fs::read(std::env::var("JUSTFIT_TEST_HTTPS_CERT").unwrap()).unwrap();
        let cert=ureq::tls::Certificate::from_pem(&cert).unwrap().to_owned();let tls=ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::Specific(vec![cert].into())).build();
        let agent:ureq::Agent=ureq::Agent::config_builder().https_only(true).tls_config(tls).timeout_per_read(Some(Duration::from_secs(5))).build().into();
        let stage=Stage::new(&std::env::temp_dir(),"justfit-range-test").unwrap();let path=stage.path().join("owned.part");
        let data=b"verified HTTPS runtime bytes";let pin=format!("{:x}",Sha256::digest(data));let size=data.len() as u64;
        for endpoint in ["/resume","/resume-ignore"] {
            fs::write(&path,&data[..7]).unwrap();assert_eq!(fetch_resumable_with(&agent,&format!("{url}{endpoint}"),&path,size,&pin,None).unwrap(),size);assert_eq!(fs::read(&path).unwrap(),data);
        }
        fs::write(&path,&data[..7]).unwrap();assert!(fetch_resumable_with(&agent,&format!("{url}/resume-wrong"),&path,size,&pin,None).is_err());assert_eq!(fs::read(&path).unwrap(),&data[..7]);
        assert!(fetch_resumable_with(&agent,&format!("{url}/resume-short"),&path,size,&pin,None).is_err());
        let received=fs::read(&path).unwrap();assert!(received.len()>7&&received.len()<data.len());assert_eq!(received,&data[..received.len()]);
        assert_eq!(fetch_resumable_with(&agent,&format!("{url}/resume"),&path,size,&pin,None).unwrap(),size);assert_eq!(fs::read(&path).unwrap(),data);
        // A complete verified partial is reusable even with no reachable server.
        assert_eq!(fetch_resumable_with(&agent,"https://127.0.0.1:1/unreachable",&path,size,&pin,None).unwrap(),size);
        fs::write(&path,vec![0u8;data.len()]).unwrap();assert_eq!(fetch_resumable_with(&agent,&format!("{url}/resume"),&path,size,&pin,None).unwrap(),size);assert_eq!(fs::read(&path).unwrap(),data);
        let owner=stage.path().join("owner");fs::write(&owner,b"owner data").unwrap();fs::remove_file(&path).unwrap();std::os::unix::fs::symlink(&owner,&path).unwrap();
        assert!(fetch_resumable_with(&agent,&format!("{url}/resume"),&path,size,&pin,None).is_err());assert_eq!(fs::read(owner).unwrap(),b"owner data");
    }

}
