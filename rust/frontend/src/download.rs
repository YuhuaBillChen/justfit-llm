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
    // A failed attempt removes its file, so each retry starts clean.
    retrying(||fetch_authenticated_with(agent,url,path,limit,expected,None))
}
pub(crate) fn fetch_resumable_authenticated(url:&str,path:&Path,limit:u64,expected:&str,token:Option<&str>)->Result<u64> {
    fetch_parallel_with(&agent(),url,path,limit,expected,token)
}
/// Files at least this large are fetched as parallel byte ranges. The Hub's CDN
/// slows each connection to 2-4 MB/s after its first ~64 MB, so one stream
/// takes most of an hour for a model; in our measurement eight workers that
/// fetch 16 MiB ranges, each on a new connection, reached about 57 MB/s.
const PARALLEL_MIN:u64=64*1024*1024;
/// Ranges are CHUNK bytes; WORKERS connections each take the next one, so a
/// slow connection holds back one chunk, not an eighth of the file.
const CHUNK:u64=16*1024*1024;
const WORKERS:u64=8;
fn segment_path(path:&Path,index:u64)->PathBuf {
    let mut name=path.as_os_str().to_owned();name.push(format!(".seg{index}"));PathBuf::from(name)
}
/// Resumable pinned download of an installer-owned partial. Large files are
/// fetched as CHUNK ranges over WORKERS parallel connections, each range kept
/// in its own `.segN` file so an interrupted run resumes every range; the
/// ranges are then joined into `path` and the whole file is checked against its
/// SHA-256. A partial left by a single-stream download (no segment files)
/// resumes as one stream.
pub(crate) fn fetch_parallel_with(agent:&ureq::Agent,url:&str,path:&Path,limit:u64,expected:&str,token:Option<&str>)->Result<u64> {
    digest(expected)?;if !url.starts_with("https://")||limit==0{return Err("resumable downloads require HTTPS and a positive size".into());}
    fetch_segments(path,limit,CHUNK,expected,|segment,start,end|fetch_range(agent,url,segment,start,end,limit,token),||retrying(||fetch_resumable_with(agent,url,path,limit,expected,token)))
}
fn fetch_segments(path:&Path,limit:u64,chunk:u64,expected:&str,range:impl Fn(&Path,u64,u64)->Result<u64>+Sync,single:impl FnOnce()->Result<u64>)->Result<u64> {
    let count=limit.div_ceil(chunk);let bounds=|index:u64|(index*chunk,((index+1)*chunk).min(limit));
    let clear=||{for index in 0..count{let _=fs::remove_file(segment_path(path,index));}};
    let first=(0..count).find(|index|segment_path(path,*index).exists());
    let prefix=fs::symlink_metadata(path).map(|meta|meta.len()).unwrap_or(0);
    if limit<PARALLEL_MIN||(prefix>0&&first.is_none()) {
        // Small file, or a single-stream partial (also what an interrupted join
        // leaves once every segment is in `path`): continue it as one stream.
        let result=single();if result.is_ok(){clear();}return result;
    }
    // Segments 0..joined are already in `path` if a join was interrupted; the
    // join removes each segment only after appending and syncing it.
    let mut joined=0;
    if let Some(next)=first.filter(|next|*next>0&&prefix>0) {
        let (start,end)=bounds(next);
        if prefix==start {joined=next;}
        else if prefix==end&&fs::metadata(segment_path(path,next))?.len()==end-start {fs::remove_file(segment_path(path,next))?;joined=next+1;}
        else {fs::remove_file(path)?;}
    } else if prefix>0 {fs::remove_file(path)?;}
    let stop=std::sync::atomic::AtomicBool::new(false);let logged=std::sync::atomic::AtomicU32::new(0);let next=AtomicU64::new(joined);
    let failures:Vec<String>=std::thread::scope(|scope| {
        let workers:Vec<_>=(0..WORKERS.min(count-joined)).map(|worker| {
            let (range,stop,logged,next,bounds)=(&range,&stop,&logged,&next,&bounds);
            scope.spawn(move||->std::result::Result<(),String> {
                loop {
                    let index=next.fetch_add(1,Ordering::Relaxed);if index>=count{return Ok(());}
                    let (start,end)=bounds(index);let segment=segment_path(path,index);
                    if let Err(error)=retry_loop(||range(&segment,start,end),Some(stop),worker,logged) {
                        // One range failing for good fails the whole file: stop the others.
                        stop.store(true,Ordering::Relaxed);return Err(error.to_string());
                    }
                }
            })
        }).collect();
        workers.into_iter().filter_map(|worker|worker.join().unwrap_or_else(|_|Err("download worker panicked".into())).err()).collect()
    });
    if failures.iter().any(|error|error==RANGE_IGNORED) {
        // A proxy that ignores Range answers 200: fall back to one stream.
        clear();if joined>0{fs::remove_file(path)?;}
        let result=single();if result.is_ok(){clear();}return result;
    }
    if let Some(error)=failures.into_iter().find(|error|error!=CANCELLED).or_else(||stop.load(Ordering::Relaxed).then(||CANCELLED.to_string())) {return Err(error.into());}
    // Join the ranges in order, hashing as we go; each segment is removed only
    // after it is appended and synced, so the extra disk space is one segment.
    use std::os::unix::fs::OpenOptionsExt;
    let mut output=fs::OpenOptions::new().read(true).write(true).create(true).truncate(joined==0).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(path)?;
    let mut hash=Sha256::new();let mut total=0u64;let mut buffer=vec![0u8;1<<20];
    loop{let n=output.read(&mut buffer)?;if n==0{break;}hash.update(&buffer[..n]);total+=n as u64;}
    if total!=bounds(joined).0 {drop(output);fs::remove_file(path)?;return Err("download join prefix changed; restart".into());}
    for index in joined..count {
        let segment=segment_path(path,index);let mut input=fs::File::open(&segment)?;
        loop{let n=input.read(&mut buffer)?;if n==0{break;}hash.update(&buffer[..n]);output.write_all(&buffer[..n])?;total+=n as u64;}
        output.sync_all()?;fs::remove_file(&segment)?;
    }
    if total!=limit||format!("{:x}",hash.finalize())!=expected {
        drop(output);fs::remove_file(path)?;return Err("download SHA256 mismatch".into());
    }
    Ok(total)
}
const CANCELLED:&str="download cancelled after another range failed";
const RANGE_IGNORED:&str="the server ignored the byte range";
/// One byte range [start, end) of a pinned file into its segment file, resuming
/// from the segment's current length.
fn fetch_range(agent:&ureq::Agent,url:&str,segment:&Path,start:u64,end:u64,limit:u64,token:Option<&str>)->Result<u64> {
    use std::os::unix::fs::OpenOptionsExt;
    let want=end-start;
    let mut output=fs::OpenOptions::new().append(true).create(true).mode(0o600).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(segment)?;
    let have=output.metadata()?.len();if !output.metadata()?.is_file()||have>want{return Err("invalid owned download partial".into());}
    if have==want{return Ok(want);}
    let mut response=response_range(agent,url,token,Some((start+have,Some(end-1))))?;
    match response.status().as_u16(){206=>(),200=>return Err(RANGE_IGNORED.into()),_=>return Err("unexpected ranged download HTTP status".into())}
    let valid=response.headers().get("content-range").and_then(|v|v.to_str().ok()).and_then(|v|v.strip_prefix("bytes ")).and_then(|v|v.split_once('/')).and_then(|(range,total)|Some((range.split_once('-')?,total.parse::<u64>().ok()?))).is_some_and(|((first,last),total)|first.parse::<u64>().ok()==Some(start+have)&&last.parse::<u64>().ok()==Some(end-1)&&total==limit);
    if !valid{return Err("download Content-Range does not match the pinned artifact".into());}
    let mut reader=response.body_mut().as_reader();let mut buffer=vec![0u8;1<<16];let mut got=have;
    loop {
        let n=reader.read(&mut buffer).map_err(|_|"HTTPS body interrupted; owned partial retained")?;if n==0{break;}
        got=got.checked_add(n as u64).ok_or("download size overflow")?;if got>want{return Err("download exceeds size limit".into());}
        output.write_all(&buffer[..n])?;
    }
    if got!=want{return Err("HTTPS body incomplete; owned partial retained".into());}
    output.sync_all()?;Ok(want)
}
fn response(agent:&ureq::Agent,url:&str,token:Option<&str>)->Result<ureq::http::Response<ureq::Body>> {
    response_at(agent,url,token,None)
}
fn response_at(agent:&ureq::Agent,url:&str,token:Option<&str>,offset:Option<u64>)->Result<ureq::http::Response<ureq::Body>> {
    response_range(agent,url,token,offset.map(|offset|(offset,None)))
}
fn response_range(agent:&ureq::Agent,url:&str,token:Option<&str>,range:Option<(u64,Option<u64>)>)->Result<ureq::http::Response<ureq::Body>> {
    if !url.starts_with("https://"){return Err("downloads require HTTPS".into());}
    let mut request=agent.get(url).header("Accept-Encoding","identity");
    if let Some((first,last))=range {request=request.header("Range",format!("bytes={first}-{}",last.map(|last|last.to_string()).unwrap_or_default()));}
    // A bounded range gets its own connection: the CDN slows a connection down
    // after its first tens of MB, so reusing one would lose the parallel gain.
    if matches!(range,Some((_,Some(_)))) {request=request.header("Connection","close");}
    if let Some(token)=token.filter(|_|url.starts_with("https://huggingface.co/")) {request=request.header("Authorization",format!("Bearer {token}"));}
    // ureq's redirect-auth policy is Never; bearer tokens are not sent to CDNs.
    Ok(request.call().map_err(|error|transport_error(&error))?)
}
/// A redacted category of a failed transfer: never the URL, headers or token.
pub(crate) fn transport_error(error:&ureq::Error)->String {
    let kind=match error {
        ureq::Error::StatusCode(code)=>return format!("download HTTP status {code}"),
        ureq::Error::Timeout(_)=>"timed out",
        ureq::Error::HostNotFound=>"host not found (DNS)",
        ureq::Error::ConnectionFailed=>"connection failed",
        ureq::Error::Io(_)=>"network I/O error",
        ureq::Error::Protocol(_)=>"HTTP protocol error",
        ureq::Error::RedirectFailed=>"redirect failed",
        ureq::Error::BodyExceedsLimit(_)=>"response too large",
        other if format!("{other:?}").contains("Tls")||format!("{other:?}").contains("Rustls")=>"TLS error",
        _=>"transfer error",
    };
    format!("HTTPS download failed: {kind}")
}
/// Run a resumable transfer, retrying transient failures a few times; each
/// retry continues from the bytes already on disk.
pub(crate) fn retrying<T>(transfer:impl FnMut()->Result<T>)->Result<T> {
    retry_loop(transfer,None,0,&std::sync::atomic::AtomicU32::new(0))
}
/// Waits of 5, 10, 20, 40 and then 60 s: about five minutes over eight retries.
/// Parallel workers add 0.7 s per worker index so they do not retry in step,
/// print each retry round once, and give up early when `stop` is set.
fn retry_loop<T>(mut transfer:impl FnMut()->Result<T>,stop:Option<&std::sync::atomic::AtomicBool>,index:u64,logged:&std::sync::atomic::AtomicU32)->Result<T> {
    let stopped=||stop.is_some_and(|stop|stop.load(Ordering::Relaxed));
    let mut attempt=0u32;
    loop {
        if stopped(){return Err(CANCELLED.into());}
        match transfer() {
            Err(error) if attempt<8&&transient(&error.to_string())=>{
                attempt+=1;
                let wait=Duration::from_secs((5u64<<(attempt-1).min(4)).min(60))+Duration::from_millis(700*index);
                if logged.fetch_max(attempt,Ordering::Relaxed)<attempt {
                    eprintln!("download interrupted ({error}); retrying in {} s ({attempt}/8), finished bytes are kept",wait.as_secs());
                }
                let until=std::time::Instant::now()+if cfg!(test){Duration::ZERO}else{wait};
                while std::time::Instant::now()<until {
                    if stopped(){return Err(CANCELLED.into());}
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
            other=>return other,
        }
    }
}
/// Whether a failed transfer is worth retrying: transport failures, 429 and 5xx.
pub(crate) fn transient(message:&str)->bool {
    message.starts_with("HTTPS download failed")||message.contains("body read failed")||message.starts_with("HTTPS body ")||
    message.strip_prefix("download HTTP status ").and_then(|c|c.parse::<u16>().ok()).is_some_and(|c|c==429||c>=500)
}
/// Inspect a bounded immutable metadata prefix without materializing weights.
pub(crate) fn inspect_prefix<T>(url:&str,token:Option<&str>,inspect:impl FnOnce(&mut dyn Read)->Result<T>)->Result<T>{
    let agent=agent();let mut response=retrying(||response_at(&agent,url,token,Some(0)))?;inspect(&mut response.body_mut().as_reader())
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
    fn large_files_download_as_parallel_ranges_that_resume() {
        use std::sync::atomic::AtomicUsize;
        let stage=Stage::new(&std::env::temp_dir(),"justfit-segments-test").unwrap();let path=stage.path().join("weights.part");
        let chunk=4*1024*1024;let limit=PARALLEL_MIN+12345;let count=limit.div_ceil(chunk);
        let data:Vec<u8>=(0..limit).map(|i|(i.wrapping_mul(2654435761)>>13) as u8).collect();let pin=format!("{:x}",Sha256::digest(&data));
        let seg=|i:u64|segment_path(&path,i);let none_left=||(0..count).all(|i|!seg(i).exists());
        // First run: ranges stop halfway with a transient error, 2 x count times in all.
        let failures=AtomicUsize::new(0);
        let range=|segment:&Path,start:u64,end:u64|->Result<u64>{
            let have=fs::metadata(segment).map(|m|m.len()).unwrap_or(0);let mut file=fs::OpenOptions::new().append(true).create(true).open(segment)?;
            let stop=if failures.fetch_add(1,Ordering::Relaxed)<2*count as usize{start+have+(end-start-have)/2}else{end};
            file.write_all(&data[(start+have) as usize..stop as usize])?;
            if stop<end{return Err("HTTPS body interrupted; owned partial retained".into());}Ok(end-start)
        };
        assert_eq!(fetch_segments(&path,limit,chunk,&pin,range,||panic!("parallel expected")).unwrap(),limit);
        assert_eq!(fs::read(&path).unwrap(),data);assert!(none_left());
        // A wrong pin removes the joined file.
        fs::remove_file(&path).unwrap();
        let exact=|segment:&Path,start:u64,end:u64|->Result<u64>{fs::write(segment,&data[start as usize..end as usize])?;Ok(end-start)};
        assert!(fetch_segments(&path,limit,chunk,&"a".repeat(64),exact,||panic!()).unwrap_err().to_string().contains("SHA256"));assert!(!path.exists());
        // A join interrupted after two ranges resumes the join without refetching them.
        fs::write(&path,&data[..2*chunk as usize]).unwrap();
        for i in 2..count{fs::write(seg(i),&data[(i*chunk) as usize..((i+1)*chunk).min(limit) as usize]).unwrap();}
        let fetched=AtomicUsize::new(0);
        let counted=|_:&Path,start:u64,end:u64|->Result<u64>{assert!(start>=2*chunk,"joined range refetched");fetched.fetch_add(1,Ordering::Relaxed);Ok(end-start)};
        assert_eq!(fetch_segments(&path,limit,chunk,&pin,counted,||panic!()).unwrap(),limit);assert_eq!(fs::read(&path).unwrap(),data);assert_eq!(fetched.load(Ordering::Relaxed),(count-2) as usize);
        // A permanent error on one range stops the other workers early.
        fs::remove_file(&path).unwrap();let calls=AtomicUsize::new(0);
        let broken=|_:&Path,start:u64,_:u64|->Result<u64>{calls.fetch_add(1,Ordering::Relaxed);if start==0{Err("download HTTP status 404".into())}else{std::thread::sleep(Duration::from_millis(50));Err("HTTPS body interrupted; owned partial retained".into())}};
        assert!(fetch_segments(&path,limit,chunk,&pin,broken,||panic!()).unwrap_err().to_string().contains("404"));assert!(calls.load(Ordering::Relaxed)<20);
        // A proxy that ignores Range falls back to one stream.
        let ignored=|_:&Path,_:u64,_:u64|->Result<u64>{Err(RANGE_IGNORED.into())};
        assert_eq!(fetch_segments(&path,limit,chunk,&pin,ignored,||Ok(11)).unwrap(),11);assert!(none_left());
        // A single-stream partial without segments resumes as one stream; small files never split.
        fs::write(&path,&data[..100]).unwrap();assert_eq!(fetch_segments(&path,limit,chunk,&pin,exact,||Ok(7)).unwrap(),7);
        fs::remove_file(&path).unwrap();assert_eq!(fetch_segments(&path,1024,chunk,&pin,exact,||Ok(9)).unwrap(),9);
    }
    #[test]
    #[ignore = "downloads 3.8 GB from the real Hugging Face Hub"]
    fn real_hub_parallel_download_resumes_after_interruption() {
        let url="https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/dedf8df68adfb1afeaf7b7480c0a0243108177b4/model.safetensors";
        let (bytes,pin)=(3848817896u64,"67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c");
        let stage=Stage::new(&std::env::temp_dir(),"justfit-real-hub").unwrap();let path=stage.path().join("draft.part");
        // The first run stops for good after 20 ranges, leaving finished and missing segments.
        let started=std::time::Instant::now();let taken=std::sync::atomic::AtomicUsize::new(0);let agent=agent();
        let first=fetch_segments(&path,bytes,CHUNK,pin,|segment,start,end|{if taken.fetch_add(1,Ordering::Relaxed)>=20{return Err("test stop".into());}fetch_range(&agent,url,segment,start,end,bytes,None)},||panic!("parallel expected"));
        assert!(first.is_err());
        let partial:u64=(0..bytes.div_ceil(CHUNK)).map(|i|fs::metadata(segment_path(&path,i)).map(|m|m.len()).unwrap_or(0)).sum();
        eprintln!("interrupted after {:.0} s with {partial} bytes in segments",started.elapsed().as_secs_f64());
        assert!(partial>0&&partial<bytes);
        let resumed=std::time::Instant::now();
        assert_eq!(fetch_parallel_with(&agent,url,&path,bytes,pin,None).unwrap(),bytes);
        let seconds=resumed.elapsed().as_secs_f64();
        eprintln!("resumed and verified in {seconds:.0} s: {:.1} MB/s for the remaining {} bytes",(bytes-partial) as f64/seconds/1e6,bytes-partial);
        assert!((0..bytes.div_ceil(CHUNK)).all(|i|!segment_path(&path,i).exists()));assert_eq!(fs::metadata(&path).unwrap().len(),bytes);
    }
    #[test]
    fn interrupted_transfers_are_retried_and_permanent_errors_are_not() {
        for message in ["HTTPS body interrupted; owned partial retained","HTTPS body incomplete; owned partial retained","HTTPS download failed: timed out","download HTTP status 503"]{assert!(transient(message),"{message}");}
        for message in ["download SHA256 mismatch","download HTTP status 404","download exceeds size limit"]{assert!(!transient(message),"{message}");}
        let mut calls=0;
        assert_eq!(retrying(||{calls+=1;if calls<3{Err("HTTPS body interrupted; owned partial retained".into())}else{Ok(7)}}).unwrap(),7);assert_eq!(calls,3);
        calls=0;assert!(retrying(||->Result<u64>{calls+=1;Err("download SHA256 mismatch".into())}).is_err());assert_eq!(calls,1);
        calls=0;assert!(retrying(||->Result<u64>{calls+=1;Err("HTTPS body incomplete; owned partial retained".into())}).is_err());assert_eq!(calls,9);
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
