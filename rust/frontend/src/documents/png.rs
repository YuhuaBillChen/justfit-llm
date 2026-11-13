//! RGB PNG encoding with the released Pillow filter order and zlib settings.
//! No metadata; opaque RGB only. Compression ABI is loaded in the PDF worker.
use crate::{bad, Result};
use std::ffi::{c_char,c_int,c_uint,c_ulong,c_void,CString};
#[repr(C)]
struct Stream {
    next_in:*const u8,avail_in:c_uint,total_in:c_ulong,
    next_out:*mut u8,avail_out:c_uint,total_out:c_ulong,
    msg:*mut c_char,state:*mut c_void,
    zalloc:Option<unsafe extern "C" fn(*mut c_void,c_uint,c_uint)->*mut c_void>,
    zfree:Option<unsafe extern "C" fn(*mut c_void,*mut c_void)>,
    opaque:*mut c_void,data_type:c_int,adler:c_ulong,reserved:c_ulong,
}
unsafe fn symbol<T:Copy>(library:*mut c_void,name:&str)->Result<T> {
    let name=CString::new(name).unwrap();let p=unsafe{libc::dlsym(library,name.as_ptr())};
    if p.is_null(){return Err(bad("PNG compression ABI unavailable"))}
    // Every caller supplies the exact zlib public ABI signature.
    Ok(unsafe{std::mem::transmute_copy(&p)})
}
fn chunk(out:&mut Vec<u8>,kind:&[u8;4],data:&[u8]) {
    out.extend((data.len() as u32).to_be_bytes());out.extend(kind);out.extend(data);
    let mut crc=0xffff_ffffu32;
    for b in kind.iter().chain(data) {
        crc^=*b as u32;
        for _ in 0..8 {crc=(crc>>1)^((0u32.wrapping_sub(crc&1))&0xedb8_8320)}
    }
    out.extend((!crc).to_be_bytes());
}
fn distance(v:u8)->usize {v.min(v.wrapping_neg()) as usize}
fn filtered(rgb:&[u8],w:usize)->Vec<u8> {
    let stride=w*3;let zeros=vec![0u8;stride];let mut previous=zeros.as_slice();
    let mut result=Vec::with_capacity(rgb.len()+rgb.len()/stride);
    for row in rgb.chunks_exact(stride) {
        let mut best=row.to_vec();let mut score:usize=row.iter().map(|v|distance(*v)).sum();let mut kind=0;
        for filter in [2,1,4] {
            if score==0 {break}
            let mut candidate=Vec::with_capacity(stride);let mut sum=0;
            for (i,&v) in row.iter().enumerate() {
                let a=if i<3{0}else{row[i-3]} as i32;let b=previous[i] as i32;
                let c=if i<3{0}else{previous[i-3]} as i32;
                let predictor=match filter {
                    2=>b,1=>a,
                    _=>{let pa=(b-c).abs();let pb=(a-c).abs();let pc=(a+b-2*c).abs();if pa<=pb&&pa<=pc{a}else if pb<=pc{b}else{c}}
                };
                let delta=v.wrapping_sub(predictor as u8);sum+=distance(delta);candidate.push(delta);
            }
            if sum<score {best=candidate;score=sum;kind=filter}
        }
        result.push(kind);result.extend(best);previous=row;
    }
    result
}
pub fn encode(rgb:&[u8],w:usize,h:usize,library:&str)->Result<Vec<u8>> {
    if w==0||h==0||w.checked_mul(h).and_then(|v|v.checked_mul(3))!=Some(rgb.len()) {return Err(bad("invalid PDF bitmap"))}
    let path=CString::new(library).map_err(|_|bad("invalid PNG compression library"))?;
    let lib=unsafe{libc::dlopen(path.as_ptr(),libc::RTLD_NOW|libc::RTLD_LOCAL)};
    if lib.is_null(){return Err(bad("PNG compression library unavailable"))}
    // Library is worker-process-lived; zlib allocations are released explicitly.
    let version:unsafe extern "C" fn()->*const c_char=unsafe{symbol(lib,"zlibVersion")?};
    let init:unsafe extern "C" fn(*mut Stream,c_int,c_int,c_int,c_int,c_int,*const c_char,c_int)->c_int=unsafe{symbol(lib,"deflateInit2_")?};
    let deflate:unsafe extern "C" fn(*mut Stream,c_int)->c_int=unsafe{symbol(lib,"deflate")?};
    let end:unsafe extern "C" fn(*mut Stream)->c_int=unsafe{symbol(lib,"deflateEnd")?};
    let input=filtered(rgb,w);let mut output=Vec::new();let mut buffer=vec![0u8;65536];
    let mut stream:Stream=unsafe{std::mem::zeroed()};
    if unsafe{init(&mut stream,-1,8,15,9,1,version(),std::mem::size_of::<Stream>() as i32)}!=0{return Err(bad("PNG compression failed"))}
    stream.next_out=buffer.as_mut_ptr();stream.avail_out=buffer.len() as u32;
    let compressed=(||->Result<()> {
        // Match Pillow's row-at-a-time Z_NO_FLUSH calls. zlib-ng's
        // decisions depend on how much input is available in each call.
        for row in input.chunks_exact(w*3+1) {
            stream.next_in=row.as_ptr();stream.avail_in=row.len() as u32;
            while stream.avail_in>0 {
                if unsafe{deflate(&mut stream,0)}<0{return Err(bad("PNG compression failed"))}
                if stream.avail_out==0 {output.extend_from_slice(&buffer);stream.next_out=buffer.as_mut_ptr();stream.avail_out=buffer.len() as u32;}
            }
        }
        loop {
            let result=unsafe{deflate(&mut stream,4)};
            if result==1 {output.extend_from_slice(&buffer[..buffer.len()-stream.avail_out as usize]);break}
            if result<0{return Err(bad("PNG compression failed"))}
            if stream.avail_out==0 {output.extend_from_slice(&buffer);stream.next_out=buffer.as_mut_ptr();stream.avail_out=buffer.len() as u32;}
        }
        Ok(())
    })();
    unsafe{end(&mut stream)};compressed?;
let mut png=b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header=Vec::new();header.extend((w as u32).to_be_bytes());header.extend((h as u32).to_be_bytes());header.extend([8,2,0,0,0]);
    chunk(&mut png,b"IHDR",&header);for part in output.chunks(65536){chunk(&mut png,b"IDAT",part)}chunk(&mut png,b"IEND",&[]);Ok(png)
}

/// dyld shared-cache libraries can load even when no filesystem entry exists.
pub(super) fn available(path:&std::path::Path)->bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path)=CString::new(path.as_os_str().as_bytes())else{return false};
    let handle=unsafe{libc::dlopen(path.as_ptr(),libc::RTLD_NOW|libc::RTLD_LOCAL)};
    if handle.is_null(){return false}
    let valid=["zlibVersion","deflateInit2_","deflate","deflateEnd"].iter().all(|name|{
        let name=CString::new(*name).unwrap();!unsafe{libc::dlsym(handle,name.as_ptr())}.is_null()
    });
    unsafe{libc::dlclose(handle)};valid
}
#[cfg(all(test,target_os="macos"))]
mod system_zlib_tests {
    use super::*;
    #[test]
    fn system_shared_cache_zlib_loads_and_encodes_without_a_disk_file() {
        let path=std::path::Path::new("/usr/lib/libz.dylib");
        assert!(available(path));assert!(!available(std::path::Path::new("/missing/zlib")));
        let rgb=[1,2,3,4,5,6];let bytes=encode(&rgb,2,1,path.to_str().unwrap()).unwrap();
        let decoded=image::load_from_memory(&bytes).unwrap().to_rgb8();assert_eq!(decoded.as_raw(),&rgb);
    }
}
