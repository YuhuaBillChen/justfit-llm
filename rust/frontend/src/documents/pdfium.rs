//! PDFium public C ABI, confined to a disposable worker process.
use super::{Budget,Limits,Page};
use crate::{bad,Result};
use base64::{engine::general_purpose::STANDARD,Engine as _};
use std::{ffi::{c_char,c_int,c_ulong,c_void,CString},ptr};
type P=*mut c_void;
#[repr(C)]
struct Rect {left:f32,top:f32,right:f32,bottom:f32}
// Public 64-bit PDFium ABI: zero optional callbacks, no Javascript platform.
// Layout verified against the qualified ctypes bindings (size280, align8).
#[repr(C,align(8))]
struct FormInfo {version:i32,rest:[u8;276]}
#[repr(C,align(8))]
struct LibraryConfig {version:i32,rest:[u8;52]}
unsafe fn symbol<T:Copy>(library:P,name:&str)->Result<T> {
    let name=CString::new(name).unwrap();let p=unsafe{libc::dlsym(library,name.as_ptr())};
    if p.is_null(){return Err(bad("PDFium ABI unavailable"))}
    Ok(unsafe{std::mem::transmute_copy(&p)})
}
struct Handle {ptr:P,close:unsafe extern "C" fn(P)}
impl Drop for Handle {fn drop(&mut self){unsafe{(self.close)(self.ptr)}}}
pub fn render(payload:&[u8],limits:&Limits,budget:&mut Budget,library:&str,zlib:&str)->Result<Vec<Page>> {
    let path=CString::new(library).map_err(|_|bad("invalid PDFium library path"))?;
    let lib=unsafe{libc::dlopen(path.as_ptr(),libc::RTLD_NOW|libc::RTLD_LOCAL)};
    if lib.is_null(){return Err(crate::ApiError(503,"document worker could not start".into(),"document_unavailable".into()))}
    macro_rules! sym {($name:literal,$ty:ty)=>{unsafe{symbol::<$ty>(lib,$name)?}}}
    let init=sym!("FPDF_InitLibraryWithConfig",unsafe extern "C" fn(*const LibraryConfig));
    let config=LibraryConfig{version:2,rest:[0;52]};unsafe{init(&config)};
    let load=sym!("FPDF_LoadMemDocument64",unsafe extern "C" fn(*const c_void,usize,*const c_char)->P);
    let raw=unsafe{load(payload.as_ptr().cast(),payload.len(),ptr::null())};
    if raw.is_null(){
        let error=sym!("FPDF_GetLastError",unsafe extern "C" fn()->c_ulong);
        return Err(bad(match unsafe{error()} {4=>"PDF documents requiring an opening password are not supported",5=>"PDF security handler is not supported",_=>"PDF document could not be decoded"}));
    }
    let document=Handle{ptr:raw,close:sym!("FPDF_CloseDocument",unsafe extern "C" fn(P))};
    let count=sym!("FPDF_GetPageCount",unsafe extern "C" fn(P)->c_int);let count=unsafe{count(raw)};
    if count<1{return Err(bad("PDF document could not be decoded"))}
    let get_type=sym!("FPDF_GetFormType",unsafe extern "C" fn(P)->c_int);
    let mut form_info=FormInfo{version:2,rest:[0;276]};
    let form=if unsafe{get_type(raw)}!=0 {
        let init=sym!("FPDFDOC_InitFormFillEnvironment",unsafe extern "C" fn(P,*mut FormInfo)->P);
        let handle=unsafe{init(raw,&mut form_info)};
        if handle.is_null(){return Err(bad("PDF document could not be decoded"))}
        Some(Handle{ptr:handle,close:sym!("FPDFDOC_ExitFormFillEnvironment",unsafe extern "C" fn(P))})
    }else{None};
    if count as usize>limits.pages{return Err(bad(format!("PDF documents must contain 1–{} pages",limits.pages)))}
    let load_page=sym!("FPDF_LoadPage",unsafe extern "C" fn(P,c_int)->P);
    let close_page=sym!("FPDF_ClosePage",unsafe extern "C" fn(P));
    let after=sym!("FORM_OnAfterLoadPage",unsafe extern "C" fn(P,P));
    let before=sym!("FORM_OnBeforeClosePage",unsafe extern "C" fn(P,P));
    let width=sym!("FPDF_GetPageWidthF",unsafe extern "C" fn(P)->f32);
    let height=sym!("FPDF_GetPageHeightF",unsafe extern "C" fn(P)->f32);
    let bbox=sym!("FPDF_GetPageBoundingBox",unsafe extern "C" fn(P,*mut Rect)->c_int);
    let text_page=sym!("FPDFText_LoadPage",unsafe extern "C" fn(P)->P);
    let text_close=sym!("FPDFText_ClosePage",unsafe extern "C" fn(P));
    let chars=sym!("FPDFText_CountChars",unsafe extern "C" fn(P)->c_int);
    let text=sym!("FPDFText_GetBoundedText",unsafe extern "C" fn(P,f64,f64,f64,f64,*mut u16,c_int)->c_int);
    let bitmap=sym!("FPDFBitmap_CreateEx",unsafe extern "C" fn(c_int,c_int,c_int,P,c_int)->P);
    let fill=sym!("FPDFBitmap_FillRect",unsafe extern "C" fn(P,c_int,c_int,c_int,c_int,c_ulong));
    let draw=sym!("FPDF_RenderPageBitmap",unsafe extern "C" fn(P,P,c_int,c_int,c_int,c_int,c_int,c_int));
    let draw_forms=sym!("FPDF_FFLDraw",unsafe extern "C" fn(P,P,P,c_int,c_int,c_int,c_int,c_int,c_int));
    let bitmap_close=sym!("FPDFBitmap_Destroy",unsafe extern "C" fn(P));
    let mut pages=Vec::new();let mut characters=0usize;let mut bytes=0;
    for index in 0..count {
        budget.check()?;
        let raw_page=unsafe{load_page(document.ptr,index)};
        if raw_page.is_null(){return Err(bad("PDF document could not be decoded"))}
        let page=Handle{ptr:raw_page,close:close_page};
        if let Some(form)=&form {unsafe{after(page.ptr,form.ptr)}}
        let result=(||{
            let w=unsafe{width(page.ptr)} as f64;let h=unsafe{height(page.ptr)} as f64;
            if !w.is_finite()||!h.is_finite()||w<=0.||h<=0.||w.max(h)/w.min(h)>200.{return Err(bad("PDF page dimensions are invalid"))}
            let tp=unsafe{text_page(page.ptr)};if tp.is_null(){return Err(bad("PDF document could not be decoded"))}
            let tp=Handle{ptr:tp,close:text_close};let n=unsafe{chars(tp.ptr)};
            if n<0{return Err(bad("PDF document could not be decoded"))}
            characters+=n as usize;if characters>limits.text_characters{return Err(bad("PDF text exceeds the size limit"))}
            let mut rect=Rect{left:0.,top:0.,right:0.,bottom:0.};if unsafe{bbox(page.ptr,&mut rect)}==0{return Err(bad("PDF document could not be decoded"))}
            let length=unsafe{text(tp.ptr,rect.left as f64,rect.top as f64,rect.right as f64,rect.bottom as f64,ptr::null_mut(),0)};
            if length>2_000_002{return Err(bad("PDF text exceeds the size limit"))}
            let text=if length>0 {
                let mut value=vec![0u16;length as usize];let used=unsafe{text(tp.ptr,rect.left as f64,rect.top as f64,rect.right as f64,rect.bottom as f64,value.as_mut_ptr(),length)};
                if used<0||used>length{return Err(bad("PDF document could not be decoded"))}
                // Python decoder errors=ignore, followed by Python str.strip().
                std::char::decode_utf16(value.into_iter().take(used as usize)).filter_map(std::result::Result::ok).collect::<String>()
                    .trim_matches(|c:char|c=='\0'||c.is_whitespace()||matches!(c,'\u{1c}'..='\u{1f}')).to_string()
            }else{String::new()};drop(tp);
            let mut scale=2f64.min((limits.page_pixels as f64/w/h).sqrt());
            while (w*scale).ceil()*(h*scale).ceil()>limits.page_pixels as f64{scale*=0.99}
            let (iw,ih)=((w*scale).ceil() as usize,(h*scale).ceil() as usize);
            if iw==0||ih==0||iw.checked_mul(ih).is_none_or(|n|n>limits.page_pixels){return Err(bad("PDF document could not be rendered"))}
            let mut pixels=vec![0u8;iw*ih*3];let bm=unsafe{bitmap(iw as i32,ih as i32,2,pixels.as_mut_ptr().cast(),(iw*3)as i32)};
            if bm.is_null(){return Err(bad("PDF document could not be rendered"))}
            let bm=Handle{ptr:bm,close:bitmap_close};unsafe{fill(bm.ptr,0,0,iw as i32,ih as i32,0xffff_ffff);draw(bm.ptr,page.ptr,0,0,iw as i32,ih as i32,0,1|16|512)};
            if let Some(form)=&form{unsafe{draw_forms(form.ptr,bm.ptr,page.ptr,0,0,iw as i32,ih as i32,0,1|16|512)}}drop(bm);
            budget.check()?;let png=super::png::encode(&pixels,iw,ih,zlib)?;
            let page=Page{text:format!("PDF page {}:\n{text}\n",index+1),image_url:format!("data:image/png;base64,{}",STANDARD.encode(png))};
            bytes+=page.size();budget.charge(page.size())?;if bytes>limits.rendered_bytes{return Err(bad("rendered PDF exceeds the size limit"))}
            Ok(page)
        })();
        if let Some(form)=&form{unsafe{before(page.ptr,form.ptr)}}
        drop(page);pages.push(result?);
    }
    drop(form);drop(document);Ok(pages)
}
