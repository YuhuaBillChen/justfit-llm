//! Narrow C ABI over the existing, pinned CoreAIBridge.swift library.
//! All raw pointers stay within this module. Handles retain their library and
//! buffers; checked tensor bounds precede every IOSurface read/write.
use crate::{bad, Result};
use serde::Deserialize;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    path::Path,
    ptr,
    rc::Rc,
};

type P = *mut c_void;
type Release = unsafe extern "C" fn(P);
type Load = unsafe extern "C" fn(*const c_char, i32, i32, *mut c_char, i32) -> P;
type Function = unsafe extern "C" fn(P, *const c_char, *mut c_char, i32) -> P;
type Describe = unsafe extern "C" fn(P, *mut c_char, i32) -> i32;
type Create = unsafe extern "C" fn(i32, i32, *const i64, *const i64, *mut c_char, i32) -> P;
type Address = unsafe extern "C" fn(P) -> P;
type Nbytes = unsafe extern "C" fn(P) -> i64;
type Bind = unsafe extern "C" fn(
    P,
    i32,
    *const *const c_char,
    *const P,
    i32,
    *const *const c_char,
    *const P,
    i32,
    *const *const c_char,
    *const P,
    *mut c_char,
    i32,
) -> P;
type Run = unsafe extern "C" fn(i32, *const P, *mut f64, *mut c_char, i32) -> i32;
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> P;
    fn dlsym(library: P, name: *const c_char) -> P;
    fn dlerror() -> *const c_char;
}

pub struct Api {
    _library: P,
    release: Release,
    load: Load,
    function: Function,
    describe: Describe,
    create: Create,
    address: Address,
    nbytes: Nbytes,
    bind: Bind,
    run: Run,
}
type MeterSelf = unsafe extern "C" fn(*mut u64) -> i32;
type MeterPid = unsafe extern "C" fn(i32, *mut u64) -> i32;
/// The library is intentionally process-lived; returned immutable function
/// pointers remain valid on the sampler thread until this worker exits.
pub fn meter_functions(path: &Path) -> Result<(MeterSelf, MeterPid)> {
    let path = cstring(
        path.to_str()
            .ok_or_else(|| bad("invalid meter library path"))?,
    )?;
    let library = unsafe { dlopen(path.as_ptr(), 2) };
    if library.is_null() {
        return Err(bad("memory attribution library unavailable"));
    }
    let name = cstring("attributed_self")?;
    let own = unsafe { dlsym(library, name.as_ptr()) };
    let name = cstring("attributed_pid")?;
    let pid = unsafe { dlsym(library, name.as_ptr()) };
    if own.is_null() || pid.is_null() {
        return Err(bad("memory attribution ABI unavailable"));
    }
    Ok(unsafe {
        (
            std::mem::transmute::<P, MeterSelf>(own),
            std::mem::transmute::<P, MeterPid>(pid),
        )
    })
}
fn cstring(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| bad("native string contains NUL"))
}
fn error(buf: &[c_char]) -> crate::ApiError {
    let raw = buf
        .iter()
        .take_while(|v| **v != 0)
        .map(|v| *v as u8)
        .collect::<Vec<_>>();
    bad(String::from_utf8_lossy(&raw))
}
impl Api {
    pub fn open(path: &Path) -> Result<Rc<Self>> {
        let path = cstring(path.to_str().ok_or_else(|| bad("invalid library path"))?)?;
        // SAFETY: dlopen copies this NUL-terminated path; the library remains
        // loaded for the worker process lifetime (Swift metadata is not unloaded).
        let lib = unsafe { dlopen(path.as_ptr(), 2) };
        if lib.is_null() {
            let p = unsafe { dlerror() };
            return Err(bad(if p.is_null() {
                "CoreAI library unavailable".into()
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }));
        }
        macro_rules! sym {
            ($name:literal,$ty:ty) => {{
                let name = cstring($name)?;
                let address = unsafe { dlsym(lib, name.as_ptr()) };
                if address.is_null() {
                    return Err(bad(concat!("missing CoreAI symbol ", $name)));
                }
                unsafe { std::mem::transmute::<P, $ty>(address) }
            }};
        }
        let version = sym!("cai_abi_version", unsafe extern "C" fn() -> i32);
        if unsafe { version() } != 1 {
            return Err(bad("unsupported CoreAI bridge ABI"));
        }
        Ok(Rc::new(Self {
            _library: lib,
            release: sym!("cai_release", Release),
            load: sym!("cai_model_load", Load),
            function: sym!("cai_function_load", Function),
            describe: sym!("cai_function_describe", Describe),
            create: sym!("cai_buffer_create", Create),
            address: sym!("cai_buffer_address", Address),
            nbytes: sym!("cai_buffer_nbytes", Nbytes),
            bind: sym!("cai_binding_create", Bind),
            run: sym!("cai_run", Run),
        }))
    }
    fn handle(self: &Rc<Self>, ptr: P, errors: &[c_char]) -> Result<Handle> {
        if ptr.is_null() {
            Err(error(errors))
        } else {
            Ok(Handle {
                ptr,
                api: self.clone(),
            })
        }
    }
    pub fn model(self: &Rc<Self>, path: &Path) -> Result<Handle> {
        let path = cstring(path.to_str().ok_or_else(|| bad("invalid asset path"))?)?;
        let mut err = [0; 4096];
        let p = unsafe { (self.load)(path.as_ptr(), 0, 1, err.as_mut_ptr(), 4096) };
        self.handle(p, &err)
    }
}
pub struct Handle {
    ptr: P,
    api: Rc<Api>,
}
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { (self.api.release)(self.ptr) }
    }
}
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Spec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub nbytes: usize,
    #[serde(default)]
    pub dynamic: bool,
}
#[derive(Deserialize)]
pub struct Descriptor {
    pub inputs: Vec<Spec>,
    pub outputs: Vec<Spec>,
    pub states: Vec<Spec>,
}
pub struct FnHandle {
    handle: Handle,
    pub descriptor: Descriptor,
    _model: Handle,
}
impl Handle {
    pub fn function(self, name: &str) -> Result<FnHandle> {
        let name = cstring(name)?;
        let mut err = [0; 4096];
        let ptr = unsafe { (self.api.function)(self.ptr, name.as_ptr(), err.as_mut_ptr(), 4096) };
        let handle = self.api.handle(ptr, &err)?;
        let len = unsafe { (self.api.describe)(handle.ptr, ptr::null_mut(), 0) };
        if !(1..=1024 * 1024).contains(&len) {
            return Err(bad("invalid CoreAI description length"));
        }
        let mut bytes = vec![0; len as usize];
        let written = unsafe { (self.api.describe)(handle.ptr, bytes.as_mut_ptr(), len) };
        if written != len {
            return Err(bad("CoreAI descriptor changed"));
        }
        let raw = bytes
            .iter()
            .take_while(|v| **v != 0)
            .map(|v| *v as u8)
            .collect::<Vec<_>>();
        let descriptor: Descriptor =
            serde_json::from_slice(&raw).map_err(|_| bad("invalid CoreAI descriptor JSON"))?;
        if !descriptor.states.is_empty() {
            return Err(bad("vision functions must be stateless"));
        }
        Ok(FnHandle {
            handle,
            descriptor,
            _model: self,
        })
    }
}
pub struct Buffer {
    handle: Handle,
    pub spec: Spec,
    address: *mut u8,
    nbytes: usize,
    element_bytes: usize,
}
fn fill_bytes(spec: &Spec, bytes: &mut [u8], element_bytes: usize, value: f32) -> Result<()> {
    let encoded = if element_bytes == 4 {
        value.to_ne_bytes().to_vec()
    } else {
        half::f16::from_f32(value).to_ne_bytes().to_vec()
    };
    let count = spec
        .shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .filter(|n| *n <= 512 * 1024 * 1024 / element_bytes)
        .ok_or_else(|| bad("logical tensor size exceeds bound"))?;
    for flat in 0..count {
        let (mut n, mut offset) = (flat, 0usize);
        for axis in (0..spec.shape.len()).rev() {
            offset += (n % spec.shape[axis]) * spec.strides[axis];
            n /= spec.shape[axis];
        }
        let offset = offset
            .checked_mul(element_bytes)
            .ok_or_else(|| bad("tensor byte offset overflow"))?;
        bytes
            .get_mut(offset..offset + element_bytes)
            .ok_or_else(|| bad("tensor fill exceeds allocation"))?
            .copy_from_slice(&encoded);
    }
    Ok(())
}
impl Buffer {
    pub fn new(api: &Rc<Api>, spec: Spec) -> Result<Rc<Self>> {
        let (code, element_bytes) = match spec.dtype.as_str() {
            "float16" => (0, 2),
            "float32" => (1, 4),
            _ => return Err(bad("unsupported vision tensor dtype")),
        };
        if spec.dynamic
            || spec.shape.is_empty()
            || spec.shape.len() > 8
            || spec.shape.len() != spec.strides.len()
            || spec.shape.contains(&0)
            || spec.strides.contains(&0)
        {
            return Err(bad("invalid static tensor layout"));
        }
        let span = spec
            .shape
            .iter()
            .zip(&spec.strides)
            .try_fold(1usize, |sum, (n, s)| {
                sum.checked_add((n - 1).checked_mul(*s)?)
            })
            .ok_or_else(|| bad("tensor layout overflow"))?;
        let needed = span
            .checked_mul(element_bytes)
            .filter(|v| *v <= 512 * 1024 * 1024)
            .ok_or_else(|| bad("tensor exceeds worker bound"))?;
        if needed > spec.nbytes {
            return Err(bad("tensor descriptor byte count too small"));
        }
        let shape = spec.shape.iter().map(|n| *n as i64).collect::<Vec<_>>();
        let strides = spec.strides.iter().map(|n| *n as i64).collect::<Vec<_>>();
        let mut err = [0; 4096];
        let p = unsafe {
            (api.create)(
                code,
                shape.len() as i32,
                shape.as_ptr(),
                strides.as_ptr(),
                err.as_mut_ptr(),
                4096,
            )
        };
        let handle = api.handle(p, &err)?;
        let address = unsafe { (api.address)(handle.ptr) } as *mut u8;
        let nbytes = unsafe { (api.nbytes)(handle.ptr) };
        if address.is_null() || nbytes < needed as i64 || nbytes > 1024 * 1024 * 1024 {
            return Err(bad("invalid IOSurface allocation"));
        }
        Ok(Rc::new(Self {
            handle,
            spec,
            address,
            nbytes: nbytes as usize,
            element_bytes,
        }))
    }
    fn offset(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.spec.shape.len()
            || index.iter().zip(&self.spec.shape).any(|(i, n)| i >= n)
        {
            return Err(bad("tensor index out of bounds"));
        }
        let offset = index
            .iter()
            .zip(&self.spec.strides)
            .map(|(i, s)| i * s)
            .sum::<usize>()
            * self.element_bytes;
        if offset + self.element_bytes > self.nbytes {
            return Err(bad("IOSurface access exceeds allocation"));
        }
        Ok(offset)
    }
    pub fn set(&self, index: &[usize], value: f32) -> Result<()> {
        let offset = self.offset(index)?;
        // SAFETY: allocation is live, IOSurface-aligned, offset checked above.
        unsafe {
            if self.element_bytes == 4 {
                (self.address.add(offset) as *mut f32).write(value)
            } else {
                (self.address.add(offset) as *mut u16).write(half::f16::from_f32(value).to_bits())
            }
        };
        Ok(())
    }
    pub fn get(&self, index: &[usize]) -> Result<f32> {
        let offset = self.offset(index)?;
        Ok(unsafe {
            if self.element_bytes == 4 {
                (self.address.add(offset) as *const f32).read()
            } else {
                half::f16::from_bits((self.address.add(offset) as *const u16).read()).to_f32()
            }
        })
    }
    pub fn fill(&self, value: f32) -> Result<()> {
        if value == 0. {
            unsafe { ptr::write_bytes(self.address, 0, self.nbytes) };
            return Ok(());
        }
        // The blocking CoreAI call has finished before this buffer is reused;
        // its verified allocation is exclusively written during preparation.
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.address, self.nbytes) };
        fill_bytes(&self.spec, bytes, self.element_bytes, value)
    }
}

impl FnHandle {
    pub fn bind_run(&self, inputs: &[(String, Rc<Buffer>)], output: Rc<Buffer>) -> Result<f64> {
        let names = inputs
            .iter()
            .map(|(n, _)| cstring(n))
            .collect::<Result<Vec<_>>>()?;
        let names = names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        let handles = inputs.iter().map(|(_, b)| b.handle.ptr).collect::<Vec<_>>();
        let outname = cstring("output")?;
        let mut err = [0; 4096];
        let api = &self.handle.api;
        let ptr = unsafe {
            (api.bind)(
                self.handle.ptr,
                inputs.len() as i32,
                names.as_ptr(),
                handles.as_ptr(),
                1,
                &outname.as_ptr(),
                &output.handle.ptr,
                0,
                ptr::null(),
                ptr::null(),
                err.as_mut_ptr(),
                4096,
            )
        };
        let binding = api.handle(ptr, &err)?;
        let mut ms = 0.;
        let code = unsafe { (api.run)(1, &binding.ptr, &mut ms, err.as_mut_ptr(), 4096) };
        if code != 0 {
            return Err(error(&err));
        }
        Ok(ms / 1000.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padded_key_mask_keeps_non_tensor_padding_zero() {
        let spec = Spec {
            name: "key_mask".into(),
            dtype: "float16".into(),
            shape: vec![1, 4096, 1, 1],
            strides: vec![131072, 32, 32, 1],
            nbytes: 262144,
            dynamic: false,
        };
        let mut storage = vec![0; spec.nbytes];
        fill_bytes(&spec, &mut storage, 2, -10000.).unwrap();
        let expected = half::f16::from_f32(-10000.).to_ne_bytes();
        for row in storage.as_chunks::<64>().0 {
            assert_eq!(&row[..2], &expected);
            assert!(
                row[2..].iter().all(|v| *v == 0),
                "non-tensor padding was overwritten"
            );
        }
    }
}
