//! Byte-exact nonwaiting input reservations, independent of request slots.
use crate::{ApiError,Result};
use serde_json::{json,Value};
use std::sync::{Arc,Mutex};
pub(super) struct Budget {capacity:usize,active:Mutex<usize>}
pub(super) struct Permit {budget:Arc<Budget>,bytes:usize}
pub(super) fn capacity(limit:usize)->Result<usize> {
    if limit==0{return Err(crate::bad("max_request_bytes must be a positive integer"))}
    limit.checked_mul(2).map(|n|n.max(512*1024*1024)).ok_or_else(||crate::bad("max_request_bytes exceeds the platform input-budget limit"))
}
impl Budget {
    pub(super) fn new(capacity:usize)->Self {Self{capacity,active:Mutex::new(0)}}
    pub(super) fn reserve(self:&Arc<Self>,bytes:usize)->Result<Permit> {
        let mut active=self.active.lock().unwrap();
        if bytes>self.capacity-*active {return Err(ApiError(503,"request body capacity is exhausted; retry shortly".into(),"frontend_overloaded".into()))}
        *active+=bytes;Ok(Permit{budget:self.clone(),bytes})
    }
    pub(super) fn stats(&self)->Value{json!({"active":*self.active.lock().unwrap(),"capacity":self.capacity})}
}
impl Drop for Permit {fn drop(&mut self){*self.budget.active.lock().unwrap()-=self.bytes}}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_byte_budget_reserves_sub_kib_and_returns_after_drop() {
        let budget=Arc::new(Budget::new(1025));
        let first=budget.reserve(1024).unwrap();let last=budget.reserve(1).unwrap();
        assert!(budget.reserve(1).is_err());assert_eq!(budget.stats(),json!({"active":1025,"capacity":1025}));
        drop(first);let more=budget.reserve(1024).unwrap();drop((last,more));assert_eq!(budget.stats()["active"],0);
    }
    #[test]
    fn larger_limits_do_not_wrap_u32_or_platform_size() {
        assert_eq!(capacity(256*1024*1024).unwrap(),512*1024*1024);
        let budget=Arc::new(Budget::new(8*1024*1024*1024));
        let reservation=budget.reserve(5*1024*1024*1024).unwrap();assert_eq!(budget.stats()["active"],5u64*1024*1024*1024);drop(reservation);
        assert!(capacity(0).is_err());assert!(capacity(usize::MAX).is_err());
    }
}

impl Permit{
    pub(super) fn resize(&mut self,bytes:usize)->Result<()>{
        let mut active=self.budget.active.lock().unwrap();let others=*active-self.bytes;
        if bytes>self.budget.capacity-others{return Err(ApiError(503,"retained input capacity is exhausted; retry shortly".into(),"frontend_overloaded".into()))}
        *active=others+bytes;self.bytes=bytes;Ok(())
    }
    pub(super) fn absorb(&mut self,other:&mut Self){assert!(Arc::ptr_eq(&self.budget,&other.budget));self.bytes+=other.bytes;other.bytes=0;}
    pub(super) fn retain(&mut self,prepared:&crate::generation::Prepared,format:Option<&Value>,history:&[Value])->Result<()>{
        let mut sizes=Vec::with_capacity(5);
        fn size<T:serde::Serialize+?Sized>(value:&T)->Result<usize>{crate::python_json::compact_size(value).map_err(|_|crate::bad("retained input could not be encoded"))}
        if !history.is_empty(){sizes.push(size(history)?)}
        if let Some(format)=format.filter(|&v|!v.is_null()&&v!=&json!({})&&v!=&json!([])&&v!=&json!(false)&&v!=&json!("")){sizes.push(size(format)?)}
        if let Some(policy)=&prepared.tools{
            if !policy.schemas.is_empty(){sizes.push(size(&policy.schemas)?)}
            if policy.namespaces.as_object().is_some_and(|m|!m.is_empty())||policy.namespaces.as_array().is_some_and(|a|!a.is_empty()){sizes.push(size(&policy.namespaces)?)}
        }
        if !prepared.stop.is_empty(){sizes.push(size(&prepared.stop)?)}
        let bytes=if sizes.is_empty(){0}else{sizes.iter().try_fold(1+sizes.len(),|total,size|total.checked_add(*size)).ok_or_else(||crate::bad("retained input size overflow"))?};
        self.resize(bytes)
    }
}
#[cfg(test)]
mod lifetime_tests{
    use super::*;
    #[test]
    fn resizing_is_atomic_and_merged_history_has_only_one_release(){
        let budget=Arc::new(Budget::new(100));let mut input=budget.reserve(60).unwrap();let mut history=budget.reserve(30).unwrap();
        input.absorb(&mut history);drop(history);assert_eq!(budget.stats()["active"],90);
        assert!(input.resize(101).is_err());assert_eq!(budget.stats()["active"],90);
        input.resize(7).unwrap();let extra=budget.reserve(93).unwrap();assert!(input.resize(8).is_err());assert_eq!(budget.stats()["active"],100);
        drop(extra);input.resize(0).unwrap();assert_eq!(budget.stats()["active"],0);drop(input);assert_eq!(budget.stats()["active"],0);
    }
}
