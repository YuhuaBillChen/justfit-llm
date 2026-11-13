//! Shared front/media preparation options. Validate before launching native.
use crate::{bad,Result};
use serde::{Deserialize,Serialize};
#[derive(Clone,Deserialize,Serialize)]
pub struct Options {
    #[serde(default)] pub default_reasoning_effort:Option<String>,
    #[serde(default="pixels")] pub max_image_pixels:usize,
}
fn pixels()->usize {crate::media::MAX_PIXELS}
impl Default for Options {fn default()->Self{Self{default_reasoning_effort:None,max_image_pixels:pixels()}}}
impl Options {
    pub fn validate(&self)->Result<()> {
        if self.default_reasoning_effort.as_deref().is_some_and(|v|!["none","minimal","low","medium","high","xhigh","max"].contains(&v)){return Err(bad("invalid default_reasoning_effort"))}
        if !(65536..=crate::media::MAX_PIXELS).contains(&self.max_image_pixels){return Err(bad("max_image_pixels must be between 65536 and 4194304 pixels"))}
        Ok(())
    }
    pub fn environment(mut self)->Result<Self> {
        if self.default_reasoning_effort.is_none() {
            self.default_reasoning_effort=match std::env::var("JUSTFIT_DEFAULT_REASONING_EFFORT") {Ok(v)=>Some(v),Err(std::env::VarError::NotPresent)=>None,Err(_)=>return Err(bad("invalid default_reasoning_effort"))};
        }
        self.validate()?;Ok(self)
    }
}
