//! Python json.dumps(ensure_ascii=False) spelling for prompt payloads.
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
fn float(text: &str) -> String {
    let value = text.parse::<f64>().expect("validated finite JSON float");
    let repr = format!("{value:?}");
    if let Some((mantissa, exponent)) = repr.split_once('e') {
        let exponent = exponent.parse::<i32>().unwrap();
        format!(
            "{mantissa}e{}{num:02}",
            if exponent < 0 { "-" } else { "+" },
            num = exponent.unsigned_abs()
        )
    } else {
        repr
    }
}
struct Formatter;
impl serde_json::ser::Formatter for Formatter {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
    fn write_number_str<W: ?Sized + Write>(
        &mut self,
        w: &mut W,
        text: &str,
    ) -> std::io::Result<()> {
        if text.contains(['.', 'e', 'E']) {
            w.write_all(float(text).as_bytes())
        } else if text == "-0" {
            w.write_all(b"0")
        } else {
            w.write_all(text.as_bytes())
        }
    }
}
pub fn dumps(value: &Value) -> String {
    let mut bytes = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes, Formatter,
        ))
        .expect("finite JSON values serialize");
    String::from_utf8(bytes).expect("JSON output is UTF8")
}

// Compact API JSON uses the same numeric spellings, with no spaces, and
// escapes the separators recognized by SSE readers. Count without allocating
// a second copy of retained schemas or a Responses conversation.
struct Compact;
impl serde_json::ser::Formatter for Compact{
    fn write_number_str<W:?Sized+Write>(&mut self,w:&mut W,text:&str)->std::io::Result<()>{Formatter.write_number_str(w,text)}
}
#[derive(Default)]
struct Counter{bytes:usize,last:[u8;2]}
impl Write for Counter{
    fn write(&mut self,bytes:&[u8])->std::io::Result<usize>{
        for &byte in bytes{
            self.bytes=self.bytes.checked_add(1).ok_or_else(||std::io::Error::other("JSON size overflow"))?;
            let extra=if self.last[1]==0xc2&&byte==0x85{4}else if self.last==[0xe2,0x80]&&(byte==0xa8||byte==0xa9){3}else{0};
            self.bytes=self.bytes.checked_add(extra).ok_or_else(||std::io::Error::other("JSON size overflow"))?;self.last=[self.last[1],byte];
        }Ok(bytes.len())
    }
    fn flush(&mut self)->std::io::Result<()>{Ok(())}
}
pub fn compact_size<T:Serialize+?Sized>(value:&T)->std::result::Result<usize,serde_json::Error>{
    let mut count=Counter::default();value.serialize(&mut serde_json::Serializer::with_formatter(&mut count,Compact))?;Ok(count.bytes)
}
