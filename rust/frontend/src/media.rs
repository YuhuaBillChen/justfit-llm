//! Native image preprocessing and Qwen template image-span expansion.
//! Bicubic coefficients use a normalized, signed 22-bit separable filter,
//! matching the existing Pillow RGB preparation contract (including pass clips).
use crate::{bad, generation::Prepared, Frontend, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::Cursor, io::Read};

pub const MAX_PIXELS: usize = 4194304;
pub struct Image {
    pub h: usize,
    pub w: usize,
    pub pixels: Vec<u8>,
    pub lo: u64,
    pub hi: u64,
}
impl Image {
    pub fn tokens(&self) -> usize {
        self.h * self.w / 4
    }
}
pub fn geometry(h: usize, w: usize, max: usize) -> Result<(usize, usize)> {
    if h == 0
        || w == 0
        || !(65536..=MAX_PIXELS).contains(&max)
        || h.max(w) as f64 / h.min(w) as f64 > 200.
    {
        return Err(bad("invalid image dimensions or pixel budget"));
    }
    let (mut oh, mut ow) = (
        ((h as f64 / 32.).round_ties_even() as usize) * 32,
        ((w as f64 / 32.).round_ties_even() as usize) * 32,
    );
    if oh * ow > max {
        let ratio = ((h * w) as f64 / max as f64).sqrt();
        oh = ((h as f64 / ratio / 32.).floor() as usize * 32).max(32);
        ow = ((w as f64 / ratio / 32.).floor() as usize * 32).max(32);
    } else if oh * ow < 65536 {
        let ratio = (65536. / (h * w) as f64).sqrt();
        oh = (h as f64 * ratio / 32.).ceil() as usize * 32;
        ow = (w as f64 * ratio / 32.).ceil() as usize * 32;
    }
    while oh * ow > max {
        if oh >= ow && oh > 32 {
            oh -= 32
        } else {
            ow -= 32
        }
    }
    Ok((oh, ow))
}
fn cubic(mut x: f64) -> f64 {
    x = x.abs();
    if x < 1. {
        ((1.5 * x - 2.5) * x) * x + 1.
    } else if x < 2. {
        (((x - 5.) * x + 8.) * x - 4.) * (-0.5)
    } else {
        0.
    }
}
fn weights(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter = scale.max(1.);
    let inverse_filter=1./filter;
    let radius = filter * 2.;
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let first = ((center - radius + 0.5) as isize).max(0) as usize;
            let end = ((center + radius + 0.5) as usize).min(input);
            let mut taps = (first..end)
                .map(|p| cubic((p as f64 - center + 0.5) * inverse_filter))
                .collect::<Vec<_>>();
            let sum = taps.iter().sum::<f64>();
            for k in &mut taps {
                *k /= sum;
            }
            (
                first,
                taps.into_iter()
                    .map(|v| (v * (1u64 << 22) as f64).round() as i64)
                    .collect(),
            )
        })
        .collect()
}
pub fn resize(rgb: &[u8], h: usize, w: usize, oh: usize, ow: usize) -> Vec<u8> {
    // Pillow 13 selects the order by the reduction in each dimension. Each
    // pass clips to u8, so interchanging these passes changes actual pixels.
    let dh=h as i64-oh as i64;let dw=w as i64-ow as i64;
    if w!=ow && dh>0 && dh>dw*2 {
        let vertical=resize(rgb,h,w,oh,w);
        return resize(&vertical,oh,w,oh,ow);
    }

    let mut horizontal = Vec::new();
    let intermediate = if w == ow {
        rgb
    } else {
        horizontal.resize(h * ow * 3, 0);
        let taps = weights(w, ow);
        for y in 0..h {
            for (x, (start, weights)) in taps.iter().enumerate() {
                for c in 0..3 {
                    let value = weights.iter().enumerate().fold(1i64 << 21, |sum, (i, k)| {
                        sum + rgb[(y * w + start + i) * 3 + c] as i64 * k
                    });
                    horizontal[(y * ow + x) * 3 + c] = (value >> 22).clamp(0, 255) as u8;
                }
            }
        }
        &horizontal
    };
    if h == oh {
        return intermediate.to_vec();
    }
    let mut result = vec![0; oh * ow * 3];
    let taps = weights(h, oh);
    for (y, (start, weights)) in taps.iter().enumerate() {
        for x in 0..ow {
            for c in 0..3 {
                let value = weights.iter().enumerate().fold(1i64 << 21, |sum, (i, k)| {
                    sum + intermediate[((start + i) * ow + x) * 3 + c] as i64 * k
                });
                result[(y * ow + x) * 3 + c] = (value >> 22).clamp(0, 255) as u8;
            }
        }
    }
    result
}
pub fn prepare(raw: &[u8], max: usize) -> Result<Image> {
    if raw.len() > 32 * 1024 * 1024 {
        return Err(bad("image exceeds the size limit"));
    }
    let mut reader = ImageReader::new(Cursor::new(raw))
        .with_guessed_format()
        .map_err(|_| bad("image could not be decoded"))?;
    if !matches!(
        reader.format(),
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif)
    ) {
        return Err(bad("unsupported image format"));
    }
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(384 * 1024 * 1024);
    reader.limits(limits);
    let is_jpeg = reader.format() == Some(ImageFormat::Jpeg);
    let mut decoder = reader
        .into_decoder()
        .map_err(|_| bad("image could not be decoded"))?;
    let (w, h) = decoder.dimensions();
    let (w, h) = (w as usize, h as usize);
    if w * h > 32 * 1024 * 1024 {
        return Err(bad("source image exceeds the pixel limit"));
    }
    let (mut oh, mut ow) = geometry(h, w, max)?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut decoded = if is_jpeg {
        let mut jpeg =
            turbojpeg::Decompressor::new().map_err(|_| bad("JPEG decoder unavailable"))?;
        let header = jpeg
            .read_header(raw)
            .map_err(|_| bad("image could not be decoded"))?;
        if (header.width, header.height) != (w, h) {
            return Err(bad("JPEG dimensions changed"));
        }
        jpeg.set_scan_limit(100)
            .map_err(|_| bad("invalid JPEG limits"))?;
        let mut output = turbojpeg::Image {
            pixels: vec![0; w * h * 3],
            width: w,
            pitch: w * 3,
            height: h,
            format: turbojpeg::PixelFormat::RGB,
        };
        jpeg.decompress(raw, output.as_deref_mut())
            .map_err(|_| bad("image could not be decoded"))?;
        DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(w as u32, h as u32, output.pixels)
                .ok_or_else(|| bad("invalid JPEG pixels"))?,
        )
    } else {
        DynamicImage::from_decoder(decoder).map_err(|_| bad("image could not be decoded"))?
    };
    decoded.apply_orientation(orientation);
    let (dh, dw) = (decoded.height() as usize, decoded.width() as usize);
    if (dw, dh) != (w, h) {
        std::mem::swap(&mut oh, &mut ow)
    }
    // Pillow's I;16 -> RGB conversion clips integer values, rather than
    // scaling the full 16-bit range to 8 bits.
    let rgba = if let DynamicImage::ImageLuma16(ref gray) = decoded {
        image::RgbaImage::from_fn(gray.width(), gray.height(), |x, y| {
            let v = gray.get_pixel(x, y).0[0].min(255) as u8;
            image::Rgba([v, v, v, 255])
        })
    } else {
        decoded.to_rgba8()
    };
    let mut rgb = Vec::with_capacity(dw * dh * 3);
    for p in rgba.as_raw().as_chunks::<4>().0 {
        let a = p[3] as u32;
        for &v in &p[..3] {
            rgb.push(((v as u32 * a + 255 * (255 - a) + 127) / 255) as u8)
        }
    }
    let pixels = resize(&rgb, dh, dw, oh, ow);
    let (h, w) = (oh / 16, ow / 16);
    let mut hash = Sha256::new();
    hash.update((h as u32).to_le_bytes());
    hash.update((w as u32).to_le_bytes());
    hash.update(&pixels);
    let hash = hash.finalize();
    let lo = u64::from_le_bytes(hash[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(hash[8..16].try_into().unwrap());
    Ok(Image {
        h,
        w,
        pixels,
        lo,
        hi,
    })
}
pub fn data_url(url: &Value) -> Result<Vec<u8>> {
    let url = url
        .as_str()
        .ok_or_else(|| bad("invalid image content part"))?;
    let (head, data) = url
        .split_once(',')
        .ok_or_else(|| bad("image URL must be base64 data"))?;
    if !head.starts_with("data:") || !head.contains(";base64") {
        return Err(bad("only base64 data: image URLs are supported"));
    }
    if data.len() > 32 * 1024 * 1024 * 4 / 3 + 4 {
        return Err(bad("image exceeds the size limit"));
    }
    STANDARD
        .decode(data)
        .map_err(|_| bad("image data URL is not valid base64"))
}

enum MediaResult {
    Count(usize),
    Rendered(String),
    Generation(Box<Prepared>),
}

impl Frontend {
    pub fn media_generation(&self, body: &Value, id: u64) -> Result<Prepared> {
        match self.media_prepare(body, id, false, false, &|| Ok(()))? {
            MediaResult::Generation(prepared) => Ok(*prepared),
            _ => unreachable!(),
        }
    }
    /// Counts expanded image placeholders without inference, ANE leases or a
    /// generation budget/context check, matching the prompt-only API contract.
    pub fn media_token_count(&self, body: &Value) -> Result<usize> {
        match self.media_prepare(body, 0, true, false, &|| Ok(()))? {
            MediaResult::Count(tokens) => Ok(tokens),
            _ => unreachable!(),
        }
    }
    pub fn media_generation_checked(&self,body:&Value,id:u64,check:&dyn Fn()->Result<()>)->Result<Prepared> {
        match self.media_prepare(body,id,false,false,check)? {MediaResult::Generation(p)=>Ok(*p),_=>unreachable!()}
    }
    pub fn media_token_count_checked(&self,body:&Value,check:&dyn Fn()->Result<()>)->Result<usize> {
        match self.media_prepare(body,0,true,false,check)? {MediaResult::Count(n)=>Ok(n),_=>unreachable!()}
    }
    pub fn media_render_checked(&self,body:&Value,check:&dyn Fn()->Result<()>)->Result<String>{
        match self.media_prepare(body,0,true,true,check)?{MediaResult::Rendered(text)=>Ok(text),_=>unreachable!()}
    }
    fn media_prepare(&self, body: &Value, id: u64, count_only: bool, render_only:bool, check:&dyn Fn()->Result<()>) -> Result<MediaResult> {
        self.check_model(body)?;
        let expanded = self.documents.expand(body,check)?;
        let body = &expanded;
        let mut stripped = body.clone();
        let mut replacements = Vec::new();
        let mut images = Vec::new();
        let mut entropy = [0u8; 16];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut entropy))
            .map_err(|_| bad("request entropy unavailable"))?;
        let nonce = entropy
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let ms = stripped["messages"]
            .as_array_mut()
            .ok_or_else(|| bad("messages must be an array"))?;
        let mut total = 0;
        for message in ms {
            check()?;
            let Some(parts) = message["content"].as_array() else {
                continue;
            };
            let mut text = String::new();
            let mut canonical = Vec::new();
            let mut has_image = false;
            for part in parts {
                match part["type"].as_str() {
                    Some("text" | "input_text") => {
                        let t = part
                            .get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| bad("text must be a string"))?;
                        text.push_str(t);
                        canonical.push(json!({"type":"text","text":t}));
                    }
                    Some("image_url") => {
                        if !matches!(message["role"].as_str(), Some("user" | "tool")) {
                            return Err(bad("only user/tool messages support images"));
                        }
                        if images.len() >= 64 {
                            return Err(bad("requests support at most 64 images"));
                        }
                        let url = if part["image_url"].is_object() {
                            &part["image_url"]["url"]
                        } else {
                            &part["image_url"]
                        };
                        let image = self.images.prepare(&data_url(url)?, self.preparation.max_image_pixels)?;
                        total += image.pixels.len();
                        if total > 256 * 1024 * 1024 {
                            return Err(crate::ApiError(
                                503,
                                "in-flight image memory budget is full".into(),
                                "frontend_overloaded".into(),
                            ));
                        }
                        images.push(image);
                        canonical.push(json!({"type":"image_url","image_url":{"url":"prepared"}}));
                        has_image = true;
                    }
                    Some("file") => return Err(crate::unsupported("native PDF rendering")),
                    _ => return Err(bad("unsupported message content part")),
                }
            }
            if has_image {
                let marker = format!("\0justfit-media-{nonce}-{}\0", replacements.len());
                replacements.push((marker.clone(), json!(canonical)));
                message["content"] = json!(marker)
            } else {
                message["content"] = json!(text)
            }
        }
        let mut normalized = crate::messages(&stripped)?;
        for message in normalized.as_array_mut().unwrap() {
            if let Some(content) = message["content"].as_str() {
                if let Some((_, parts)) = replacements.iter().find(|(marker, _)| marker == content)
                {
                    message["content"] = parts.clone();
                }
            }
        }
        let marker = format!("\0justfit-pad-{nonce}\0");
        let selected=self.chat_templates.select(body);
        crate::chat_templates::accepts(selected,&normalized)?;
        let template = selected.source.replace("<|image_pad|>", &marker);
        let mut request = body.clone();
        let add_generation=if render_only {body.get("add_generation_prompt").map_or(Ok(true),|value|value.as_bool().ok_or_else(||bad("add_generation_prompt must be a boolean")))?}else{true};
        request["add_generation_prompt"] = json!(add_generation);
        let marked = self.render_messages(&request, normalized.clone(), &template)?;
        request["add_generation_prompt"] = json!(false);
        let history = self.render_messages(&request, normalized, &template)?;
        let suffix = marked
            .strip_prefix(&history)
            .ok_or_else(|| bad("chat template must end with an assistant generation prefix"))?;
        let thinking = suffix.rfind("<think>") > suffix.rfind("</think>");
        if add_generation&&suffix.is_empty(){return Err(bad("chat template must end with an assistant generation prefix"));}
        let generation = self.encode(suffix, false)?.len() as u32;
        let mut positions = Vec::new();
        let mut prompt = String::new();
        let parts = marked.split(&marker).collect::<Vec<_>>();
        for (i, part) in parts.iter().enumerate() {
            prompt.push_str(part);
            if i + 1 < parts.len() {
                positions.push((prompt.len(), prompt.len() + "<|image_pad|>".len()));
                prompt.push_str("<|image_pad|>");
            }
        }
        let encoded = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| bad("image prompt could not be tokenized"))?;
        let offsets = encoded.get_offsets();
        let slots = offsets
            .iter()
            .enumerate()
            .filter_map(|(i, p)| positions.contains(p).then_some(i))
            .collect::<Vec<_>>();
        if slots.len() != images.len() {
            return Err(bad("image count does not match rendered template"));
        }
        let pad = self
            .tokenizer
            .token_to_id("<|image_pad|>")
            .ok_or_else(|| bad("image pad token unavailable"))?;
        let tokens = encoded.get_ids();
        let expanded_len = tokens.len() + images.iter().map(|i| i.tokens() - 1).sum::<usize>();
        if render_only{return Ok(MediaResult::Rendered(prompt));}
        if count_only {
            return Ok(MediaResult::Count(expanded_len));
        }
        if expanded_len >= self.context as usize {
            return Err(crate::ApiError(
                400,
                "prompt exceeds the context window".into(),
                "context_length_exceeded".into(),
            ));
        }
        if 94 + expanded_len * 4 + images.len() * 32 + total > 256 * 1024 * 1024 {
            return Err(bad("images exceed request size limit"));
        }
        let mut expanded = Vec::with_capacity(expanded_len);
        let mut spans = Vec::new();
        let mut cursor = 0;
        for (position, image) in slots.into_iter().zip(&images) {
            expanded.extend(&tokens[cursor..position]);
            spans.push([
                expanded.len() as u32,
                image.tokens() as u32,
                image.h as u32,
                image.w as u32,
            ]);
            expanded.resize(expanded.len() + image.tokens(), pad);
            cursor = position + 1;
        }
        expanded.extend(&tokens[cursor..]);
        let mut prepared = self.generation_media(body, id, expanded, thinking, generation)?;
        prepared.frame.payload[34..38].copy_from_slice(&(images.len() as u32).to_le_bytes());
        for (span, image) in spans.iter().zip(&images) {
            for v in span {
                prepared.frame.payload.extend(v.to_le_bytes())
            }
            prepared.frame.payload.extend(image.lo.to_le_bytes());
            prepared.frame.payload.extend(image.hi.to_le_bytes());
        }
        for image in images {
            prepared.frame.payload.extend_from_slice(&image.pixels)
        }
        Ok(MediaResult::Generation(Box::new(prepared)))
    }
}

#[cfg(test)]
mod resize_precision_tests{
    use super::*;
    #[test]
    fn asymmetric_resize_preserves_pillow13_pass_order(){
        let rgb=(0..257*319*3).map(|i|((i*17+i/13)%256)as u8).collect::<Vec<_>>();
        let pixels=resize(&rgb,257,319,256,320);
        assert_eq!(format!("{:x}",Sha256::digest(pixels)),"0d1b2255bcefa049ce23899844ebdd17e1a5f502413e3e9060072fef33c9d845");
    }
}
