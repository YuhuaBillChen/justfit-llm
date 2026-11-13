//! Bounded GGUF v3 metadata reader; weight payloads are never mapped or read.
use crate::download::Result;
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Read, os::unix::fs::OpenOptionsExt, path::Path};
const MAX_BYTES: u64 = 128 * 1024 * 1024;
const MAX_FIELDS: u64 = 16384;
const MAX_ITEMS: u64 = 1_000_000;
#[derive(Debug)]
pub struct Tensor {
    pub kind: u32,
    pub extents: Vec<u64>,
    pub offset: u64,
}
#[derive(Debug)]
pub struct Metadata {
    pub values: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, Tensor>,
}
struct Reader<R> {
    input: R,
    consumed: u64,
    items: u64,
}
impl<R: Read> Reader<R> {
    fn bytes(&mut self, size: u64) -> Result<Vec<u8>> {
        if size > MAX_BYTES - self.consumed {
            return Err("GGUF metadata exceeds the size limit".into());
        }
        let mut bytes = vec![0; usize::try_from(size)?];
        self.input
            .read_exact(&mut bytes)
            .map_err(|_| "truncated GGUF metadata")?;
        self.consumed += size;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let size = self.u64()?;
        Ok(String::from_utf8(self.bytes(size)?).map_err(|_| "invalid UTF-8 in GGUF metadata")?)
    }
    fn value(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => json!(self.bytes(1)?[0]),
            1 => json!(self.bytes(1)?[0] as i8),
            2 => json!(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            3 => json!(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            4 => json!(self.u32()?),
            5 => json!(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap())),
            6 => json!(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()) as f64),
            7 => json!(self.bytes(1)?[0] != 0),
            8 => json!(self.string()?),
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                if element > 12
                    || element == 9
                    || count > MAX_ITEMS
                    || self.items.checked_add(count).is_none_or(|n| n > MAX_ITEMS)
                {
                    return Err("unsupported or oversized GGUF metadata array".into());
                }
                self.items += count;
                let mut values = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    values.push(self.value(element)?);
                }
                Value::Array(values)
            }
            10 => json!(self.u64()?),
            11 => json!(i64::from_le_bytes(self.bytes(8)?.try_into().unwrap())),
            12 => json!(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap())),
            _ => return Err("unknown GGUF metadata type".into()),
        })
    }
}
impl Metadata {
    pub fn open(path: &Path, tensors: bool) -> Result<Self> {
        let input = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)?;
        if !input.metadata()?.is_file() {
            return Err("GGUF input is not a regular file".into());
        }
        Self::read(input, tensors)
    }
    pub fn read(input: impl Read, tensors: bool) -> Result<Self> {
        let mut input = Reader {
            input,
            consumed: 0,
            items: 0,
        };
        if input.bytes(4)? != b"GGUF" || input.u32()? != 3 {
            return Err("unsupported GGUF header (expected little-endian v3)".into());
        }
        let tensor_count = input.u64()?;
        let keys = input.u64()?;
        if keys > MAX_FIELDS || tensor_count > MAX_FIELDS {
            return Err("GGUF metadata has too many fields".into());
        }
        let mut values = BTreeMap::new();
        for _ in 0..keys {
            let key = input.string()?;
            let kind = input.u32()?;
            let value = input.value(kind)?;
            if values.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate GGUF metadata key: {key}").into());
            }
        }
        let mut entries = BTreeMap::new();
        for _ in 0..if tensors { tensor_count } else { 0 } {
            let name = input.string()?;
            let dimensions = input.u32()?;
            if dimensions > 4 {
                return Err(format!("invalid GGUF tensor rank: {name}").into());
            }
            let mut extents = vec![];
            for _ in 0..dimensions {
                extents.push(input.u64()?);
            }
            let kind = input.u32()?;
            let offset = input.u64()?;
            if entries
                .insert(
                    name.clone(),
                    Tensor {
                        kind,
                        extents,
                        offset,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate GGUF tensor: {name}").into());
            }
        }
        Ok(Self {
            values,
            tensors: entries,
        })
    }
    pub fn positive(&self, key: &str) -> Result<u64> {
        self.values
            .get(key)
            .and_then(Value::as_u64)
            .filter(|v| *v > 0)
            .ok_or_else(|| format!("missing or invalid positive GGUF metadata: {key}").into())
    }
    pub fn string(&self, key: &str) -> Result<&str> {
        self.values
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing or invalid GGUF metadata: {key}").into())
    }
    pub fn scalar_metadata(&self) -> Value {
        let mut unsigned = serde_json::Map::new();
        let mut float = serde_json::Map::new();
        let mut string = serde_json::Map::new();
        for (key, value) in &self.values {
            match value {
                Value::Bool(v) => {
                    unsigned.insert(key.clone(), json!(u64::from(*v)));
                }
                Value::Number(v) => {
                    if let Some(v) = v.as_u64() {
                        unsigned.insert(key.clone(), json!(v));
                    } else if let Some(v) = v.as_i64() {
                        unsigned.insert(key.clone(), json!(v as u64));
                    } else if v.as_f64().is_some_and(|v| v.is_finite()) {
                        float.insert(key.clone(), value.clone());
                    }
                }
                Value::String(_) => {
                    string.insert(key.clone(), value.clone());
                }
                _ => (),
            }
        }
        json!({"unsigned":unsigned,"float":float,"string":string})
    }
    pub fn model_config(&self) -> Result<Value> {
        let arch = self.string("general.architecture")?;
        let model_type = match arch {
            "qwen35" => "qwen3_5_text",
            "qwen35moe" => "qwen3_5_moe_text",
            _ => return Err(format!("unsupported GGUF model architecture: {arch}").into()),
        };
        let fields = [
            ("hidden_size", "embedding_length"),
            ("max_position_embeddings", "context_length"),
            ("num_attention_heads", "attention.head_count"),
            ("num_key_value_heads", "attention.head_count_kv"),
            ("head_dim", "attention.key_length"),
        ];
        let mut text = serde_json::Map::new();
        for (name, key) in fields {
            text.insert(name.into(), json!(self.positive(&format!("{arch}.{key}"))?));
        }
        let layers = self.positive(&format!("{arch}.block_count"))?;
        let mtp = self
            .values
            .get(&format!("{arch}.nextn_predict_layers"))
            .map(Value::as_u64)
            .unwrap_or(Some(0))
            .filter(|v| *v < layers)
            .ok_or("invalid GGUF MTP layer count")?;
        let tokens = self
            .values
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or("GGUF tokenizer tokens missing")?;
        text.insert("num_hidden_layers".into(), json!(layers - mtp));
        text.insert("vocab_size".into(), json!(tokens.len()));
        text.insert("model_type".into(), json!(model_type));
        if arch == "qwen35moe" {
            text.insert(
                "num_experts".into(),
                json!(self.positive(&format!("{arch}.expert_count"))?),
            );
        }
        Ok(json!({"model_type":model_type.strip_suffix("_text").unwrap(),"text_config":text}))
    }
    pub fn merge(headers: Vec<Self>) -> Result<Self> {
        let mut headers = headers.into_iter();
        let mut first = headers.next().ok_or("GGUF target has no files")?;
        let count = first
            .values
            .get("split.count")
            .map(Value::as_u64)
            .unwrap_or(Some(1))
            .ok_or("invalid split count")?;
        let mut actual = 1;
        if count > 1 && first.values.get("split.no").and_then(Value::as_u64) != Some(0) {
            return Err("GGUF first split is out of order".into());
        }
        for header in headers {
            if header.values.get("split.no").and_then(Value::as_u64) != Some(actual)
                || header.values.get("split.count").and_then(Value::as_u64) != Some(count)
            {
                return Err("GGUF split is out of order".into());
            }
            for (name, tensor) in header.tensors {
                if first.tensors.insert(name.clone(), tensor).is_some() {
                    return Err(format!("duplicate GGUF tensor across split files: {name}").into());
                }
            }
            actual += 1;
        }
        if actual != count
            || first
                .values
                .get("split.tensors.count")
                .is_some_and(|n| n.as_u64() != Some(first.tensors.len() as u64))
        {
            return Err("GGUF split files hold another number of tensors".into());
        }
        Ok(first)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(fields: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend((fields.len() as u64).to_le_bytes());
        for (key, kind, v) in fields {
            b.extend((key.len() as u64).to_le_bytes());
            b.extend(key.as_bytes());
            b.extend(kind.to_le_bytes());
            b.extend(v);
        }
        b
    }
    #[test]
    fn typed_scalar_values_keep_signed_boolean_float_and_string_distinct() {
        let mut text = 2u64.to_le_bytes().to_vec();
        text.extend(b"ok");
        let data = fixture(&[
            ("negative", 11, (-1i64).to_le_bytes().to_vec()),
            ("flag", 7, vec![1]),
            ("value", 6, 1.5f32.to_le_bytes().to_vec()),
            ("text", 8, text),
        ]);
        let m = Metadata::read(data.as_slice(), true).unwrap();
        assert_eq!(
            m.scalar_metadata(),
            json!({"unsigned":{"negative":u64::MAX,"flag":1},"float":{"value":1.5},"string":{"text":"ok"}})
        );
    }
    #[test]
    fn malformed_duplicate_truncated_and_oversized_metadata_is_refused() {
        let data = fixture(&[
            ("a", 4, 1u32.to_le_bytes().to_vec()),
            ("a", 4, 2u32.to_le_bytes().to_vec()),
        ]);
        assert!(Metadata::read(data.as_slice(), true).is_err());
        for length in 0..24 {
            assert!(Metadata::read(&data[..length], true).is_err());
        }
        let mut bad = 8u32.to_le_bytes().to_vec();
        bad.extend((MAX_BYTES + 1).to_le_bytes());
        let data = fixture(&[("bad", 9, bad)]);
        assert!(Metadata::read(data.as_slice(), true).is_err());
        let data = fixture(&[("bad", 8, (MAX_BYTES + 1).to_le_bytes().to_vec())]);
        assert!(Metadata::read(data.as_slice(), true).is_err());
    }
}

impl Metadata {
    pub fn vision_config(&self) -> Result<Value> {
        if self.string("general.architecture")? != "clip"
            || self.string("clip.projector_type")? != "qwen3vl_merger"
            || self.values.get("clip.use_gelu") != Some(&json!(true))
        {
            return Err("unsupported GGUF vision architecture".into());
        }
        let mut config = serde_json::Map::new();
        for (name, key) in [
            ("depth", "block_count"),
            ("hidden_size", "embedding_length"),
            ("num_heads", "attention.head_count"),
            ("intermediate_size", "feed_forward_length"),
            ("out_hidden_size", "projection_dim"),
            ("patch_size", "patch_size"),
            ("spatial_merge_size", "spatial_merge_size"),
        ] {
            config.insert(
                name.into(),
                json!(self.positive(&format!("clip.vision.{key}"))?),
            );
        }
        let image = self.positive("clip.vision.image_size")?;
        let patch = config["patch_size"].as_u64().unwrap();
        if image % patch != 0 {
            return Err("invalid GGUF vision position grid".into());
        }
        if self
            .values
            .get("clip.vision.is_deepstack_layers")
            .and_then(Value::as_array)
            .ok_or("GGUF vision deepstack flags missing")?
            .iter()
            .any(|v| v != &json!(false) && v != &json!(0))
        {
            return Err("GGUF vision deepstack layers are unsupported".into());
        }
        config.insert("temporal_patch_size".into(), json!(2));
        config.insert("in_channels".into(), json!(3));
        config.insert(
            "num_position_embeddings".into(),
            json!((image / patch)
                .checked_pow(2)
                .ok_or("vision position overflow")?),
        );
        config.insert("hidden_act".into(), json!("gelu_pytorch_tanh"));
        config.insert("deepstack_visual_indexes".into(), json!([]));
        Ok(Value::Object(config))
    }
    pub fn tokenizer_files(&self) -> Result<BTreeMap<String, Vec<u8>>> {
        if self.string("tokenizer.ggml.model")? != "gpt2"
            || self.string("tokenizer.ggml.pre")? != "qwen35"
        {
            return Err("unsupported GGUF tokenizer profile; expected gpt2/qwen35".into());
        }
        let tokens = self
            .values
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .filter(|v| !v.is_empty())
            .ok_or("invalid GGUF tokenizer vocabulary")?;
        let kinds = self
            .values
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .filter(|v| v.len() == tokens.len())
            .ok_or("invalid GGUF tokenizer token types")?;
        let merges = self
            .values
            .get("tokenizer.ggml.merges")
            .and_then(Value::as_array)
            .ok_or("GGUF BPE merges missing")?;
        let mut vocab = serde_json::Map::new();
        let mut added = vec![];
        let mut decoder = serde_json::Map::new();
        for (index, (token, kind)) in tokens.iter().zip(kinds).enumerate() {
            let token = token
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or("invalid GGUF tokenizer token")?;
            let kind = kind
                .as_u64()
                .filter(|v| matches!(v, 1 | 3 | 4 | 5))
                .ok_or("unsupported GGUF token type")?;
            if vocab.insert(token.into(), json!(index)).is_some() {
                return Err("duplicate GGUF tokenizer token".into());
            }
            if kind == 3 || kind == 4 {
                let value = json!({"content":token,"special":kind==3,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false});
                decoder.insert(index.to_string(), value.clone());
                let mut value = value;
                value["id"] = json!(index);
                added.push(value);
            }
        }
        // GPT-2 byte-to-Unicode alphabet, including the 68 displaced byte values.
        let direct = |b: u32| {
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
        };
        let mut extra = 0;
        for b in 0..256 {
            let code = if direct(b) {
                b
            } else {
                let n = 256 + extra;
                extra += 1;
                n
            };
            let token = char::from_u32(code).unwrap().to_string();
            if !vocab.contains_key(&token) {
                return Err("GGUF byte-level tokenizer is missing byte tokens".into());
            }
        }
        let mut pairs = vec![];
        for merge in merges {
            let merge = merge.as_str().ok_or("invalid GGUF BPE merge")?;
            let pair = merge.split(' ').collect::<Vec<_>>();
            if pair.len() != 2 {
                return Err("invalid GGUF BPE merge".into());
            }
            pairs.push(json!(pair));
        }
        for key in [
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token",
        ] {
            if self.values.get(key).is_some_and(|v| v != &json!(false)) {
                return Err(
                    format!("GGUF automatic BOS/EOS insertion is unsupported: {key}").into(),
                );
            }
        }
        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":added,"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":pattern},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":vocab,"merges":pairs}});
        let parsed = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&tokenizer)?)
            .map_err(|e| format!("invalid GGUF BPE vocabulary or merges: {e}"))?;
        let special = |name: &str, optional: bool| -> Result<Value> {
            let key = format!("tokenizer.ggml.{name}_token_id");
            if optional && !self.values.contains_key(&key) {
                return Ok(Value::Null);
            }
            let index = self
                .values
                .get(&key)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("invalid GGUF special token: {key}"))?
                as usize;
            if index >= tokens.len() || kinds[index] != 3 {
                return Err(format!("invalid GGUF special token: {key}").into());
            }
            Ok(tokens[index].clone())
        };
        let template = self.string("tokenizer.chat_template")?;
        if template.trim().is_empty() {
            return Err("GGUF chat template is empty".into());
        }
        let config = json!({"tokenizer_class":"TokenizersBackend","model_max_length":self.positive(&format!("{}.context_length",self.string("general.architecture")?))?,"clean_up_tokenization_spaces":false,"bos_token":special("bos",true)?,"eos_token":special("eos",false)?,"pad_token":special("padding",true)?,"added_tokens_decoder":decoder});
        Ok(BTreeMap::from([
            (
                "tokenizer/tokenizer.json".into(),
                parsed
                    .to_string(false)
                    .map_err(|e| e.to_string())?
                    .into_bytes(),
            ),
            (
                "tokenizer/tokenizer_config.json".into(),
                crate::source_mlx::json_bytes(config)?,
            ),
            (
                "tokenizer/chat_template.jinja".into(),
                template.as_bytes().to_vec(),
            ),
        ]))
    }
}

impl Metadata {
    pub fn require_loadable(&self) -> Result<()> {
        let arch = self.string("general.architecture")?;
        if !matches!(arch, "qwen35" | "qwen35moe") {
            return Err("unsupported GGUF target family".into());
        }
        let rotation = self
            .values
            .keys()
            .filter_map(|k| k.strip_prefix("prism.hadamard."))
            .collect::<Vec<_>>();
        if !rotation.is_empty() {
            let fixed = json!({"version":1,"block_size":1024,"transform":"normalized-sylvester-walsh-hadamard","axis":"input-last-dimension","sign_mode":"explicit","gdn_v_grouped":true});
            if arch != "qwen35"
                || rotation.iter().any(|k| {
                    !fixed.as_object().unwrap().contains_key(*k)
                        && ![
                            "weight_names",
                            "inverse_weight_names",
                            "sign_widths",
                            "sign_values",
                        ]
                        .contains(k)
                })
                || fixed
                    .as_object()
                    .unwrap()
                    .iter()
                    .any(|(k, v)| self.values.get(&format!("prism.hadamard.{k}")) != Some(v))
            {
                return Err("this GGUF input rotation is not one JustFit runs".into());
            }
        }
        let quant: &[u32] = &[
            2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40, 42, 142,
        ];
        let mut formats = quant.to_vec();
        formats.extend([1, 30]);
        let norms = &[0, 1, 30];
        let mut router = quant.to_vec();
        router.extend([0, 30]);
        let mut vectors = formats.clone();
        vectors.push(0);
        let mut required: BTreeMap<String, Vec<u32>> = BTreeMap::from([
            ("token_embd.weight".into(), formats.clone()),
            ("output_norm.weight".into(), norms.to_vec()),
            ("output.weight".into(), formats.clone()),
        ]);
        let config = self.model_config()?;
        let layers = config["text_config"]["num_hidden_layers"].as_u64().unwrap();
        let period = self.positive(&format!("{arch}.full_attention_interval"))?;
        for layer in 0..layers {
            let prefix = format!("blk.{layer}.");
            let mut add = |name: &str, kinds: &[u32]| {
                required.insert(format!("{prefix}{name}"), kinds.to_vec());
            };
            for name in ["attn_norm.weight", "post_attention_norm.weight"] {
                add(name, norms);
            }
            if (layer + 1) % period == 0 {
                for name in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ] {
                    add(name, &formats);
                }
                for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    add(name, norms);
                }
            } else {
                for name in ["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"] {
                    add(name, &formats);
                }
                for name in ["ssm_alpha.weight", "ssm_beta.weight"] {
                    add(name, &vectors);
                }
                for name in ["ssm_conv1d.weight", "ssm_a", "ssm_dt.bias"] {
                    add(name, &[0]);
                }
                add("ssm_norm.weight", norms);
            }
            if arch == "qwen35" {
                for name in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    add(name, &formats);
                }
            } else {
                add("ffn_gate_inp.weight", &router);
                add("ffn_gate_inp_shexp.weight", &[0, 30, 8]);
                for name in [
                    "ffn_gate_shexp.weight",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                    "ffn_down_exps.weight",
                ] {
                    add(name, &formats);
                }
                if self
                    .tensors
                    .contains_key(&format!("{prefix}ffn_gate_up_exps.weight"))
                {
                    add("ffn_gate_up_exps.weight", &formats);
                } else {
                    add("ffn_gate_exps.weight", &formats);
                    add("ffn_up_exps.weight", &formats);
                }
            }
        }
        let mut any_quantized = false;
        for (name, kinds) in &required {
            let tensor = self
                .tensors
                .get(name)
                .ok_or_else(|| format!("this GGUF is missing tensor {name}"))?;
            if !kinds.contains(&tensor.kind) {
                return Err(format!(
                    "this GGUF stores unsupported tensor type {} for {name}",
                    tensor.kind
                )
                .into());
            }
            if name != "token_embd.weight" && kinds == &formats && quant.contains(&tensor.kind) {
                any_quantized = true;
            }
            if let Some(stem) = name.strip_suffix(".weight") {
                if tensor.kind != 40 && self.tensors.contains_key(&format!("{stem}.scale")) {
                    return Err(format!("unsupported GGUF scale tensor beside {name}").into());
                }
            }
        }
        if !any_quantized {
            return Err(
                "this GGUF stores every projection unquantized; choose a quantized variant".into(),
            );
        }
        Ok(())
    }
}
#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    #[test]
    fn derived_byte_tokenizer_normalizes_and_keeps_special_ids() {
        let direct = |b: u32| {
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
        };
        let mut extra = 0;
        let mut tokens = vec![];
        for b in 0..256 {
            let code = if direct(b) {
                b
            } else {
                let n = 256 + extra;
                extra += 1;
                n
            };
            tokens.push(char::from_u32(code).unwrap().to_string());
        }
        tokens.push("ab".into());
        tokens.push("<eos>".into());
        let mut kinds = vec![1; tokens.len()];
        *kinds.last_mut().unwrap() = 3;
        let values = json!({"general.architecture":"qwen35","qwen35.context_length":4096,"tokenizer.ggml.model":"gpt2","tokenizer.ggml.pre":"qwen35","tokenizer.ggml.tokens":tokens,"tokenizer.ggml.token_type":kinds,"tokenizer.ggml.merges":["a b"],"tokenizer.ggml.eos_token_id":257,"tokenizer.chat_template":"{{ messages }}"});
        let metadata = Metadata {
            values: values
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            tensors: BTreeMap::new(),
        };
        let files = metadata.tokenizer_files().unwrap();
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(&files["tokenizer/tokenizer.json"]).unwrap();
        assert_eq!(tokenizer.encode("<eos>", false).unwrap().get_ids(), [257]);
        let encoded = tokenizer.encode("ab a\u{301} 中文\n", false).unwrap();
        assert_eq!(
            tokenizer.decode(encoded.get_ids(), true).unwrap(),
            "ab á 中文\n"
        );
        let config: Value =
            serde_json::from_slice(&files["tokenizer/tokenizer_config.json"]).unwrap();
        assert_eq!(config["eos_token"], "<eos>");
        assert_eq!(config["model_max_length"], 4096);
    }
}
