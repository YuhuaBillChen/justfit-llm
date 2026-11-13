//! Protocol9 generation and finite-option scoring transport; C++ engine unchanged.
use std::io::{self, ErrorKind, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

pub const VERSION: u16 = 9;
pub const READY: u16 = 0x100;
pub const START: u16 = 0x101;
pub const TOKENS: u16 = 0x102;
pub const MASK: u16 = 0x103;
pub const DONE: u16 = 0x104;
pub const ERROR: u16 = 0x105;
pub const STATUS: u16 = 0x107;
pub const PROGRESS: u16 = 0x108;
// Experiment (JUSTFIT_TREE): a draft tree's mask request, rows then parents.
pub const MASK_TREE: u16 = 0x109;

fn invalid(message: &str) -> io::Error { io::Error::new(ErrorKind::InvalidData, message) }

fn bounds(kind: u16) -> io::Result<(usize, usize)> {
    Ok(match kind {
        1 => (94, 256 << 20), 2 | 4 => (8, 8), 3 => (20, 20 + (1 << 20) * 4),
        5 => (24, 24), READY => (9, 9), START => (16, 16),
        TOKENS => (16, 16 + 4096 * 4), MASK => (24, 24 + 32 * 4), MASK_TREE => (24, 24 + 2 * 32 * 4),
        DONE => (45, 45 + 255 * 4), ERROR => (18, 18 + (2 << 20)),
        STATUS => (8, 8 + (32 << 20)), PROGRESS => (20, 20),
        _ => return Err(invalid("unknown frame type")),
    })
}

#[derive(Debug, PartialEq, Eq)]
pub struct Frame { pub kind: u16, pub payload: Vec<u8> }

pub fn read_frame(reader: &mut impl Read) -> io::Result<Option<Frame>> {
    let mut header = [0u8; 16];
    loop {
        match reader.read(&mut header[..1]) {
            Ok(0) => return Ok(None), Ok(_) => break,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    reader.read_exact(&mut header[1..])?;
    if &header[..4] != b"SPLH" { return Err(invalid("bad magic")); }
    if u16::from_le_bytes(header[4..6].try_into().unwrap()) != VERSION {
        return Err(invalid("unsupported protocol version"));
    }
    let kind = u16::from_le_bytes(header[6..8].try_into().unwrap());
    let size = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let (minimum, maximum) = bounds(kind)?;
    if size < minimum as u64 || size > maximum as u64 { return Err(invalid("invalid payload length")); }
    let mut payload = Vec::new();
    payload.try_reserve_exact(size as usize).map_err(|_| invalid("payload allocation refused"))?;
    payload.resize(size as usize, 0);
    reader.read_exact(&mut payload)?;
    Ok(Some(Frame { kind, payload }))
}

pub fn write_frame(writer: &mut impl Write, frame: &Frame) -> io::Result<()> {
    let (minimum, maximum) = bounds(frame.kind)?;
    if !(minimum..=maximum).contains(&frame.payload.len()) { return Err(invalid("invalid payload length")); }
    writer.write_all(b"SPLH")?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&frame.kind.to_le_bytes())?;
    writer.write_all(&(frame.payload.len() as u64).to_le_bytes())?;
    writer.write_all(&frame.payload)?;
    writer.flush()
}

pub fn text_request(id: u64, tokens: &[u32], output: u32, thinking: u32, end: u32) -> io::Result<Frame> {
    if id == 0 || tokens.is_empty() || tokens.len() > 1 << 20 || output == 0 || output > 1 << 20 {
        return Err(invalid("invalid request id or token budget"));
    }
    if (thinking > 0 && (end == 0 || thinking as u64 + 1 >= output as u64)) || (thinking == 0 && end != 0) {
        return Err(invalid("thinking cap must preserve final-answer capacity"));
    }
    let mut p = Vec::with_capacity(94 + tokens.len() * 4);
    p.extend(id.to_le_bytes()); p.extend([1, 0]); // normal priority, unconstrained
    p.extend(u64::MAX.to_le_bytes()); p.extend(u64::MAX.to_le_bytes());
    for value in [output, tokens.len() as u32, 0] { p.extend(value.to_le_bytes()); }
    for bits in [0, 1f32.to_bits(), 0, 0, 0, 1f32.to_bits(), 0] { p.extend(bits.to_le_bytes()); }
    p.extend(0u64.to_le_bytes());
    for value in [0, 0, 0, thinking, end] { p.extend(value.to_le_bytes()); }
    for value in tokens { p.extend(value.to_le_bytes()); }
    Ok(Frame { kind: 1, payload: p })
}

pub fn cancel(id: u64) -> io::Result<Frame> {
    if id == 0 { return Err(invalid("cancel id must be nonzero")); }
    Ok(Frame { kind: 2, payload: id.to_le_bytes().to_vec() })
}

pub fn score_request(id:u64,tokens:&[u32],options:&[u32])->io::Result<Frame> {
    if !(2..=255).contains(&options.len())||options.iter().collect::<std::collections::BTreeSet<_>>().len()!=options.len(){return Err(invalid("score tokens must be distinct ids,2..255"));}
    let mut frame=text_request(id,tokens,1,0,0)?;
    frame.payload[26..30].copy_from_slice(&0u32.to_le_bytes());frame.payload[74..78].copy_from_slice(&(options.len()as u32).to_le_bytes());
    for token in options{frame.payload.extend(token.to_le_bytes());}Ok(frame)
}

fn u32_at(p: &[u8], offset: usize) -> io::Result<u32> {
    Ok(u32::from_le_bytes(p.get(offset..offset + 4).ok_or_else(|| invalid("short uint32"))?.try_into().unwrap()))
}
fn u64_at(p: &[u8], offset: usize) -> io::Result<u64> {
    Ok(u64::from_le_bytes(p.get(offset..offset + 8).ok_or_else(|| invalid("short uint64"))?.try_into().unwrap()))
}

#[derive(Debug)]
pub struct Outcome { pub tokens: Vec<u32>, pub reason: u8, pub prompt: u32 }

pub struct Client {
    child: Child, input: Option<ChildStdin>, output: ChildStdout,
    pub lanes: u32, pub context: u32, pub vision: bool,
}

impl Client {
    pub fn spawn(command: &[String]) -> io::Result<Self> {
        let program = command.first().ok_or_else(|| invalid("missing native command"))?;
        let mut child = Command::new(program).args(&command[1..]).stdin(Stdio::piped())
            .stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn()?;
        let input = child.stdin.take();
        let output = child.stdout.take().ok_or_else(|| invalid("missing native stdout"))?;
        let mut client = Self { child, input, output, lanes: 0, context: 0, vision: false };
        let ready = read_frame(&mut client.output)?.ok_or_else(|| invalid("EOF before Ready"))?;
        if ready.kind != READY { return Err(invalid("native did not announce Ready")); }
        client.lanes = u32_at(&ready.payload, 0)?;
        client.context = u32_at(&ready.payload, 4)?;
        if client.lanes == 0 || client.context == 0 || ready.payload[8] > 1 { return Err(invalid("invalid Ready fields")); }
        client.vision = ready.payload[8] != 0;
        Ok(client)
    }

    pub fn run(&mut self, id: u64, prompt: &[u32], budget: u32, cancel_after: Option<usize>) -> io::Result<Outcome> {
        write_frame(self.input.as_mut().ok_or_else(|| invalid("client closed"))?, &text_request(id, prompt, budget, 0, 0)?)?;
        let mut tokens = Vec::new();
        let mut cancelled = false;
        loop {
            let frame = read_frame(&mut self.output)?.ok_or_else(|| invalid("EOF before Done"))?;
            let p = &frame.payload;
            match frame.kind {
                START | PROGRESS => {
                    if u64_at(p, 0)? != id { return Err(invalid("event belongs to another request")); }
                }
                TOKENS => {
                    if u64_at(p, 0)? != id || u32_at(p, 8)? as usize != tokens.len() { return Err(invalid("token sequence discontinuity")); }
                    let count = u32_at(p, 12)? as usize;
                    if count == 0 || p.len() != 16 + count * 4 { return Err(invalid("invalid token batch")); }
                    for i in 0..count { tokens.push(u32_at(p, 16 + i * 4)?); }
                    if !cancelled && cancel_after.is_some_and(|n| tokens.len() >= n) {
                        write_frame(self.input.as_mut().unwrap(), &cancel(id)?)?;
                        cancelled = true;
                    }
                }
                DONE => {
                    if u64_at(p, 0)? != id || p[8] > 2 || u32_at(p, 41)? != 0 || p.len() != 45 {
                        return Err(invalid("invalid generation Done"));
                    }
                    if u32_at(p, 13)? as usize != tokens.len() { return Err(invalid("Done count differs from streamed tokens")); }
                    if cancelled && p[8] != 2 { return Err(invalid("cancellation was not exercised")); }
                    return Ok(Outcome { tokens, reason: p[8], prompt: u32_at(p, 9)? });
                }
                MASK => return Err(io::Error::new(ErrorKind::Unsupported, "prototype does not evaluate grammar masks")),
                ERROR => return Err(invalid(&format!("native error event: {:?}", p))),
                _ => return Err(invalid("unexpected engine event")),
            }
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.input.take();
        for _ in 0..25 {
            if matches!(self.child.try_wait(), Ok(Some(_))) { return; }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill(); // Only this client's owned native child.
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    struct Fragments(Cursor<Vec<u8>>);
    impl Read for Fragments {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            let n = bytes.len().min(3); self.0.read(&mut bytes[..n])
        }
    }
    #[test]
    fn fragmented_concatenated_frames_preserve_boundaries() {
        let first = cancel(7).unwrap(); let second = cancel(8).unwrap();
        let mut bytes = Vec::new(); write_frame(&mut bytes, &first).unwrap(); write_frame(&mut bytes, &second).unwrap();
        let mut reader = Fragments(Cursor::new(bytes));
        assert_eq!(read_frame(&mut reader).unwrap(), Some(first));
        assert_eq!(read_frame(&mut reader).unwrap(), Some(second));
        assert_eq!(read_frame(&mut reader).unwrap(), None);
    }
    #[test]
    fn truncated_and_invalid_headers_are_rejected_before_payload_allocation() {
        let mut valid = Vec::new(); write_frame(&mut valid, &cancel(7).unwrap()).unwrap();
        for end in 1..valid.len() { assert!(read_frame(&mut Cursor::new(&valid[..end])).is_err()); }
        for (offset, value) in [(0, 0), (4, 8), (6, 99), (8, 255)] {
            let mut broken = valid.clone(); broken[offset] = value;
            assert!(read_frame(&mut Cursor::new(broken)).is_err());
        }
    }
}
