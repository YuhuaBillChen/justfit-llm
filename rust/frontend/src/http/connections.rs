//! Bound sockets before parsing HTTP, leaving incomplete heads in the kernel.
//! Ready heads keep their slot; idle heads and refused uploads give way.
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io,
    net::{Shutdown, SocketAddr, TcpListener as StdListener, TcpStream as StdStream},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::net::UnixStream,
    },
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    },
    task::{Context, Poll},
    thread,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::TcpStream,
    sync::mpsc as channel,
};
const HEAD_BYTES: usize = 65536;
const LINGER: Duration = Duration::from_secs(2);
const REFUSED: usize = 1024;
const OVERLOADED:&[u8]=b"{\"error\":{\"type\":\"server_error\",\"code\":\"frontend_overloaded\",\"message\":\"HTTP connection capacity is exhausted\"}}";
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Waiting,
    Serving,
    Draining,
}
struct Entry {
    fd: RawFd,
    phase: Phase,
    order: u64,
}
#[derive(Default)]
struct State {
    next: u64,
    entries: BTreeMap<u64, Entry>,
}
pub struct Gate {
    capacity: usize,
    state: Mutex<State>,
    stop: AtomicBool,
    wake: UnixStream,
    drains: mpsc::SyncSender<Drain>,
}
impl Gate {
    pub fn snapshot(&self) -> Value {
        json!({"active":self.state.lock().unwrap().entries.len(),"capacity":self.capacity})
    }
    fn remove(&self, id: u64) {
        self.state.lock().unwrap().entries.remove(&id);
    }
    fn contains(&self, id: u64) -> bool {
        self.state.lock().unwrap().entries.contains_key(&id)
    }
    fn wake(&self) {
        unsafe {
            libc::send(
                self.wake.as_raw_fd(),
                b"x".as_ptr().cast(),
                1,
                libc::MSG_DONTWAIT,
            );
        }
    }
    fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        self.wake();
    }
}
#[derive(Clone)]
pub struct Info {
    gate: Arc<Gate>,
    id: u64,
    head: usize,
    read: Arc<AtomicUsize>,
    body: Arc<Mutex<Option<(usize, Instant)>>>,
}
impl Info {
    pub fn body(&self, size: usize, deadline: Duration) {
        *self.body.lock().unwrap() = Some((size, Instant::now() + deadline));
    }
}
pub struct Socket {
    inner: Option<TcpStream>,
    info: Info,
}
impl Drop for Socket {
    fn drop(&mut self) {
        let body = *self.info.body.lock().unwrap();
        if let Some((size, deadline)) = body {
            let remaining = self
                .info
                .head
                .saturating_add(size)
                .saturating_sub(self.info.read.load(Ordering::Relaxed));
            if remaining > 0 && !self.info.gate.stop.load(Ordering::Acquire) {
                let stream = self.inner.take().expect("owned connection").into_std();
                if let Ok(stream) = stream {
                    let mut state = self.info.gate.state.lock().unwrap();
                    state.next += 1;
                    let order = state.next;
                    if let Some(entry) = state.entries.get_mut(&self.info.id) {
                        entry.phase = Phase::Draining;
                        entry.order = order;
                    }
                    drop(state);
                    let drain = Drain {
                        stream,
                        id: self.info.id,
                        remaining,
                        deadline,
                    };
                    match self.info.gate.drains.try_send(drain) {
                        Ok(()) => {
                            self.info.gate.wake();
                            return;
                        }
                        Err(mpsc::TrySendError::Full(drain) | mpsc::TrySendError::Disconnected(drain)) => {
                            // Release ownership under the lock before closing the FD.
                            self.info.gate.remove(self.info.id);
                            drop(drain);
                            return;
                        }
                    }
                }
            }
        }
        self.info.gate.remove(self.info.id);
    }
}
impl AsyncRead for Socket {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let before = buf.filled().len();
        let result = Pin::new(this.inner.as_mut().unwrap()).poll_read(cx, buf);
        this.info
            .read
            .fetch_add(buf.filled().len() - before, Ordering::Relaxed);
        result
    }
}
impl AsyncWrite for Socket {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(self.get_mut().inner.as_mut().unwrap()).poll_write(cx, buf)
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(self.get_mut().inner.as_mut().unwrap()).poll_flush(cx)
    }
    // Half-close only after any response bytes have been flushed by Hyper.
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(self.get_mut().inner.as_mut().unwrap()).poll_shutdown(cx)
    }
}
pub struct Listener {
    rx: channel::Receiver<(Socket, SocketAddr)>,
    address: SocketAddr,
    gate: Arc<Gate>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Drop for Listener {
    fn drop(&mut self) {
        self.gate.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
impl axum::serve::Listener for Listener {
    type Io = Socket;
    type Addr = SocketAddr;
    async fn accept(&mut self) -> (Socket, SocketAddr) {
        match self.rx.recv().await {
            Some(socket) => socket,
            None => std::future::pending().await,
        }
    }
    fn local_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.address)
    }
}
impl axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, Listener>> for Info {
    fn connect_info(stream: axum::serve::IncomingStream<'_, Listener>) -> Self {
        stream.io().info.clone()
    }
}
struct Waiting {
    stream: StdStream,
    address: SocketAddr,
    deadline: Instant,
}
struct Drain {
    stream: StdStream,
    id: u64,
    remaining: usize,
    deadline: Instant,
}
struct Lingering {
    stream: StdStream,
    deadline: Instant,
}
// Check exactly the same request-head boundary as released connections._ready.
pub fn ready_bytes(sent: &[u8]) -> bool {
    if sent.is_empty() || sent.len() >= HEAD_BYTES {
        return true;
    }
    let Some(newline) = sent.iter().position(|byte| *byte == b'\n') else {
        return false;
    };
    let words: Vec<_> = sent[..newline]
        .split(|byte| byte.is_ascii_whitespace() || *byte == 0x85 || *byte == 0xa0 || (0x1c..=0x1f).contains(byte))
        .filter(|word| !word.is_empty())
        .collect();
    let version = if words.len() == 3 {
        words[2].strip_prefix(b"HTTP/").and_then(|value| {
            let dot = value.iter().position(|byte| *byte == b'.')?;
            let (major, minor) = (&value[..dot], &value[dot + 1..]);
            if major.is_empty()
                || minor.is_empty()
                || major.len() > 10
                || minor.len() > 10
                || !major.iter().chain(minor).all(u8::is_ascii_digit)
            {
                return None;
            }
            Some((
                std::str::from_utf8(major).ok()?.parse::<u64>().ok()?,
                std::str::from_utf8(minor).ok()?.parse::<u64>().ok()?,
            ))
        })
    } else {
        None
    };
    if version.is_none_or(|version| version >= (2, 0)) {
        return true;
    }
    sent.windows(2).any(|bytes| bytes == b"\n\n") || sent.windows(3).any(|bytes| bytes == b"\n\r\n")
}
fn peek(fd: RawFd, buffer: &mut [u8]) -> io::Result<usize> {
    let size = unsafe {
        libc::recv(
            fd,
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    if size < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(size as usize)
    }
}
fn ready(fd: RawFd) -> bool {
    let mut buffer = [0; HEAD_BYTES];
    match peek(fd, &mut buffer) {
        Ok(size) => ready_bytes(&buffer[..size]),
        Err(error) => error.kind() != io::ErrorKind::WouldBlock,
    }
}
fn head_bytes(fd: RawFd) -> usize {
    let mut buffer = [0; HEAD_BYTES];
    let Ok(size) = peek(fd, &mut buffer) else {
        return 0;
    };
    let bytes = &buffer[..size];
    for index in 0..size {
        if bytes[index..].starts_with(b"\n\n") {
            return index + 2;
        }
        if bytes[index..].starts_with(b"\n\r\n") {
            return index + 3;
        }
    }
    size
}
fn drain(stream: &StdStream, remaining: &mut usize) -> bool {
    let mut buffer = [0u8; 65536];
    while *remaining > 0 {
        let amount = (*remaining).min(buffer.len());
        let size = unsafe {
            libc::recv(
                stream.as_raw_fd(),
                buffer.as_mut_ptr().cast(),
                amount,
                libc::MSG_DONTWAIT,
            )
        };
        if size > 0 {
            *remaining -= size as usize;
        } else if size == 0 {
            return false;
        } else {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return error.kind() == io::ErrorKind::WouldBlock;
        }
    }
    false
}
struct Events(OwnedFd);
#[cfg(target_os = "macos")]
impl Events {
    fn new() -> io::Result<Self> {
        let fd = unsafe { libc::kqueue() };
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            let owned = unsafe { OwnedFd::from_raw_fd(fd) };
            if unsafe { libc::fcntl(owned.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self(owned))
        }
    }
    fn watch(&self, fd: RawFd, add: bool) -> io::Result<()> {
        let change = libc::kevent {
            ident: fd as _,
            filter: libc::EVFILT_READ,
            flags: if add {
                libc::EV_ADD | libc::EV_CLEAR
            } else {
                libc::EV_DELETE
            },
            fflags: 0,
            data: 0,
            udata: std::ptr::null_mut(),
        };
        if unsafe {
            libc::kevent(
                self.0.as_raw_fd(),
                &change,
                1,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        } < 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    fn wait(&self, timeout: Duration) -> io::Result<()> {
        let mut events: [libc::kevent; 64] = unsafe { std::mem::zeroed() };
        let timeout = libc::timespec {
            tv_sec: timeout.as_secs() as _,
            tv_nsec: timeout.subsec_nanos() as _,
        };
        if unsafe {
            libc::kevent(
                self.0.as_raw_fd(),
                std::ptr::null(),
                0,
                events.as_mut_ptr(),
                events.len() as _,
                &timeout,
            )
        } < 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}
#[cfg(not(target_os = "macos"))]
impl Events {
    fn new() -> io::Result<Self> {
        let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(Self(unsafe { OwnedFd::from_raw_fd(fd) }))
        }
    }
    fn watch(&self, fd: RawFd, add: bool) -> io::Result<()> {
        let mut event = libc::epoll_event {
            events: (libc::EPOLLIN | libc::EPOLLET) as _,
            u64: fd as _,
        };
        if unsafe {
            libc::epoll_ctl(
                self.0.as_raw_fd(),
                if add {
                    libc::EPOLL_CTL_ADD
                } else {
                    libc::EPOLL_CTL_DEL
                },
                fd,
                &mut event,
            )
        } < 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    fn wait(&self, timeout: Duration) -> io::Result<()> {
        let mut events: [libc::epoll_event; 64] = unsafe { std::mem::zeroed() };
        if unsafe {
            libc::epoll_wait(
                self.0.as_raw_fd(),
                events.as_mut_ptr(),
                events.len() as _,
                timeout.as_millis().min(i32::MAX as u128) as _,
            )
        } < 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}
pub fn descriptor_limit() {
    unsafe {
        let mut limit: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) == 0 {
            let wanted = 10240.min(limit.rlim_max);
            if limit.rlim_cur < wanted {
                limit.rlim_cur = wanted;
                if libc::setrlimit(libc::RLIMIT_NOFILE, &limit) != 0 {
                    eprintln!(
                        "could not raise HTTP descriptor limit: {}",
                        io::Error::last_os_error()
                    )
                }
            }
        }
    }
}
impl Listener {
    pub fn new(
        listener: tokio::net::TcpListener,
        capacity: usize,
    ) -> io::Result<(Self, Arc<Gate>)> {
        let address = listener.local_addr()?;
        let listener = listener.into_std()?;
        listener.set_nonblocking(true)?;
        let (wake, reader) = UnixStream::pair()?;
        wake.set_nonblocking(true)?;
        reader.set_nonblocking(true)?;
        let (drain_tx, drain_rx) = mpsc::sync_channel(capacity);
        let gate = Arc::new(Gate {
            capacity,
            state: Mutex::new(State::default()),
            stop: AtomicBool::new(false),
            wake,
            drains: drain_tx,
        });
        let events = Events::new()?;
        events.watch(listener.as_raw_fd(), true)?;
        events.watch(reader.as_raw_fd(), true)?;
        let (tx, rx) = channel::channel(capacity);
        let core = gate.clone();
        let handle = tokio::runtime::Handle::current();
        let worker = thread::Builder::new()
            .name("HTTP admission".into())
            .spawn(move || {
                if let Err(error) =
                    broker(listener, reader, events, core.clone(), drain_rx, tx, handle)
                {
                    eprintln!("HTTP admission stopped: {error}");
                    core.stop();
                }
            })?;
        Ok((
            Self {
                rx,
                address,
                gate: gate.clone(),
                worker: Some(worker),
            },
            gate,
        ))
    }
}
fn refuse(
    stream: StdStream,
    events: &Events,
    lingering: &mut BTreeMap<u64, Lingering>,
    next: &mut u64,
) {
    let mut response=format!("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\nContent-Length: {}\r\n\r\n",OVERLOADED.len()).into_bytes();
    response.extend_from_slice(OVERLOADED);
    unsafe {
        libc::send(
            stream.as_raw_fd(),
            response.as_ptr().cast(),
            response.len(),
            libc::MSG_DONTWAIT,
        );
    }
    let _ = stream.shutdown(Shutdown::Write);
    let mut remaining = usize::MAX;
    if !drain(&stream, &mut remaining) {
        return;
    }
    if lingering.len() >= REFUSED {
        let id = *lingering.keys().next().unwrap();
        if let Some(old) = lingering.remove(&id) {
            let _ = events.watch(old.stream.as_raw_fd(), false);
        }
    }
    if events.watch(stream.as_raw_fd(), true).is_ok() {
        *next += 1;
        lingering.insert(
            *next,
            Lingering {
                stream,
                deadline: Instant::now() + LINGER,
            },
        );
    }
}
fn broker(
    listener: StdListener,
    reader: UnixStream,
    events: Events,
    gate: Arc<Gate>,
    drain_rx: mpsc::Receiver<Drain>,
    tx: channel::Sender<(Socket, SocketAddr)>,
    handle: tokio::runtime::Handle,
) -> io::Result<()> {
    let mut waiting: BTreeMap<u64, Waiting> = BTreeMap::new();
    let mut draining: BTreeMap<u64, Drain> = BTreeMap::new();
    let mut lingering = BTreeMap::new();
    let mut next_linger = 0;
    let result = (|| {
        while !gate.stop.load(Ordering::Acquire) {
            let mut bytes = [0; 256];
            while unsafe {
                libc::recv(
                    reader.as_raw_fd(),
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                    libc::MSG_DONTWAIT,
                )
            } > 0
            {}
            while let Ok(item) = drain_rx.try_recv() {
                if gate.contains(item.id) {
                    let _ = item.stream.shutdown(Shutdown::Write);
                    events.watch(item.stream.as_raw_fd(), true)?;
                    draining.insert(item.id, item);
                }
            }
            loop {
                if gate.stop.load(Ordering::Acquire) {
                    break;
                }
                let (stream, address) = match listener.accept() {
                    Ok(value) => value,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                };
                stream.set_nonblocking(true)?;
                let mut state = gate.state.lock().unwrap();
                if state.entries.len() >= gate.capacity {
                    let giving_way = state
                        .entries
                        .iter()
                        .filter(|(_, entry)| {
                            entry.phase == Phase::Draining
                                || (entry.phase == Phase::Waiting && !ready(entry.fd))
                        })
                        .min_by_key(|(_, entry)| entry.order)
                        .map(|(id, _)| *id);
                    let Some(id) = giving_way else {
                        drop(state);
                        refuse(stream, &events, &mut lingering, &mut next_linger);
                        continue;
                    };
                    let entry = state.entries.remove(&id).unwrap();
                    if entry.phase == Phase::Draining {
                        unsafe {
                            libc::shutdown(entry.fd, libc::SHUT_RDWR);
                        }
                    }
                    drop(state);
                    if let Some(old) = waiting.remove(&id) {
                        let _ = events.watch(old.stream.as_raw_fd(), false);
                        refuse(old.stream, &events, &mut lingering, &mut next_linger);
                    }
                    state = gate.state.lock().unwrap();
                }
                state.next += 1;
                let id = state.next;
                state.entries.insert(
                    id,
                    Entry {
                        fd: stream.as_raw_fd(),
                        phase: Phase::Waiting,
                        order: id,
                    },
                );
                drop(state);
                events.watch(stream.as_raw_fd(), true)?;
                waiting.insert(
                    id,
                    Waiting {
                        stream,
                        address,
                        deadline: Instant::now() + Duration::from_secs(30),
                    },
                );
            }
            let now = Instant::now();
            let mut remove = Vec::new();
            for (id, item) in &waiting {
                if now >= item.deadline || ready(item.stream.as_raw_fd()) {
                    remove.push(*id)
                }
            }
            for id in remove {
                let item = waiting.remove(&id).unwrap();
                let _ = events.watch(item.stream.as_raw_fd(), false);
                if now >= item.deadline {
                    gate.remove(id);
                    let mut remaining=usize::MAX;
                    drain(&item.stream,&mut remaining);
                    continue;
                }
                let head = head_bytes(item.stream.as_raw_fd());
                let _entered = handle.enter();
                let stream = TcpStream::from_std(item.stream)?;
                if let Some(entry) = gate.state.lock().unwrap().entries.get_mut(&id) {
                    entry.phase = Phase::Serving;
                }
                let info = Info {
                    gate: gate.clone(),
                    id,
                    head,
                    read: Arc::new(AtomicUsize::new(0)),
                    body: Arc::new(Mutex::new(None)),
                };
                if tx
                    .try_send((
                        Socket {
                            inner: Some(stream),
                            info,
                        },
                        item.address,
                    ))
                    .is_err()
                {
                    return Err(io::Error::other(
                        "HTTP ready queue closed or inconsistent capacity",
                    ));
                }
            }
            draining.retain(|id, item| {
                let keep = gate.contains(*id)
                    && now < item.deadline
                    && drain(&item.stream, &mut item.remaining);
                if !keep {
                    let _ = events.watch(item.stream.as_raw_fd(), false);
                    gate.remove(*id);
                }
                keep
            });
            lingering.retain(|_, item| {
                let mut remaining = usize::MAX;
                let keep = now < item.deadline && drain(&item.stream, &mut remaining);
                if !keep {
                    let _ = events.watch(item.stream.as_raw_fd(), false);
                }
                keep
            });
            let deadline = waiting
                .values()
                .map(|item| item.deadline)
                .chain(draining.values().map(|item| item.deadline))
                .chain(lingering.values().map(|item| item.deadline))
                .min();
            let timeout = deadline.map_or(Duration::from_secs(30), |deadline| {
                deadline.saturating_duration_since(Instant::now())
            });
            if let Err(error) = events.wait(timeout) {
                if error.kind() != io::ErrorKind::Interrupted {
                    return Err(error);
                }
            }
        }
        Ok(())
    })();
    gate.state
        .lock()
        .unwrap()
        .entries
        .retain(|_, entry| entry.phase == Phase::Serving);
    // Match _close_at_once: consume pending bytes before closing to avoid RST.
    for stream in waiting.values().map(|item|&item.stream)
        .chain(draining.values().map(|item|&item.stream))
        .chain(lingering.values().map(|item|&item.stream)) {
        let mut remaining=usize::MAX;drain(stream,&mut remaining);
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn event_descriptor_is_closed_on_native_exec() {
        let events = Events::new().unwrap();
        let flags = unsafe { libc::fcntl(events.0.as_raw_fd(), libc::F_GETFD) };
        assert_ne!(flags & libc::FD_CLOEXEC, 0);
    }
    #[test]
    fn incomplete_heads_wait_but_invalid_lines_and_versions_are_ready() {
        for bytes in [
            b"GET / HTTP/1.1\r\nHost: local\r\n".as_slice(),
            b"GET /",
            b"GET / HTTP/1.1\n",
        ] {
            assert!(!ready_bytes(bytes))
        }
        for bytes in [
            b"".as_slice(),
            b"GET / HTTP/1.1\r\nHost: local\r\n\r\n",
            b"GET /\n",
            b"GET / HTTP/2.0\n",
            b"GET / HTTP/x.1\n",
        ] {
            assert!(ready_bytes(bytes))
        }
        assert!(ready_bytes(&vec![b'x'; HEAD_BYTES]));
    }
}
