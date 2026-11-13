//! Conservative PoC: only a sole started request at two fault-signal deaths
//! is quarantined. Ambiguous concurrent failures never implicate a peer.
use justfit_native_client_prototype::Frame;
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};

pub(super) type Fingerprint = [u8; 32];
const LIMIT: usize = 4096;

pub(super) fn fingerprint(frame: &Frame) -> Option<Fingerprint> {
    if frame.kind != 1 || frame.payload.len() < 94 { return None; }
    let mut hash = Sha256::new();
    // Request id and deadline/remaining time vary on each retry.
    hash.update(&frame.payload[8..10]);
    hash.update(&frame.payload[26..]);
    Some(hash.finalize().into())
}

pub(super) fn fault_signal(signal: Option<i32>) -> bool {
    matches!(signal, Some(libc::SIGABRT | libc::SIGSEGV | libc::SIGBUS | libc::SIGILL | libc::SIGFPE))
}

#[derive(Default)]
pub(super) struct Quarantine {
    suspect: Option<Fingerprint>,
    refused: HashSet<Fingerprint>,
    order: VecDeque<Fingerprint>,
}
impl Quarantine {
    pub fn refuses(&self, key: &Fingerprint) -> bool { self.refused.contains(key) }
    pub fn after_fault(&mut self, started: &[Fingerprint]) {
        let sole = if started.len() == 1 { Some(started[0]) } else { None };
        if let Some(key) = sole.filter(|key| Some(*key) == self.suspect) {
            if self.refused.insert(key) {
                self.order.push_back(key);
                if self.order.len() > LIMIT {
                    self.refused.remove(&self.order.pop_front().unwrap());
                }
            }
        }
        self.suspect = sole;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use justfit_native_client_prototype::text_request;
    #[test]
    fn retries_ignore_id_deadline_but_not_tokens_or_sampling() {
        let a = text_request(1, &[10, 11], 16, 0, 0).unwrap();
        let mut b = text_request(2, &[10, 11], 16, 0, 0).unwrap();
        b.payload[10..26].fill(0);
        assert_eq!(fingerprint(&a), fingerprint(&b));
        b.payload[94] ^= 1;
        assert_ne!(fingerprint(&a), fingerprint(&b));
        let c = text_request(3, &[10, 11], 17, 0, 0).unwrap();
        assert_ne!(fingerprint(&a), fingerprint(&c));
    }
    #[test]
    fn only_repeated_unambiguous_faults_are_quarantined() {
        let mut q = Quarantine::default(); let a = [1;32]; let b = [2;32];
        q.after_fault(&[a]); assert!(!q.refuses(&a));
        q.after_fault(&[a,b]); q.after_fault(&[a]); assert!(!q.refuses(&a));
        q.after_fault(&[a]); assert!(q.refuses(&a)); assert!(!q.refuses(&b));
        assert!(!fault_signal(Some(libc::SIGKILL)));
        assert!(!fault_signal(Some(libc::SIGTERM)));
        assert!(!fault_signal(None));
        assert!(fault_signal(Some(libc::SIGABRT)));
    }
}
