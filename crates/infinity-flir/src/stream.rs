use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};

pub const CHUNK_CHARS: usize = 6 * 1024;
const MAGIC: [u8; 2] = *b"FL";
const HEADER_LEN: usize = 8;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Encoding {
    #[default]
    Gray8,
    Rgba8,
}

impl Encoding {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Encoding::Gray8 => 1,
            Encoding::Rgba8 => 4,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Encoding::Gray8),
            1 => Some(Encoding::Rgba8),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Packet {
    pub width: u16,
    pub height: u16,
    pub encoding: Encoding,
    pub data: Vec<u8>,
}

impl Packet {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.data.len());
        out.extend_from_slice(&MAGIC);
        out.push(self.encoding as u8);
        out.push(0);
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER_LEN || b[..2] != MAGIC {
            return Err("not a flir packet".into());
        }
        let encoding = Encoding::from_u8(b[2]).ok_or(format!("unknown encoding {}", b[2]))?;
        let width = u16::from_le_bytes([b[4], b[5]]);
        let height = u16::from_le_bytes([b[6], b[7]]);
        let data = b[HEADER_LEN..].to_vec();
        let expected = width as usize * height as usize * encoding.bytes_per_pixel();
        if data.len() != expected {
            return Err(format!("expected {expected} pixel bytes, got {}", data.len()));
        }
        Ok(Self {
            width,
            height,
            encoding,
            data,
        })
    }

    pub fn to_rgba(&self) -> Vec<u8> {
        match self.encoding {
            Encoding::Rgba8 => self.data.clone(),
            Encoding::Gray8 => self.data.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct Chunk {
    pub seq: u64,
    pub idx: usize,
    pub total: usize,
    pub data: String,
}

pub fn chunks(seq: u64, packet: &Packet, chunk_chars: usize) -> Vec<Chunk> {
    let b64 = STANDARD.encode(packet.to_bytes());
    let parts: Vec<&str> = b64
        .as_bytes()
        .chunks(chunk_chars.max(4))
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    let total = parts.len();
    parts
        .into_iter()
        .enumerate()
        .map(|(idx, data)| Chunk {
            seq,
            idx,
            total,
            data: data.to_owned(),
        })
        .collect()
}

#[derive(Default)]
pub struct Reassembler {
    seq: u64,
    parts: Vec<Option<String>>,
    received: usize,
}

impl Reassembler {
    pub fn push(&mut self, chunk: Chunk) -> Option<Result<Packet, String>> {
        if chunk.total == 0 || chunk.idx >= chunk.total {
            return Some(Err(format!("bad chunk {}/{}", chunk.idx, chunk.total)));
        }
        if chunk.seq < self.seq {
            return None;
        }
        if chunk.seq != self.seq || self.parts.len() != chunk.total {
            self.seq = chunk.seq;
            self.parts = vec![None; chunk.total];
            self.received = 0;
        }
        let slot = &mut self.parts[chunk.idx];
        if slot.is_none() {
            self.received += 1;
        }
        *slot = Some(chunk.data);
        if self.received < self.parts.len() {
            return None;
        }
        let b64: String = self.parts.drain(..).flatten().collect();
        self.received = 0;
        self.seq += 1;
        Some(
            STANDARD
                .decode(b64)
                .map_err(|e| e.to_string())
                .and_then(|b| Packet::from_bytes(&b)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_out_of_order() {
        let packet = Packet {
            width: 64,
            height: 48,
            encoding: Encoding::Gray8,
            data: (0..64 * 48).map(|i| i as u8).collect(),
        };
        let mut parts = chunks(7, &packet, 256);
        assert!(parts.len() > 2);
        parts.reverse();
        let mut r = Reassembler::default();
        let last = parts.pop().unwrap();
        for c in parts {
            assert!(r.push(c).is_none());
        }
        assert_eq!(r.push(last).unwrap().unwrap(), packet);
    }

    #[test]
    fn stale_frames_are_dropped() {
        let packet = Packet {
            width: 1,
            height: 1,
            encoding: Encoding::Rgba8,
            data: vec![1, 2, 3, 4],
        };
        let mut r = Reassembler::default();
        let [c] = chunks(5, &packet, CHUNK_CHARS).try_into().unwrap();
        assert!(r.push(c.clone()).unwrap().is_ok());
        assert!(r.push(c).is_none());
    }
}
