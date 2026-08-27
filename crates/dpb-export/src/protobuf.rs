//! A minimal protobuf wire-format writer.
//!
//! ONNX models are protobuf, and this crate needs to emit them without taking
//! on a code-generation dependency for a schema it only writes a handful of
//! messages from. Only the encoding side is implemented -- nothing here parses.
//!
//! The wire format is documented at
//! <https://protobuf.dev/programming-guides/encoding/>. Each field is preceded
//! by a tag byte-sequence carrying the field number and a wire type:
//!
//! | Wire type | Meaning              | Used for                        |
//! |-----------|----------------------|---------------------------------|
//! | 0         | varint               | `int32`, `int64`, `bool`, enums |
//! | 2         | length-delimited     | `string`, `bytes`, sub-messages, packed repeated |
//! | 5         | 32-bit little-endian | `float`, `fixed32`              |

/// Wire type 0: variable-length integer.
const WIRE_VARINT: u32 = 0;
/// Wire type 2: length-delimited (strings, bytes, messages, packed arrays).
const WIRE_LEN: u32 = 2;
/// Wire type 5: fixed 32-bit little-endian.
const WIRE_I32: u32 = 5;

/// Accumulates protobuf-encoded bytes for one message.
///
/// Sub-messages are built with their own `Writer` and then embedded with
/// [`Writer::message`], which is what supplies their length prefix.
#[derive(Debug, Default, Clone)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    /// Creates an empty writer.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Consumes the writer and returns the encoded bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Returns the bytes encoded so far.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Returns true when nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn raw_varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                self.buf.push(byte);
                return;
            }
            self.buf.push(byte | 0x80);
        }
    }

    fn tag(&mut self, field: u32, wire: u32) {
        self.raw_varint(u64::from(field << 3 | wire));
    }

    /// Writes an `int64`/`int32`/enum field.
    ///
    /// Negative values occupy the full ten bytes, because protobuf sign-extends
    /// them to 64 bits rather than zig-zag encoding (that is `sint64`).
    pub fn int64(&mut self, field: u32, value: i64) {
        self.tag(field, WIRE_VARINT);
        self.raw_varint(value as u64);
    }

    /// Writes an `int32` field. Encoded exactly as [`Writer::int64`].
    pub fn int32(&mut self, field: u32, value: i32) {
        self.int64(field, i64::from(value));
    }

    /// Writes a `float` field.
    pub fn float(&mut self, field: u32, value: f32) {
        self.tag(field, WIRE_I32);
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Writes a length-delimited byte string.
    pub fn bytes(&mut self, field: u32, value: &[u8]) {
        self.tag(field, WIRE_LEN);
        self.raw_varint(value.len() as u64);
        self.buf.extend_from_slice(value);
    }

    /// Writes a `string` field. Skipped entirely when empty, matching proto3,
    /// where an empty string is indistinguishable from an absent one.
    pub fn string(&mut self, field: u32, value: &str) {
        if value.is_empty() {
            return;
        }
        self.bytes(field, value.as_bytes());
    }

    /// Embeds a sub-message, supplying its length prefix.
    pub fn message(&mut self, field: u32, inner: &Writer) {
        self.bytes(field, inner.as_bytes());
    }

    /// Writes a packed `repeated int64`, the proto3 default for scalars.
    ///
    /// Nothing is written for an empty slice: a zero-length packed field and an
    /// absent one decode identically, and omitting it keeps the output minimal.
    pub fn packed_int64(&mut self, field: u32, values: &[i64]) {
        if values.is_empty() {
            return;
        }
        let mut inner = Writer::new();
        for &v in values {
            inner.raw_varint(v as u64);
        }
        self.bytes(field, inner.as_bytes());
    }

    /// Writes a packed `repeated float`.
    pub fn packed_float(&mut self, field: u32, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let mut inner = Vec::with_capacity(values.len() * 4);
        for &v in values {
            inner.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes(field, &inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads one varint back, returning the value and how many bytes it used.
    fn read_varint(bytes: &[u8]) -> (u64, usize) {
        let mut value = 0u64;
        let mut shift = 0;
        for (i, &b) in bytes.iter().enumerate() {
            value |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return (value, i + 1);
            }
            shift += 7;
        }
        panic!("truncated varint");
    }

    #[test]
    fn varints_match_the_reference_encoding() {
        // Values and encodings taken from the protobuf encoding guide.
        for (value, expected) in [
            (0u64, vec![0x00]),
            (1, vec![0x01]),
            (127, vec![0x7f]),
            (128, vec![0x80, 0x01]),
            (300, vec![0xac, 0x02]),
        ] {
            let mut w = Writer::new();
            w.raw_varint(value);
            assert_eq!(w.finish(), expected, "varint {value}");
        }
    }

    #[test]
    fn negative_int64_sign_extends_to_ten_bytes() {
        // protobuf sign-extends negative varints rather than zig-zag encoding
        // them; -1 is therefore ten 0xff-ish bytes, not one.
        let mut w = Writer::new();
        w.raw_varint(-1i64 as u64);
        let bytes = w.finish();
        assert_eq!(bytes.len(), 10, "got {bytes:?}");
        let (back, used) = read_varint(&bytes);
        assert_eq!(used, 10);
        assert_eq!(back as i64, -1);
    }

    #[test]
    fn tag_packs_field_number_and_wire_type() {
        let mut w = Writer::new();
        w.string(2, "hi");
        let bytes = w.finish();
        // field 2, wire type 2 -> (2 << 3) | 2 == 0x12, then length 2.
        assert_eq!(bytes, vec![0x12, 0x02, b'h', b'i']);
    }

    #[test]
    fn empty_strings_and_packed_fields_are_omitted() {
        let mut w = Writer::new();
        w.string(1, "");
        w.packed_int64(2, &[]);
        w.packed_float(3, &[]);
        assert!(w.is_empty());
    }

    #[test]
    fn floats_are_little_endian_32_bit() {
        let mut w = Writer::new();
        w.float(1, 1.0);
        let bytes = w.finish();
        assert_eq!(bytes[0], (1 << 3) | 5);
        assert_eq!(&bytes[1..], &1.0f32.to_le_bytes());
    }

    #[test]
    fn nested_messages_carry_their_own_length() {
        let mut inner = Writer::new();
        inner.int64(1, 300);

        let mut outer = Writer::new();
        outer.message(7, &inner);

        let bytes = outer.finish();
        assert_eq!(bytes[0], (7 << 3) | 2, "outer tag");
        let (len, used) = read_varint(&bytes[1..]);
        assert_eq!(len as usize, bytes.len() - 1 - used);
    }

    #[test]
    fn packed_int64_round_trips() {
        let values = [1i64, 300, 0, 70000];
        let mut w = Writer::new();
        w.packed_int64(1, &values);
        let bytes = w.finish();

        let (len, used) = read_varint(&bytes[1..]);
        let mut body = &bytes[1 + used..];
        assert_eq!(body.len(), len as usize);

        let mut back = Vec::new();
        while !body.is_empty() {
            let (v, n) = read_varint(body);
            back.push(v as i64);
            body = &body[n..];
        }
        assert_eq!(back, values);
    }
}
