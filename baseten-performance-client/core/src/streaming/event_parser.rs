use std::{collections::VecDeque, convert::TryFrom, str::from_utf8};

use pin_project::pin_project;
use tracing::{debug, trace};

use super::response::Response;
use super::error::{Error, Result};
use super::types::Event;

#[derive(Default, PartialEq)]
struct EventData {
    pub event_type: String,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

impl EventData {
    fn new() -> Self {
        Self::default()
    }

    pub fn add_data_field(&mut self, value: &str) {
        // Handle multiple data fields according to SSE spec
        if self.data.is_empty() {
            self.data = value.to_string();
        } else {
            // Multiple data fields should be joined with newlines
            self.data.push('\n');
            self.data.push_str(value);
        }
    }

    pub fn get_final_data(&self) -> String {
        // Remove trailing newline as per SSE spec
        match self.data.strip_suffix('\n') {
            Some(data_without_newline) => data_without_newline.to_string(),
            None => self.data.clone(),
        }
    }

    pub fn with_id(mut self, value: Option<String>) -> Self {
        self.id = value;
        self
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum SSE {
    Connected(ConnectionDetails),
    Event(Event),
    Comment(String),
}

impl TryFrom<EventData> for Option<SSE> {
    type Error = Error;

    fn try_from(event_data: EventData) -> std::result::Result<Self, Self::Error> {
        if event_data == EventData::default() {
            return Err(Error::InvalidEvent);
        }

        if event_data.data.is_empty() {
            return Ok(None);
        }

        let event_type = if event_data.event_type.is_empty() {
            String::from("message")
        } else {
            event_data.event_type.clone()
        };

        let data = event_data.get_final_data();
        let id = event_data.id.clone();
        let retry = event_data.retry;

        Ok(Some(SSE::Event(Event {
            event_type,
            data,
            id,
            retry,
        })))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConnectionDetails {
    response: Response,
}

impl ConnectionDetails {
    pub(crate) fn new(response: Response) -> Self {
        Self { response }
    }

    /// Returns information describing the response at the time of connection.
    pub fn response(&self) -> &Response {
        &self.response
    }
}

const LOGIFY_MAX_CHARS: usize = 100;
fn logify(bytes: &[u8]) -> String {
    let stringified = from_utf8(bytes).unwrap_or("<bad utf8>");
    stringified.chars().take(LOGIFY_MAX_CHARS).collect()
}

fn parse_field(line: &[u8]) -> Result<Option<(&str, &str)>> {
    if line.is_empty() {
        return Err(Error::InvalidLine(
            "should never try to parse an empty line (probably a bug)".into(),
        ));
    }

    match line.iter().position(|&b| b':' == b) {
        Some(0) => {
            let value = &line[1..];
            Ok(Some(("comment", parse_value(value)?)))
        }
        Some(colon_pos) => {
            let key = &line[0..colon_pos];
            let key = parse_key(key)?;

            let mut value = &line[colon_pos + 1..];
            // remove the first initial space character if any (but remove no other whitespace)
            if value.starts_with(b" ") {
                value = &value[1..];
            }

            Ok(Some((key, parse_value(value)?)))
        }
        None => Ok(Some((parse_key(line)?, ""))),
    }
}

fn parse_key(key: &[u8]) -> Result<&str> {
    from_utf8(key).map_err(|e| Error::InvalidLine(format!("malformed key: {e:?}")))
}

fn parse_value(value: &[u8]) -> Result<&str> {
    from_utf8(value).map_err(|e| Error::InvalidLine(format!("malformed value: {e:?}")))
}

#[pin_project]
#[must_use = "streams do nothing unless polled"]
pub struct EventParser {
    /// buffer for lines we know are complete (terminated) but not yet parsed into event fields, in
    /// the order received
    complete_lines: VecDeque<Vec<u8>>,
    /// buffer for the most-recently received line, pending completion (by a newline terminator) or
    /// extension (by more non-newline bytes)
    incomplete_line: Option<Vec<u8>>,
    /// flagged if the last character processed as a carriage return; used to help process CRLF
    /// pairs
    last_char_was_cr: bool,
    /// the event data currently being decoded
    event_data: Option<EventData>,
    /// the last-seen event ID; events without an ID will take on this value until it is updated.
    last_event_id: Option<String>,
    sse: VecDeque<SSE>,
}

impl EventParser {
    pub fn new() -> Self {
        Self {
            complete_lines: VecDeque::with_capacity(10),
            incomplete_line: None,
            last_char_was_cr: false,
            event_data: None,
            last_event_id: None,
            sse: VecDeque::with_capacity(3),
        }
    }

    pub fn was_processing(&self) -> bool {
        if self.incomplete_line.is_some() || !self.complete_lines.is_empty() {
            true
        } else {
            !self.sse.is_empty()
        }
    }

    pub fn get_event(&mut self) -> Option<SSE> {
        self.sse.pop_front()
    }

    pub fn process_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        trace!("Parsing bytes {:?}", bytes);
        // We get bytes from the underlying stream in chunks.  Decoding a chunk has two phases:
        // decode the chunk into lines, and decode the lines into events.
        //
        // We counterintuitively do these two phases in reverse order. Because both lines and
        // events may be split across chunks, we need to ensure we have a complete
        // (newline-terminated) line before parsing it, and a complete event
        // (empty-line-terminated) before returning it. So we buffer lines between poll()
        // invocations, and begin by processing any incomplete events from previous invocations,
        // before requesting new input from the underlying stream and processing that.

        self.decode_and_buffer_lines(bytes);
        self.parse_complete_lines_into_event()?;

        Ok(())
    }

    // Populate the event fields from the complete lines already seen, until we either encounter an
    // empty line - indicating we've decoded a complete event - or we run out of complete lines to
    // process.
    //
    // Returns the event for dispatch if it is complete.
    fn parse_complete_lines_into_event(&mut self) -> Result<()> {
        loop {
            let mut seen_empty_line = false;

            while let Some(line) = self.complete_lines.pop_front() {
                if line.is_empty() && self.event_data.is_some() {
                    seen_empty_line = true;
                    break;
                } else if line.is_empty() {
                    continue;
                }

                if let Some((key, value)) = parse_field(&line)? {
                    if key == "comment" {
                        self.sse.push_back(SSE::Comment(value.to_string()));
                        continue;
                    }

                    let id = &self.last_event_id;
                    let event_data = self
                        .event_data
                        .get_or_insert_with(|| EventData::new().with_id(id.clone()));

                    if key == "event" {
                        event_data.event_type = value.to_string()
                    } else if key == "data" {
                        event_data.add_data_field(value);
                    } else if key == "id" {
                        // If id contains a null byte, it is a non-fatal error and the rest of
                        // the event should be parsed if possible.
                        if value.chars().any(|c| c == '\0') {
                            debug!("Ignoring event ID containing null byte");
                            continue;
                        }

                        if value.is_empty() {
                            self.last_event_id = Some("".to_string());
                        } else {
                            self.last_event_id = Some(value.to_string());
                        }

                        event_data.id.clone_from(&self.last_event_id)
                    } else if key == "retry" {
                        match value.parse::<u64>() {
                            Ok(retry) => {
                                event_data.retry = Some(retry);
                            }
                            _ => debug!("Failed to parse {:?} into retry value", value),
                        };
                    }
                }
            }

            if seen_empty_line {
                let event_data = self.event_data.take();

                trace!(
                    "seen empty line, event_data is {:?})",
                    event_data.as_ref().map(|event_data| &event_data.event_type)
                );

                if let Some(event_data) = event_data {
                    match Option::<SSE>::try_from(event_data) {
                        Err(e) => return Err(e),
                        Ok(None) => (),
                        Ok(Some(event)) => self.sse.push_back(event),
                    };
                }

                continue;
            } else {
                trace!("processed all complete lines but event_data not yet complete");
            }

            break;
        }

        Ok(())
    }

    // Decode a chunk into lines and buffer them for subsequent parsing, taking account of
    // incomplete lines from previous chunks.
    fn decode_and_buffer_lines(&mut self, chunk: &[u8]) {
        let mut lines = chunk.split_inclusive(|&b| b == b'\n' || b == b'\r');
        // The first and last elements in this split are special. The spec requires lines to be
        // terminated. But lines may span chunks, so:
        //  * the last line, if non-empty (i.e. if chunk didn't end with a line terminator),
        //    should be buffered as an incomplete line
        //  * the first line should be appended to the incomplete line, if any

        if let Some(incomplete_line) = self.incomplete_line.as_mut() {
            if let Some(line) = lines.next() {
                trace!(
                    "extending line from previous chunk: {:?}+{:?}",
                    logify(incomplete_line),
                    logify(line)
                );

                self.last_char_was_cr = false;
                if !line.is_empty() {
                    // Checking the last character handles lines where the last character is a
                    // terminator, but also where the entire line is a terminator.
                    match line.last().unwrap() {
                        b'\r' => {
                            incomplete_line.extend_from_slice(&line[..line.len() - 1]);
                            let il = self.incomplete_line.take();
                            self.complete_lines.push_back(il.unwrap());
                            self.last_char_was_cr = true;
                        }
                        b'\n' => {
                            incomplete_line.extend_from_slice(&line[..line.len() - 1]);
                            let il = self.incomplete_line.take();
                            self.complete_lines.push_back(il.unwrap());
                        }
                        _ => incomplete_line.extend_from_slice(line),
                    };
                }
            }
        }

        let mut lines = lines.peekable();
        while let Some(line) = lines.next() {
            if let Some(actually_complete_line) = self.incomplete_line.take() {
                // we saw the next line, so the previous one must have been complete after all
                trace!(
                    "previous line was complete: {:?}",
                    logify(&actually_complete_line)
                );
                self.complete_lines.push_back(actually_complete_line);
            }

            if self.last_char_was_cr && line == [b'\n'] {
                // This is a continuation of a \r\n pair, so we can ignore this line. We do need to
                // reset our flag though.
                self.last_char_was_cr = false;
                continue;
            }

            self.last_char_was_cr = false;
            if line.ends_with(b"\r") {
                self.complete_lines
                    .push_back(line[..line.len() - 1].to_vec());
                self.last_char_was_cr = true;
            } else if line.ends_with(b"\n") {
                // self isn't a continuation, but rather a line ending with a LF terminator.
                self.complete_lines
                    .push_back(line[..line.len() - 1].to_vec());
            } else if line.is_empty() {
                // this is the last line and it's empty, no need to buffer it
                trace!("chunk ended with a line terminator");
            } else if lines.peek().is_some() {
                // this line isn't the last and we know from previous checks it doesn't end in a
                // terminator, so we can consider it complete
                self.complete_lines.push_back(line.to_vec());
            } else {
                // last line needs to be buffered as it may be incomplete
                trace!("buffering incomplete line: {:?}", logify(line));
                self.incomplete_line = Some(line.to_vec());
            }
        }

        if tracing::level_enabled!(tracing::Level::TRACE) {
            for line in &self.complete_lines {
                trace!("complete line: {:?}", logify(line));
            }
            if let Some(line) = &self.incomplete_line {
                trace!("incomplete line: {:?}", logify(line));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_data_field_handling() {
        let mut event_data = EventData::new();

        // Test single data field
        event_data.add_data_field("hello");
        assert_eq!(event_data.get_final_data(), "hello");

        // Test multiple data fields (should be joined with newlines)
        event_data.add_data_field("world");
        assert_eq!(event_data.get_final_data(), "hello\nworld");

        // Test data field with trailing newline (should be removed)
        let mut event_data_with_newline = EventData::new();
        event_data_with_newline.add_data_field("test\n");
        assert_eq!(event_data_with_newline.get_final_data(), "test");
    }

    #[test]
    fn test_sse_event_creation() {
        let mut event_data = EventData::new();
        event_data.add_data_field("line1");
        event_data.add_data_field("line2");

        let result = Option::<SSE>::try_from(event_data);
        assert!(result.is_ok());

        if let Ok(Some(SSE::Event(event))) = result {
            assert_eq!(event.data, "line1\nline2");
            assert_eq!(event.event_type, "message"); // default event type
        } else {
            panic!("Expected valid SSE event");
        }
    }

    #[test]
    fn test_empty_event_data() {
        let event_data = EventData::new();
        let result = Option::<SSE>::try_from(event_data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidEvent));
    }
}
